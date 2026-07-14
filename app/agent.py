"""
app.agent
~~~~~~~~~
LangGraph-powered security orchestration graphs.

This module exposes two compiled graphs:

1. ``build_agent()``  — ReAct (Reason + Act) LLM agent used by ``POST /run``.
   Drives a free-form natural-language security investigation using tools.

2. ``build_analysis_graph()``  — Deterministic three-node analysis pipeline
   used by ``POST /analyze``.  Processes a ``NetworkScan`` through a fixed
   sequence of specialist nodes and returns an ``AttackPathReport``.

ReAct graph (build_agent)
--------------------------

    ┌─────────┐     ┌──────────────┐
    │  START  │────▶│  call_model  │◀──┐
    └─────────┘     └──────┬───────┘   │
                           │           │ (tool calls pending)
                    ┌──────▼───────┐   │
                    │  call_tools  │───┘
                    └──────┬───────┘
                           │ (no more tool calls)
                    ┌──────▼───────┐
                    │     END      │
                    └──────────────┘

Analysis graph (build_analysis_graph)
--------------------------------------

    ┌─────────┐     ┌───────────────┐     ┌────────────────────┐     ┌─────────────────┐     ┌─────┐
    │  START  │────▶│ analyzer_node │────▶│ tool_interface_node│────▶│ summarizer_node │────▶│ END │
    └─────────┘     └───────────────┘     └────────────────────┘     └─────────────────┘     └─────┘

    State carried:  scan ──▶ identified_vulnerabilities ──▶ attack_paths ──▶ report

Usage
-----
    from app.agent import build_agent, build_analysis_graph

    # ReAct agent
    agent = build_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})

    # Analysis pipeline
    graph = build_analysis_graph()
    result = await graph.ainvoke({"scan": scan_payload})
    report: AttackPathReport = result["report"]
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Annotated, Any, Sequence

import datetime as dt

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.schemas import (
    AttackPath,
    AttackPathReport,
    AttackStep,
    AttackTechnique,
    NetworkScan,
    PortState,
    SeverityLevel,
    Vulnerability,
)
from app.tools import SENTINEL_TOOLS

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ReAct agent state + graph  (used by POST /run)
# ===========================================================================


class AgentState(TypedDict):
    """
    Mutable state carried through every node of the ReAct graph.

    ``messages`` uses LangGraph's ``add_messages`` reducer so that each node
    *appends* to the conversation history rather than replacing it wholesale.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _make_llm(model_name: str, temperature: float) -> ChatOpenAI:
    """Construct the language model client, bound to the sentinel tool set."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy_offline_key"),
    ).bind_tools(SENTINEL_TOOLS)


def _should_continue(state: AgentState) -> str:
    """
    Conditional edge: route back to ``call_tools`` if the last message
    contains pending tool calls, otherwise terminate at END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.debug("Routing to call_tools (%d calls)", len(last_message.tool_calls))
        return "call_tools"
    logger.debug("No pending tool calls — routing to END")
    return END


def build_agent(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
) -> StateGraph:
    """
    Compile and return the sentinel-core ReAct LangGraph agent.

    Parameters
    ----------
    model_name:
        OpenAI model identifier.  Defaults to ``gpt-4o`` for best
        function-calling reliability.
    temperature:
        Sampling temperature.  Keep at ``0.0`` for deterministic,
        security-critical decisions.

    Returns
    -------
    A compiled ``CompiledGraph`` ready for ``.invoke()`` or ``.ainvoke()``.
    """
    llm = _make_llm(model_name=model_name, temperature=temperature)
    tool_node = ToolNode(tools=SENTINEL_TOOLS)

    def call_model(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        logger.info("call_model — message count=%d", len(state["messages"]))
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", tool_node)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        _should_continue,
        {"call_tools": "call_tools", END: END},
    )
    graph.add_edge("call_tools", "call_model")

    compiled = graph.compile()
    logger.info("ReAct agent compiled with %d tools", len(SENTINEL_TOOLS))
    return compiled


# ===========================================================================
# 2. Analysis pipeline — module-level constants
# ===========================================================================

# Rule table used by analyzer_node.
# Each entry maps a port number to a tuple of:
#   (title, vuln_id_prefix, severity, service_label, description, remediation)
#
# These rules fire when a port is found in PortState.OPEN during the scan,
# and no CVE-identified vulnerability is already attached to that port.
# The identifiers use the "SENTINEL-PORT-XXX" namespace to distinguish
# them from real CVEs ingested from the scan payload.
_RISKY_PORT_RULES: dict[int, tuple[str, str, SeverityLevel, str, str, str]] = {
    21: (
        "FTP Service Exposed",
        "SENTINEL-PORT-021",
        SeverityLevel.MEDIUM,
        "ftp",
        (
            "Port 21 (FTP) is open. FTP transmits credentials and data in plaintext, "
            "making it trivially interceptable on any network path the attacker can observe. "
            "It also lacks integrity protection, allowing in-transit data tampering."
        ),
        "Disable FTP and migrate to SFTP (SSH File Transfer Protocol) or FTPS. "
        "If FTP cannot be removed, restrict access via firewall to trusted source IPs only.",
    ),
    22: (
        "SSH Service Internet-Exposed",
        "SENTINEL-PORT-022",
        SeverityLevel.LOW,
        "ssh",
        (
            "Port 22 (SSH) is reachable. While SSH itself is cryptographically sound, "
            "an internet-exposed SSH service expands the attack surface through "
            "brute-force attacks, weak key management, and exploitation of version-specific bugs."
        ),
        "Restrict SSH access to a VPN or bastion host. Disable password authentication "
        "and require key-based auth. Keep OpenSSH patched to the latest stable release.",
    ),
    23: (
        "Telnet Service Exposed — Plaintext Remote Access",
        "SENTINEL-PORT-023",
        SeverityLevel.CRITICAL,
        "telnet",
        (
            "Port 23 (Telnet) is open. Telnet provides interactive remote shell access "
            "with zero encryption — every keystroke, including passwords, is visible on "
            "the wire. This is universally considered a critical finding on any network."
        ),
        "Disable Telnet immediately. Replace with SSH. Audit all accounts that may have "
        "transmitted credentials over Telnet and rotate them.",
    ),
    445: (
        "SMB Service Exposed — High-Risk Lateral Movement Vector",
        "SENTINEL-PORT-445",
        SeverityLevel.HIGH,
        "smb",
        (
            "Port 445 (SMB — Server Message Block) is open. SMB has been the vector for "
            "numerous critical exploits including EternalBlue (MS17-010 / WannaCry), "
            "PrintNightmare, and Pass-the-Hash attacks. An exposed SMB port, especially "
            "reachable from untrusted networks, represents a high-priority lateral movement "
            "and remote code execution risk."
        ),
        "Block port 445 at the network perimeter immediately. Apply all Microsoft security "
        "patches (prioritise MS17-010). Disable SMBv1 on all hosts via Group Policy. "
        "Segment SMB traffic to dedicated VLANs with host-based firewall rules.",
    ),
    1433: (
        "MSSQL Server Port Exposed",
        "SENTINEL-PORT-1433",
        SeverityLevel.HIGH,
        "ms-sql-s",
        (
            "Port 1433 (Microsoft SQL Server) is reachable. Direct database access from "
            "untrusted networks enables brute-force of SA/sysadmin credentials, SQL injection "
            "escalation to OS command execution (xp_cmdshell), and data exfiltration."
        ),
        "Firewall port 1433 to allow only application-tier source IPs. Disable the SA account. "
        "Disable xp_cmdshell. Enforce strong, unique SQL Server service account passwords.",
    ),
    3306: (
        "MySQL / MariaDB Port Exposed",
        "SENTINEL-PORT-3306",
        SeverityLevel.HIGH,
        "mysql",
        (
            "Port 3306 (MySQL/MariaDB) is reachable from the network. Exposed database ports "
            "enable credential brute-force, unauthenticated access if misconfigured, and "
            "direct data exfiltration without application-layer controls."
        ),
        "Bind MySQL to 127.0.0.1 or restrict via firewall to application-server IPs only. "
        "Rotate the root and all application database passwords. Review GRANT privileges.",
    ),
    3389: (
        "RDP Service Exposed — Remote Desktop Attack Surface",
        "SENTINEL-PORT-3389",
        SeverityLevel.HIGH,
        "rdp",
        (
            "Port 3389 (Remote Desktop Protocol) is open. RDP is one of the most commonly "
            "exploited initial-access vectors. Risks include BlueKeep (CVE-2019-0708), "
            "DejaBlue, credential brute-force, Pass-the-Hash, and ransomware deployment "
            "after gaining a foothold."
        ),
        "Restrict RDP to VPN / jump-host access only. Apply all Windows RDP patches. "
        "Enable Network Level Authentication (NLA). Enforce account lockout policies. "
        "Consider disabling RDP and using a privileged access workstation model.",
    ),
    5900: (
        "VNC Service Exposed — Unencrypted Remote Access",
        "SENTINEL-PORT-5900",
        SeverityLevel.HIGH,
        "vnc",
        (
            "Port 5900 (VNC) is open. Many VNC implementations transmit video frames and "
            "input events with weak or no encryption. VNC is a common target for "
            "credential brute-force and unauthenticated access when misconfigured."
        ),
        "Disable VNC if not operationally required. If required, tunnel it exclusively "
        "over SSH and enforce a strong VNC password. Firewall the port at the perimeter.",
    ),
}


class GraphState(TypedDict):
    """
    Shared, mutable state threaded through the three analysis nodes.

    Each node reads only the fields it needs and writes only the fields
    it is responsible for — this makes every node independently testable.

    Fields
    ------
    scan:
        The raw ``NetworkScan`` payload supplied by the caller.
        Written once at graph entry; treated as read-only by all nodes.

    identified_vulnerabilities:
        Populated by ``analyzer_node``.  The flattened, de-duplicated list
        of ``Vulnerability`` objects discovered via two mechanisms:
        (1) CVE objects already present in ``scan.hosts[*].vulnerabilities``;
        (2) port-rule detections fired by ``_RISKY_PORT_RULES``.

    analyzer_reasoning:
        Populated by ``analyzer_node``.  A human-readable string that
        explains *why* each vulnerability was flagged — produced by a
        LangChain chain backed by ChatOpenAI, with a deterministic offline
        fallback when no API key is present.

    tool_outputs:
        Populated by ``tool_interface_node``. Holds the results of
        simulated or real tool executions against the identified vulnerabilities.

    attack_paths:
        Populated by ``tool_interface_node``.  Each entry is an ``AttackPath``
        assembled by simulating how an adversary would chain the identified
        vulnerabilities across the scan's host topology.

    report:
        Populated by ``summarizer_node``.  The final ``AttackPathReport``
        ready to be serialised and returned to the caller.

    node_log:
        An append-only audit trail.  Each node pushes a short status string
        so that callers can trace execution without reading the full state.
    """

    scan: NetworkScan
    identified_vulnerabilities: list[Vulnerability]
    analyzer_reasoning: str
    tool_outputs: list[dict[str, Any]]
    attack_paths: list[AttackPath]
    report: AttackPathReport | None
    node_log: list[str]


# ===========================================================================
# 3a. Analyzer node helpers
# ===========================================================================


_ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert security analyst embedded inside an air-gapped "
            "security orchestration engine.  Your task is to produce a concise, "
            "technical reasoning paragraph that explains the vulnerabilities "
            "identified in a network scan.  Prioritize vulnerabilities where "
            "the CVSS score is > 8.0 or where an exploit is readily available. "
            "Be specific: reference port numbers, service names, CVSS scores, "
            "and CVE IDs where available.  Do NOT make up CVEs that were not "
            "provided.  Do NOT recommend actions — your output feeds into a "
            "downstream remediation node.  Limit output to 5 sentences.",
        ),
        (
            "human",
            "Scan ID: {scan_id}\n"
            "Hosts scanned: {host_count}\n"
            "Hosts online: {hosts_up}\n"
            "Vulnerabilities identified ({vuln_count} total):\n{vuln_summary}\n\n"
            "Port-based detections:\n{port_findings}\n\n"
            "Produce a technical reasoning paragraph for this scan.",
        ),
    ]
)


def _build_analyzer_reasoning(
    scan: "NetworkScan",
    vulns: list["Vulnerability"],
    port_findings: list[str],
) -> str:
    """
    Produce a human-readable reasoning string that explains why each
    vulnerability was flagged.

    Attempts to call ChatOpenAI using the ``OPENAI_API_KEY`` environment
    variable.  If the key is absent or the call fails, falls back to a
    deterministic, template-rendered string so the graph never blocks on
    LLM availability.

    Parameters
    ----------
    scan:
        The ``NetworkScan`` being analysed.
    vulns:
        All vulnerabilities already identified (CVE-sourced + port-rule).
    port_findings:
        Human-readable strings describing each port-rule detection, e.g.
        ``["Port 445 (SMB) open on prod-web-01"]``.

    Returns
    -------
    A reasoning string suitable for storage in ``GraphState.analyzer_reasoning``.
    """
    vuln_summary_lines = [
        f"  - [{v.severity.value.upper()}] {v.vuln_id}: {v.title}"
        + (f" (CVSS: {v.cvss_score})" if getattr(v, "cvss_score", None) is not None else "")
        + (f" (port {v.affected_port})" if v.affected_port else "")
        + (" [EXPLOIT AVAILABLE]" if getattr(v, "exploit_available", False) else "")
        for v in vulns
    ]
    vuln_summary = "\n".join(vuln_summary_lines) if vuln_summary_lines else "  (none)"
    port_findings_str = "\n".join(f"  - {f}" for f in port_findings) if port_findings else "  (none)"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            chain = _ANALYZER_PROMPT | ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=api_key,
            ) | StrOutputParser()

            reasoning = chain.invoke(
                {
                    "scan_id": scan.scan_id,
                    "host_count": len(scan.hosts),
                    "hosts_up": sum(1 for h in scan.hosts if h.is_up),
                    "vuln_count": len(vulns),
                    "vuln_summary": vuln_summary,
                    "port_findings": port_findings_str,
                }
            )
            logger.debug("[analyzer_node] LLM reasoning produced (%d chars)", len(reasoning))
            return reasoning

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[analyzer_node] LLM reasoning call failed (%s) — using offline fallback", exc
            )

    # --- Offline / fallback reasoning -----------------------------------
    severity_counts: dict[str, int] = {}
    for v in vulns:
        severity_counts[v.severity.value] = severity_counts.get(v.severity.value, 0) + 1

    sev_summary = ", ".join(
        f"{count} {sev}" for sev, count in sorted(severity_counts.items())
    ) or "none"

    return (
        f"[OFFLINE FALLBACK — OPENAI_API_KEY not set or LLM call failed] "
        f"Scan '{scan.scan_id}' covered {len(scan.hosts)} host(s). "
        f"{len(vulns)} vulnerability/ies were identified ({sev_summary}). "
        + (
            f"Port-based detections: {'; '.join(port_findings)}. "
            if port_findings
            else "No additional port-based detections. "
        )
        + "Downstream nodes will use identified_vulnerabilities for path analysis."
    )


# ===========================================================================
# 3b. Analysis pipeline nodes
# ===========================================================================


def analyzer_node(state: GraphState) -> dict:
    """
    **Node 1 — analyzer_node**

    Responsibility
    --------------
    Inspect the ``NetworkScan`` inside the state and produce a ranked list of
    ``Vulnerability`` objects using two complementary detection mechanisms:

    1. **CVE pass-through** — any ``Vulnerability`` already attached to a
       ``ScannedHost`` in ``scan.hosts`` is collected, de-duplicated, and
       forwarded as-is.  These come from scanner plugins (Nessus, OpenVAS, etc.)
       and carry real CVE IDs and CVSS scores.

    2. **Port-rule detection** — each open port on every live host is checked
       against ``_RISKY_PORT_RULES``.  If a match is found and no CVE already
       covers that port, a synthetic ``Vulnerability`` is synthesised with the
       ``SENTINEL-PORT-XXX`` ID namespace.  This catches misconfigurations that
       don't have a CVE (e.g. intentionally open Telnet) but are still
       high-risk.

    After collecting all vulnerabilities, the node calls ``_build_analyzer_reasoning``
    to produce a reasoning string.  This function uses a LangChain chain backed
    by ``gpt-4o-mini`` when an API key is available, and falls back to a
    deterministic template string otherwise.

    Input state fields read
    -----------------------
    - ``scan``  (``NetworkScan``)

    Output state fields written
    ---------------------------
    - ``identified_vulnerabilities``  (``list[Vulnerability]``)
    - ``analyzer_reasoning``  (``str``)
    - ``node_log``  (appended)
    """
    scan: NetworkScan = state["scan"]
    log: list[str] = list(state.get("node_log", []))

    logger.info(
        "[analyzer_node] Processing scan_id=%s — %d host(s), %d target port(s)",
        scan.scan_id,
        len(scan.hosts),
        len(scan.target_ports),
    )

    # -----------------------------------------------------------------------
    # Pass 1 — collect CVE-sourced vulnerabilities from the scan payload
    # -----------------------------------------------------------------------
    all_vulns: list[Vulnerability] = []
    seen_ids: set[str] = set()
    # Track which ports already have a CVE so port-rules don't double-count
    cve_covered_ports: set[int] = set()

    for host in scan.hosts:
        if not host.is_up:
            continue
        for vuln in host.vulnerabilities:
            if vuln.vuln_id not in seen_ids:
                seen_ids.add(vuln.vuln_id)
                all_vulns.append(vuln)
            if vuln.affected_port is not None:
                cve_covered_ports.add(vuln.affected_port)

    logger.info(
        "[analyzer_node] Pass 1 complete — %d CVE-sourced vulnerability/ies",
        len(all_vulns),
    )

    # -----------------------------------------------------------------------
    # Pass 2 — port-rule detection
    # -----------------------------------------------------------------------
    port_findings: list[str] = []  # human-readable strings fed to the LLM

    for host in scan.hosts:
        if not host.is_up:
            continue

        host_label = host.address.hostname or str(host.address.ipv4 or host.address.ipv6 or "unknown")

        for port_entry in host.open_ports:
            if port_entry.state != PortState.OPEN:
                continue
            if port_entry.port in cve_covered_ports:
                # A CVE-level finding already covers this port — skip rule
                continue
            if port_entry.port not in _RISKY_PORT_RULES:
                continue

            title, vuln_id_prefix, severity, service_label, description, remediation = (
                _RISKY_PORT_RULES[port_entry.port]
            )

            # Unique ID per (rule, host) so the same port on two hosts creates
            # two separate findings rather than being silently de-duplicated.
            host_key = str(host.address.ipv4 or host.address.ipv6 or host_label)
            synthetic_id = f"{vuln_id_prefix}-{host_key}"

            if synthetic_id in seen_ids:
                continue
            seen_ids.add(synthetic_id)

            vuln = Vulnerability(
                vuln_id=synthetic_id,
                title=title,
                description=description,
                severity=severity,
                cvss_score=None,       # No CVSS — rule-based, not CVE-backed
                cvss_vector=None,
                cvss_version=None,
                cwe_ids=[],
                affected_port=port_entry.port,
                affected_service=port_entry.service_name or port_entry.service_version or service_label,
                exploit_available=severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH),
                patch_available=True,  # All rule-based findings have known mitigations
                remediation=remediation,
                references=[],
            )
            all_vulns.append(vuln)

            finding_str = (
                f"Port {port_entry.port}/{port_entry.protocol.value} "
                f"({port_entry.service_name or service_label}) "
                f"OPEN on {host_label} — rule {synthetic_id} fired"
            )
            port_findings.append(finding_str)
            logger.info("[analyzer_node] Port-rule: %s", finding_str)

    logger.info(
        "[analyzer_node] Pass 2 complete — %d port-rule detection(s)",
        len(port_findings),
    )

    # -----------------------------------------------------------------------
    # Sort: Exploit available → CVSS score (desc) → Critical → High → Medium → Low → Info
    # -----------------------------------------------------------------------
    _SEVERITY_RANK: dict[SeverityLevel, int] = {
        SeverityLevel.CRITICAL: 0,
        SeverityLevel.HIGH: 1,
        SeverityLevel.MEDIUM: 2,
        SeverityLevel.LOW: 3,
        SeverityLevel.INFO: 4,
    }
    
    def vuln_sort_key(v: Vulnerability):
        exploit_penalty = 0 if getattr(v, "exploit_available", False) else 1
        cvss_score = getattr(v, "cvss_score", None)
        cvss_penalty = -cvss_score if cvss_score is not None else 0
        severity_penalty = _SEVERITY_RANK.get(v.severity, 99)
        return (exploit_penalty, cvss_penalty, severity_penalty)

    all_vulns.sort(key=vuln_sort_key)

    # -----------------------------------------------------------------------
    # Build reasoning string (LLM-backed with offline fallback)
    # -----------------------------------------------------------------------
    reasoning = _build_analyzer_reasoning(scan, all_vulns, port_findings)

    log_entry = (
        f"analyzer_node: {len(all_vulns)} vulnerability/ies identified "
        f"({len(all_vulns) - len(port_findings)} CVE-sourced, "
        f"{len(port_findings)} port-rule); "
        f"LLM reasoning {'generated' if os.environ.get('OPENAI_API_KEY') else 'offline fallback'}"
    )
    log.append(log_entry)
    logger.info("[analyzer_node] %s", log_entry)

    return {
        "identified_vulnerabilities": all_vulns,
        "analyzer_reasoning": reasoning,
        "node_log": log,
    }


def tool_interface_node(state: GraphState) -> dict:
    """
    **Node 2 — tool_interface_node**

    Responsibility
    --------------
    Simulate the adversarial 'attack' by constructing ``AttackPath`` objects
    that chain the identified vulnerabilities across the scan's host topology.

    This node represents the interface boundary to the real tool layer
    (``tools.py``).  When the full implementation lands, this node will:

    - Call ``tools.search_siem_logs`` to correlate scan findings with live alerts.
    - Call ``tools.lookup_ioc`` to check whether any discovered IPs or hashes
      are present in threat-intelligence feeds.
    - Query the MITRE ATT&CK STIX bundle to map CVEs → technique IDs.
    - Use a graph-reachability algorithm (DFS / BFS) to enumerate feasible
      attack paths between hosts, respecting firewall topology inferred from
      closed/filtered port evidence.

    Input state fields read
    -----------------------
    - ``scan``  (``NetworkScan``)
    - ``identified_vulnerabilities``  (``list[Vulnerability]``)

    Output state fields written
    ---------------------------
    - ``tool_outputs``  (``list[dict]``)
    - ``attack_paths``  (``list[AttackPath]``)
    - ``node_log``  (appended)

    Implementation note
    -------------------
    The stub below creates synthetic tool outputs and one synthetic ``AttackPath``
    per critical/high vulnerability so the downstream summarizer always receives
    a non-empty structure. Real tool execution and path enumeration are a TODO.
    """
    scan: NetworkScan = state["scan"]
    vulns: list[Vulnerability] = state.get("identified_vulnerabilities", [])
    log: list[str] = list(state.get("node_log", []))
    tool_outputs: list[dict[str, Any]] = list(state.get("tool_outputs", []))

    logger.info(
        "[tool_interface_node] Simulating tool execution and building attack paths from %d vulnerability/ies",
        len(vulns),
    )

    # --- MVP: Simulate 'Tool Execution' and synthesise paths ---
    paths: list[AttackPath] = []

    actionable = [
        v for v in vulns
        if v.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH) or getattr(v, "exploit_available", False)
    ]
    
    # Map to track which hosts have already been attacked to prioritize highest severity per host
    attacked_hosts: set[str] = set()

    for vuln in actionable:
        # 1. Simulate tool execution for the MVP
        simulated_output = {
            "vuln_id": vuln.vuln_id,
            "tool_status": "success",
            "action": f"Execute Nmap service version detection & Exploit validation for {vuln.vuln_id}",
            "output": f"Version detected: {vuln.affected_service or 'Unknown'}, Exploit validated: {vuln.exploit_available}"
        }
        tool_outputs.append(simulated_output)
        logger.info("[tool_interface_node] Simulated tool execution: %s", simulated_output)

        # 2. Find the first host that carries this vulnerability and hasn't had a path built yet
        origin_host = None
        for h in scan.hosts:
            host_key = str(h.address.ipv4 or h.address.ipv6 or h.address.hostname)
            if host_key not in attacked_hosts and any(v.vuln_id == vuln.vuln_id for v in h.vulnerabilities):
                origin_host = h
                attacked_hosts.add(host_key)
                break
                
        if origin_host is None:
            continue

        # Build a single-step path representing initial exploitation
        step = AttackStep(
            step_index=0,
            technique_id="T1190",                        # Exploit Public-Facing Application
            technique_name="Exploit Public-Facing Application",
            tactic=AttackTechnique.INITIAL_ACCESS,
            source_address=None,                         # External adversary — no source addr
            destination_address=origin_host.address,
            destination_port=vuln.affected_port,
            exploited_vulnerability=vuln,
            preconditions=[
                f"Port {vuln.affected_port} reachable from internet"
                if vuln.affected_port
                else "Service reachable from internet",
                f"{vuln.vuln_id} unpatched on target",
            ],
            postconditions=[
                f"Remote code execution on {origin_host.address.hostname or str(origin_host.address.ipv4)}",
                "Initial foothold established",
            ],
            # Stub likelihood: use CVSS score normalised to 0–1 if available
            likelihood=round(vuln.cvss_score / 10.0, 2) if vuln.cvss_score is not None else None,
            detection_coverage=None,   # TODO: derive from SIEM rule coverage
            notes="Stub path — tool_interface_node placeholder",
        )

        path = AttackPath(
            path_id=str(uuid.uuid4()),
            path_label=(
                f"[STUB] External → {origin_host.address.hostname or str(origin_host.address.ipv4)}"
                f" via {vuln.vuln_id}"
            ),
            steps=[step],
            entry_point=origin_host.address,
            target_asset=origin_host.address,   # TODO: propagate to crown-jewel target
            objective=f"Remote code execution via {vuln.vuln_id} ({vuln.title})",
            overall_likelihood=step.likelihood,
            overall_severity=vuln.severity,
            hop_count=1,
            mitre_tactics=[AttackTechnique.INITIAL_ACCESS],
        )
        paths.append(path)

    log.append(
        f"tool_interface_node: simulated {len(tool_outputs)} tool execution(s), "
        f"synthesised {len(paths)} attack path(s) "
        f"from {len(actionable)} critical/high vulnerability/ies"
    )
    logger.info("[tool_interface_node] Synthesised %d attack path(s)", len(paths))

    return {
        "tool_outputs": tool_outputs,
        "attack_paths": paths,
        "node_log": log,
    }


# ===========================================================================
# 3c. Summarizer node helpers
# ===========================================================================

_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Principal Security Architect presenting an Attack Path "
            "Report to an executive board. You are summarising a network scan. "
            "Draft a 1-2 paragraph executive summary that highlights the total "
            "vulnerabilities found, the number of critical attack paths, the "
            "most pressing tool execution findings, and the overall business risk. "
            "Do NOT include greeting/closing, just the summary text.",
        ),
        (
            "human",
            "Scan ID: {scan_id}\n"
            "Vulnerabilities: {vuln_count}\n"
            "Attack paths mapped: {path_count} (Critical: {critical_count})\n"
            "Assets at risk: {asset_count}\n\n"
            "Analyzer Reasoning:\n{analyzer_reasoning}\n\n"
            "Tool Outputs:\n{tool_outputs_str}\n\n"
            "Draft the executive summary.",
        ),
    ]
)

def _build_executive_summary(
    scan: "NetworkScan",
    vuln_count: int,
    path_count: int,
    critical_count: int,
    asset_count: int,
    analyzer_reasoning: str,
    tool_outputs: list[dict[str, Any]],
) -> str:
    """Produce the executive summary prose via LLM with a graceful fallback."""
    tool_outputs_str = "\n".join(
        f"  - Action: {t.get('action', 'Unknown')} -> Status: {t.get('tool_status')} -> Output: {t.get('output')}"
        for t in tool_outputs
    ) if tool_outputs else "  (none)"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            chain = _SUMMARIZER_PROMPT | ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=api_key,
            ) | StrOutputParser()

            summary = chain.invoke(
                {
                    "scan_id": scan.scan_id,
                    "vuln_count": vuln_count,
                    "path_count": path_count,
                    "critical_count": critical_count,
                    "asset_count": asset_count,
                    "analyzer_reasoning": analyzer_reasoning,
                    "tool_outputs_str": tool_outputs_str,
                }
            )
            return summary
        except Exception as exc:  # noqa: BLE001
            logger.warning("[summarizer_node] LLM summary failed (%s) — using offline fallback", exc)

    return (
        f"Sentinel-Core analysis of scan '{scan.scan_id}' identified "
        f"{vuln_count} vulnerability/ies across {len(scan.hosts)} host(s). "
        f"{path_count} exploitable attack path(s) were mapped, "
        f"of which {critical_count} are rated CRITICAL. "
        f"Immediate remediation is required for {asset_count} asset(s) at risk. "
        f"(Note: this summary was generated via offline fallback.)"
    )

def summarizer_node(state: GraphState) -> dict:
    """
    **Node 3 — summarizer_node**

    Responsibility
    --------------
    Assemble the final ``AttackPathReport`` from the analysis state produced
    by the two upstream nodes.

    Input state fields read
    -----------------------
    - ``scan``                      (``NetworkScan``)
    - ``identified_vulnerabilities``  (``list[Vulnerability]``)
    - ``analyzer_reasoning``        (``str``)
    - ``tool_outputs``              (``list[dict]``)
    - ``attack_paths``              (``list[AttackPath]``)

    Output state fields written
    ---------------------------
    - ``report``  (``AttackPathReport``)
    - ``node_log``  (appended)
    """
    scan: NetworkScan = state["scan"]
    vulns: list[Vulnerability] = state.get("identified_vulnerabilities", [])
    analyzer_reasoning: str = state.get("analyzer_reasoning", "")
    tool_outputs: list[dict[str, Any]] = state.get("tool_outputs", [])
    paths: list[AttackPath] = state.get("attack_paths", [])
    log: list[str] = list(state.get("node_log", []))

    logger.info(
        "[summarizer_node] Assembling report for scan_id=%s "
        "(%d path(s), %d vuln(s), %d tool output(s))",
        scan.scan_id,
        len(paths),
        len(vulns),
        len(tool_outputs),
    )

    # --- Derive aggregate stats ------------------------------------------
    critical_count = sum(
        1 for p in paths if p.overall_severity == SeverityLevel.CRITICAL
    )

    # Unique target assets across all paths
    seen_assets: set[str] = set()
    assets_at_risk = []
    for path in paths:
        key = str(path.target_asset.ipv4 or path.target_asset.ipv6 or path.target_asset.hostname)
        if key not in seen_assets:
            seen_assets.add(key)
            assets_at_risk.append(path.target_asset)

    # --- Aggregate recommendations from vulnerabilities + tool outputs -----
    recommendations: list[str] = []
    for path in paths:
        for step in path.steps:
            if step.exploited_vulnerability and step.exploited_vulnerability.remediation:
                recommendations.append(
                    f"[{step.exploited_vulnerability.severity.value.upper()}] "
                    f"{step.exploited_vulnerability.remediation} "
                    f"(breaks path: {path.path_label})"
                )

    for output in tool_outputs:
        if output.get("tool_status") == "success":
            recommendations.append(
                f"[TOOL FINDING] Result of '{output.get('action')}': {output.get('output')} "
                f"— Review system configuration for {output.get('vuln_id')}."
            )

    # --- Build executive summary -----------------------------------------
    executive_summary = _build_executive_summary(
        scan=scan,
        vuln_count=len(vulns),
        path_count=len(paths),
        critical_count=critical_count,
        asset_count=len(assets_at_risk),
        analyzer_reasoning=analyzer_reasoning,
        tool_outputs=tool_outputs,
    )

    analyst_notes_content = [
        "== Agent Node Log ==",
        *log,
        "",
        "== Analyzer Reasoning ==",
        analyzer_reasoning,
    ]

    report = AttackPathReport(
        report_id=str(uuid.uuid4()),
        report_name=f"Attack Path Report — scan {scan.scan_id}",
        generated_at=dt.datetime.now(tz=dt.timezone.utc),
        generated_by="sentinel-core/analysis-graph v0.1.0",
        source_scan_ids=[scan.scan_id],
        scope_description=(
            f"Hosts in scope: {', '.join(scan.target_cidrs)}. "
            f"Scanner: {scan.scanner_tool or 'unknown'}."
        ),
        attack_paths=sorted(
            paths,
            key=lambda p: p.overall_likelihood if p.overall_likelihood is not None else 0.0,
            reverse=True,
        ),
        total_paths_found=len(paths),
        critical_paths=critical_count,
        assets_at_risk=assets_at_risk,
        recommendations=recommendations,
        executive_summary=executive_summary,
        analyst_notes="\n".join(analyst_notes_content),
    )

    log.append(
        f"summarizer_node: report assembled — "
        f"{len(paths)} path(s), {critical_count} critical"
    )
    logger.info("[summarizer_node] Report assembled (id=%s)", report.report_id)

    return {
        "report": report,
        "node_log": log,
    }


# ===========================================================================
# 4. Analysis graph factory
# ===========================================================================


def build_analysis_graph() -> StateGraph:
    """
    Compile and return the deterministic three-node attack-path analysis graph.

    Graph topology
    --------------
    START → analyzer_node → tool_interface_node → summarizer_node → END

    The graph is stateful: every node receives the full ``GraphState`` and
    returns a partial dict that LangGraph merges back into the shared state
    before routing to the next node.

    Returns
    -------
    A compiled ``CompiledGraph`` ready for ``.invoke()`` or ``.ainvoke()``.

    Example
    -------
    ::

        graph = build_analysis_graph()
        result = await graph.ainvoke({
            "scan": my_network_scan,
            "identified_vulnerabilities": [],
            "analyzer_reasoning": "",
            "tool_outputs": [],
            "attack_paths": [],
            "report": None,
            "node_log": [],
        })
        report: AttackPathReport = result["report"]
    """
    graph: StateGraph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("analyzer_node", analyzer_node)
    graph.add_node("tool_interface_node", tool_interface_node)
    graph.add_node("summarizer_node", summarizer_node)

    # Wire edges — strictly linear; no conditional branching yet
    graph.add_edge(START, "analyzer_node")
    graph.add_edge("analyzer_node", "tool_interface_node")
    graph.add_edge("tool_interface_node", "summarizer_node")
    graph.add_edge("summarizer_node", END)

    compiled = graph.compile()
    logger.info("Analysis graph compiled (3 nodes: analyzer → tool_interface → summarizer)")
    return compiled
