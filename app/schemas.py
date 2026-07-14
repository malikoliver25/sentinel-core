"""
app.schemas
~~~~~~~~~~~
Pydantic v2 data models for request validation, response serialization,
and internal agent state shapes.

Sections
--------
1. Shared enumerations
2. Agent API models  (AgentRequest, AgentResponse, ToolResult)
3. Network scan models  (NetworkAddress, PortEntry, Vulnerability, ScannedHost, NetworkScan)
4. Attack path models   (AttackStep, AttackPath, AttackPathReport)

All models are pure data containers — no methods or business logic.
Field descriptions surface verbatim in the FastAPI / Swagger UI.
"""

from __future__ import annotations

import datetime
from enum import Enum
from ipaddress import IPv4Address, IPv4Network, IPv6Address
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# 1. Shared enumerations
# ===========================================================================


class SeverityLevel(str, Enum):
    """Threat severity classification used across the orchestration engine."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    """Terminal and intermediate states of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Protocol(str, Enum):
    """Layer-4 transport protocol."""

    TCP = "tcp"
    UDP = "udp"
    SCTP = "sctp"
    ICMP = "icmp"


class PortState(str, Enum):
    """Scanner-reported state of a network port."""

    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    OPEN_FILTERED = "open|filtered"
    UNFILTERED = "unfiltered"


class ScanStatus(str, Enum):
    """Lifecycle state of a network scan job."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class AttackTechnique(str, Enum):
    """
    High-level MITRE ATT&CK tactic category represented by an attack step.
    Values mirror ATT&CK tactic IDs for direct cross-reference.
    """

    RECONNAISSANCE = "TA0043"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


# ===========================================================================
# 2. Agent API models
# ===========================================================================


class AgentRequest(BaseModel):
    """Payload submitted to the orchestration engine to initiate an agent run."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language task or alert description for the agent.",
        examples=["Investigate lateral movement detected on host PROD-WEB-01."],
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context (e.g. alert metadata, asset inventory).",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Caller-assessed severity that may influence agent prioritisation.",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum reasoning iterations before the agent is forcibly halted.",
    )

    model_config = {"str_strip_whitespace": True}


class ToolResult(BaseModel):
    """Standardised wrapper returned by every tool in tools.py."""

    tool_name: str = Field(..., description="Name of the tool that produced this result.")
    success: bool = Field(..., description="Whether the tool execution succeeded.")
    data: Any = Field(default=None, description="Structured payload returned by the tool.")
    error: str | None = Field(
        default=None,
        description="Human-readable error message when success is False.",
    )


class AgentResponse(BaseModel):
    """Top-level response envelope returned by the /run endpoint."""

    run_id: str = Field(..., description="Unique identifier for this agent run.")
    status: AgentStatus = Field(..., description="Terminal status of the completed run.")
    summary: str = Field(..., description="Human-readable summary of findings and actions taken.")
    tool_calls: list[ToolResult] = Field(
        default_factory=list,
        description="Ordered list of tool invocations made during the run.",
    )
    iterations_used: int = Field(
        ...,
        ge=0,
        description="Number of reasoning iterations consumed.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional run-level metadata (timing, model used, etc.).",
    )


# ===========================================================================
# 3. Network scan models
# ===========================================================================


class NetworkAddress(BaseModel):
    """
    A fully described network address, supporting both IPv4 and IPv6.
    Exactly one of ``ipv4`` or ``ipv6`` must be supplied.
    """

    ipv4: IPv4Address | None = Field(
        default=None,
        description=(
            "IPv4 address of the host in dotted-decimal notation "
            "(e.g. '192.168.1.10'). Mutually exclusive with ``ipv6``."
        ),
        examples=["192.168.1.10"],
    )
    ipv6: IPv6Address | None = Field(
        default=None,
        description=(
            "IPv6 address of the host in colon-hexadecimal notation "
            "(e.g. '2001:db8::1'). Mutually exclusive with ``ipv4``."
        ),
        examples=["2001:db8::1"],
    )
    subnet: IPv4Network | None = Field(
        default=None,
        description=(
            "CIDR subnet the address belongs to (e.g. '192.168.1.0/24'). "
            "Informational — not validated against the IP value."
        ),
        examples=["192.168.1.0/24"],
    )
    hostname: str | None = Field(
        default=None,
        min_length=1,
        max_length=253,
        description="Reverse-DNS hostname resolved for this address, if available.",
        examples=["prod-web-01.internal"],
    )
    mac_address: str | None = Field(
        default=None,
        description=(
            "MAC address of the network interface in IEEE 802 format "
            "(e.g. 'AA:BB:CC:DD:EE:FF'). Present only for directly reachable hosts."
        ),
        examples=["AA:BB:CC:DD:EE:FF"],
        pattern=r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$",
    )

    @field_validator("ipv4", "ipv6", mode="before")
    @classmethod
    def coerce_ip_string(cls, value: Any) -> Any:
        """Accept plain strings so callers don't need to pre-cast."""
        return value  # Pydantic handles str→IPv4Address coercion natively


class PortEntry(BaseModel):
    """
    A single scanned port on a host, capturing transport-layer details
    and any service fingerprint collected during the scan.
    """

    port: int = Field(
        ...,
        ge=0,
        le=65535,
        description="Port number in the range 0–65535.",
        examples=[443],
    )
    protocol: Protocol = Field(
        ...,
        description="Layer-4 transport protocol over which the port was observed.",
        examples=["tcp"],
    )
    state: PortState = Field(
        ...,
        description=(
            "Scanner-reported port state. ``open`` means a service accepted "
            "the probe; ``filtered`` means a firewall dropped the probe."
        ),
        examples=["open"],
    )
    service_name: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Well-known service name for this port/protocol combination "
            "(e.g. 'https', 'ssh', 'rdp')."
        ),
        examples=["https"],
    )
    service_version: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Version string returned by service-version detection "
            "(e.g. 'Apache httpd 2.4.54'). ``null`` when detection was not run "
            "or the service did not respond."
        ),
        examples=["Apache httpd 2.4.54"],
    )
    banner: str | None = Field(
        default=None,
        max_length=1024,
        description=(
            "Raw service banner grabbed during the scan, truncated to 1 024 characters. "
            "May contain sensitive version or configuration data."
        ),
    )
    tunnel: str | None = Field(
        default=None,
        description=(
            "Tunnelling layer detected above the transport protocol "
            "(e.g. 'ssl', 'tls1.3'). ``null`` when not detected."
        ),
        examples=["ssl"],
    )


class Vulnerability(BaseModel):
    """
    A discrete security vulnerability identified on a scanned host or service.
    Maps to a single CVE (or scanner-proprietary finding) with full CVSS scoring.
    """

    vuln_id: str = Field(
        ...,
        description=(
            "Primary identifier for this vulnerability. Use a CVE ID when available "
            "(e.g. 'CVE-2021-44228'); fall back to the scanner's own plugin/check ID."
        ),
        examples=["CVE-2021-44228"],
    )
    title: str = Field(
        ...,
        max_length=512,
        description="Short, human-readable title of the vulnerability.",
        examples=["Apache Log4j Remote Code Execution (Log4Shell)"],
    )
    description: str = Field(
        ...,
        description=(
            "Detailed description of the vulnerability, its root cause, "
            "and the conditions under which it is exploitable."
        ),
    )
    severity: SeverityLevel = Field(
        ...,
        description=(
            "Qualitative severity rating derived from the CVSS base score: "
            "info (0.0), low (0.1–3.9), medium (4.0–6.9), high (7.0–8.9), "
            "critical (9.0–10.0)."
        ),
    )
    cvss_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description=(
            "CVSS base score in the range 0.0–10.0. ``null`` when CVSS data "
            "is unavailable (e.g. proprietary scanner findings)."
        ),
        examples=[10.0],
    )
    cvss_vector: str | None = Field(
        default=None,
        description=(
            "Full CVSS vector string (CVSS v3.1 preferred). "
            "Example: 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H'."
        ),
        examples=["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"],
    )
    cvss_version: str | None = Field(
        default=None,
        description="CVSS specification version used to compute the score (e.g. '3.1', '4.0').",
        examples=["3.1"],
    )
    cwe_ids: list[str] = Field(
        default_factory=list,
        description=(
            "List of CWE identifiers that describe the weakness class underlying "
            "this vulnerability (e.g. ['CWE-502', 'CWE-917'])."
        ),
        examples=[["CWE-502"]],
    )
    affected_port: int | None = Field(
        default=None,
        ge=0,
        le=65535,
        description=(
            "Port number on which the vulnerable service was detected. "
            "``null`` when the vulnerability is host-level (e.g. OS patch missing)."
        ),
        examples=[8080],
    )
    affected_service: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Name and version of the specific service or component that is vulnerable "
            "(e.g. 'log4j-core 2.14.1'). Complements ``affected_port``."
        ),
        examples=["log4j-core 2.14.1"],
    )
    exploit_available: bool = Field(
        default=False,
        description=(
            "``true`` when a public or commercial exploit is known to exist "
            "for this vulnerability at the time of the scan."
        ),
    )
    patch_available: bool = Field(
        default=False,
        description="``true`` when an official vendor patch or mitigation is publicly available.",
    )
    remediation: str | None = Field(
        default=None,
        description=(
            "Recommended remediation action. Should include specific patch versions, "
            "configuration changes, or compensating controls."
        ),
        examples=["Upgrade log4j-core to >= 2.17.1 or apply vendor mitigations."],
    )
    references: list[str] = Field(
        default_factory=list,
        description=(
            "List of authoritative URLs for further reading "
            "(NVD, vendor advisories, PoC repositories, etc.)."
        ),
        examples=[["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"]],
    )
    first_seen: datetime.datetime | None = Field(
        default=None,
        description="UTC timestamp when this vulnerability was first observed on this asset.",
    )


class ScannedHost(BaseModel):
    """
    Complete scan results for a single network host, aggregating its
    address information, open ports, fingerprinted OS, and discovered vulnerabilities.
    """

    address: NetworkAddress = Field(
        ...,
        description="Primary network address of the scanned host.",
    )
    additional_addresses: list[NetworkAddress] = Field(
        default_factory=list,
        description=(
            "Secondary addresses bound to the same host "
            "(e.g. additional NICs, virtual IPs, IPv6 link-local)."
        ),
    )
    is_up: bool = Field(
        ...,
        description=(
            "``true`` when the host responded to at least one probe during the scan; "
            "``false`` when the host was unreachable or blocked all probes."
        ),
    )
    os_family: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Operating system family as fingerprinted by the scanner "
            "(e.g. 'Windows', 'Linux', 'FreeBSD'). ``null`` when OS detection "
            "was not run or inconclusive."
        ),
        examples=["Linux"],
    )
    os_version: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Specific OS version string as fingerprinted "
            "(e.g. 'Ubuntu 22.04 LTS', 'Windows Server 2019 Datacenter'). "
            "``null`` when unavailable."
        ),
        examples=["Ubuntu 22.04 LTS"],
    )
    os_confidence: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Scanner confidence score for the OS fingerprint (0–100). "
            "Values below 70 should be treated as indicative only."
        ),
        examples=[95],
    )
    open_ports: list[PortEntry] = Field(
        default_factory=list,
        description=(
            "All ports for which the scanner received a response, regardless of state. "
            "Filter by ``state == 'open'`` for actionable results."
        ),
    )
    vulnerabilities: list[Vulnerability] = Field(
        default_factory=list,
        description=(
            "Vulnerabilities identified on this host, ordered by CVSS score descending. "
            "An empty list indicates a clean host or that vulnerability checks were skipped."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Analyst or scanner-assigned tags for grouping and filtering "
            "(e.g. ['internet-facing', 'pci-scope', 'critical-asset'])."
        ),
        examples=[["internet-facing", "pci-scope"]],
    )
    scan_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time in milliseconds taken to scan this individual host.",
    )


class NetworkScan(BaseModel):
    """
    Top-level container for a complete network scan job.

    Captures job identity, scope (target CIDRs and ports), execution metadata,
    and the full list of host results.  Suitable for storage, diff-comparison
    between scan runs, and direct ingestion into the attack-path analyser.
    """

    scan_id: str = Field(
        ...,
        description=(
            "Unique identifier for this scan job. Should be a UUID v4 "
            "generated by the scanning orchestrator."
        ),
        examples=["e3b0c442-98fc-4c14-9afb-1234567890ab"],
    )
    scan_name: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Human-readable label for the scan (e.g. 'Weekly DMZ sweep — 2026-07-14'). "
            "Optional but strongly recommended for audit trails."
        ),
        examples=["Weekly DMZ sweep — 2026-07-14"],
    )
    status: ScanStatus = Field(
        ...,
        description="Current lifecycle state of the scan job.",
    )
    initiated_by: str = Field(
        default="system",
        max_length=256,
        description=(
            "Identity of the actor that triggered the scan. "
            "Use a service-account name for automated scans or a user UPN for manual runs."
        ),
        examples=["svc-scanner@sentinel.internal"],
    )
    target_cidrs: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Must contain at least one entry. "
            "Example: ['10.0.0.0/8', '172.16.0.0/12']."
        ),
        examples=[["10.10.0.0/16", "172.16.42.0/24"]],
    )
    excluded_cidrs: list[str] = Field(
        default_factory=list,
        description=(
            "CIDRs explicitly excluded from the scan despite falling within "
            "``target_cidrs`` (e.g. honeypot ranges or out-of-scope segments)."
        ),
        examples=[["10.10.99.0/24"]],
    )
    target_ports: list[int] = Field(
        default_factory=list,
        description=(
            "Specific port numbers to scan. An empty list means the scanner "
            "used its own default port set (commonly top-1000 ports)."
        ),
        examples=[[22, 80, 443, 3389, 8080, 8443]],
    )
    scan_profile: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Named scan profile or policy applied (e.g. 'stealth', 'full-tcp', "
            "'vuln-check'). Scanner-specific — stored for reproducibility."
        ),
        examples=["full-tcp-with-vuln"],
    )
    started_at: datetime.datetime | None = Field(
        default=None,
        description="UTC timestamp when the scan job began execution.",
    )
    completed_at: datetime.datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp when the scan job finished (success or failure). "
            "``null`` while the scan is still running."
        ),
    )
    total_hosts_scanned: int = Field(
        default=0,
        ge=0,
        description="Count of IP addresses that the scanner attempted to probe.",
    )
    total_hosts_up: int = Field(
        default=0,
        ge=0,
        description="Count of hosts that responded to at least one probe.",
    )
    total_vulnerabilities: int = Field(
        default=0,
        ge=0,
        description="Aggregate count of vulnerabilities found across all scanned hosts.",
    )
    hosts: list[ScannedHost] = Field(
        default_factory=list,
        description=(
            "Detailed results for each host that was probed. "
            "Hosts that were unreachable (``is_up=false``) may be omitted "
            "depending on scanner verbosity settings."
        ),
    )
    scanner_tool: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Name and version of the scanning tool that produced this data "
            "(e.g. 'Nmap 7.94', 'Nessus 10.6.3'). Used for reproducibility and auditing."
        ),
        examples=["Nmap 7.94"],
    )
    raw_output_uri: str | None = Field(
        default=None,
        description=(
            "URI pointing to the scanner's original raw output file "
            "(e.g. an S3 object, GCS URI, or internal file path). "
            "Provides full fidelity beyond this schema's normalisation."
        ),
        examples=["s3://sentinel-scans/2026-07-14/scan-e3b0c442.xml"],
    )


# ===========================================================================
# 4. Attack path models
# ===========================================================================


class AttackStep(BaseModel):
    """
    A single discrete step within an attack path, representing one
    technique an adversary uses to advance toward their objective.

    Each step maps to a MITRE ATT&CK technique and is anchored to a
    specific source and destination asset, enabling graph-based path analysis.
    """

    step_index: int = Field(
        ...,
        ge=0,
        description=(
            "Zero-based ordinal position of this step within the attack path. "
            "Steps must be contiguous and ordered by execution sequence."
        ),
        examples=[0],
    )
    technique_id: str = Field(
        ...,
        description=(
            "MITRE ATT&CK technique or sub-technique ID "
            "(e.g. 'T1190' for Exploit Public-Facing Application, "
            "'T1021.002' for SMB/Windows Admin Shares)."
        ),
        examples=["T1190"],
    )
    technique_name: str = Field(
        ...,
        max_length=256,
        description="Human-readable name of the MITRE ATT&CK technique.",
        examples=["Exploit Public-Facing Application"],
    )
    tactic: AttackTechnique = Field(
        ...,
        description=(
            "ATT&CK tactic category this technique falls under, "
            "expressed as the tactic ID (e.g. 'TA0001' for Initial Access)."
        ),
    )
    source_address: NetworkAddress | None = Field(
        default=None,
        description=(
            "Network address of the asset that initiates or enables this step. "
            "``null`` for the first step when the adversary originates externally."
        ),
    )
    destination_address: NetworkAddress = Field(
        ...,
        description="Network address of the asset targeted by this step.",
    )
    destination_port: int | None = Field(
        default=None,
        ge=0,
        le=65535,
        description=(
            "Port number on the destination asset exploited during this step. "
            "``null`` when the technique is not port-specific."
        ),
        examples=[445],
    )
    exploited_vulnerability: Vulnerability | None = Field(
        default=None,
        description=(
            "Vulnerability leveraged to execute this step, if applicable. "
            "``null`` for steps that rely on misconfigurations or stolen credentials "
            "rather than a discrete CVE."
        ),
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable conditions that must hold true before this step can execute "
            "(e.g. ['attacker has SYSTEM on PROD-WEB-01', 'port 445 open to 10.0.0.0/8'])."
        ),
        examples=[["attacker has valid domain credentials", "SMB reachable from DMZ"]],
    )
    postconditions: list[str] = Field(
        default_factory=list,
        description=(
            "State achieved by the attacker after successfully completing this step "
            "(e.g. ['lateral access to DC-01', 'SYSTEM shell on PROD-DB-01'])."
        ),
        examples=[["lateral access to DC-01 via SMB"]],
    )
    likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated probability (0.0–1.0) that an attacker can successfully execute "
            "this step given the current network and security posture. "
            "``null`` when not quantified."
        ),
        examples=[0.87],
    )
    detection_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Estimated probability (0.0–1.0) that existing detective controls "
            "(SIEM rules, EDR, NDR) would surface this step. "
            "Low values indicate detection gaps."
        ),
        examples=[0.30],
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Analyst commentary on this step — contextual nuance, "
            "compensating control observations, or remediation notes."
        ),
    )


class AttackPath(BaseModel):
    """
    An ordered sequence of ``AttackStep`` objects that describes a complete
    adversarial trajectory from initial access to a terminal objective.

    A single ``AttackPathReport`` may contain multiple ``AttackPath`` instances
    when the analyser discovers several viable routes to the same target.
    """

    path_id: str = Field(
        ...,
        description=(
            "Unique identifier for this path within its parent report. "
            "Should be a UUID v4."
        ),
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )
    path_label: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Short descriptive label summarising the path "
            "(e.g. 'Internet → DMZ web server → internal DB via Log4Shell + SMB'). "
            "Auto-generated or analyst-supplied."
        ),
    )
    steps: list[AttackStep] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of attack steps constituting this path. "
            "Must contain at least one step. Steps are executed in index order."
        ),
    )
    entry_point: NetworkAddress = Field(
        ...,
        description=(
            "Network address of the first host the adversary gains a foothold on. "
            "For external attacks, this is the first internet-reachable asset compromised."
        ),
    )
    target_asset: NetworkAddress = Field(
        ...,
        description=(
            "Network address of the high-value asset the path is directed toward — "
            "the 'crown jewel' that defines the path's objective."
        ),
    )
    objective: str = Field(
        ...,
        max_length=512,
        description=(
            "Human-readable statement of the adversarial objective achieved "
            "at the end of this path "
            "(e.g. 'Exfiltrate customer PII from PROD-DB-01', "
            "'Ransomware deployment across domain')."
        ),
        examples=["Exfiltrate customer PII from PROD-DB-01"],
    )
    overall_likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate probability that the entire path succeeds end-to-end. "
            "Typically the product of per-step likelihoods, adjusted for correlation. "
            "``null`` when not quantified."
        ),
        examples=[0.62],
    )
    overall_severity: SeverityLevel = Field(
        ...,
        description=(
            "Highest severity of any vulnerability exploited along this path, "
            "used as a conservative upper-bound risk rating for the path as a whole."
        ),
    )
    hop_count: int = Field(
        ...,
        ge=1,
        description=(
            "Number of distinct network hops (asset pivots) the attacker must traverse. "
            "Equals ``len(steps)`` for simple linear paths."
        ),
        examples=[3],
    )
    mitre_tactics: list[AttackTechnique] = Field(
        default_factory=list,
        description=(
            "Deduplicated, ordered list of ATT&CK tactic IDs covered by this path, "
            "useful for filtering and tactic-level coverage analysis."
        ),
    )


class AttackPathReport(BaseModel):
    """
    Top-level container for an attack-path analysis report.

    Produced by the sentinel-core engine after ingesting one or more
    ``NetworkScan`` results and running graph-based reachability and
    exploitability analysis across the target environment.
    """

    report_id: str = Field(
        ...,
        description=(
            "Unique identifier for this report. Should be a UUID v4, "
            "and should be stable across incremental report updates."
        ),
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    report_name: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Human-readable report title "
            "(e.g. 'Q3 2026 Attack Path Assessment — Production Environment')."
        ),
        examples=["Q3 2026 Attack Path Assessment — Production Environment"],
    )
    generated_at: datetime.datetime = Field(
        ...,
        description="UTC timestamp when this report was generated.",
    )
    generated_by: str = Field(
        ...,
        max_length=256,
        description=(
            "Identity of the system or user that produced the report "
            "(e.g. 'sentinel-core/attack-path-analyser v0.1.0')."
        ),
        examples=["sentinel-core/attack-path-analyser v0.1.0"],
    )
    source_scan_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of ``NetworkScan.scan_id`` values whose data was used as input "
            "to the attack-path analysis. Must reference at least one scan."
        ),
        examples=[["e3b0c442-98fc-4c14-9afb-1234567890ab"]],
    )
    scope_description: str | None = Field(
        default=None,
        description=(
            "Narrative description of the environment and business context "
            "covered by this assessment (e.g. 'All internet-facing assets in the "
            "production VPC and their lateral movement paths to the database tier')."
        ),
    )
    attack_paths: list[AttackPath] = Field(
        default_factory=list,
        description=(
            "All attack paths discovered during the analysis, ordered by "
            "``overall_likelihood`` descending (highest risk first). "
            "An empty list indicates no exploitable paths were found."
        ),
    )
    total_paths_found: int = Field(
        default=0,
        ge=0,
        description="Total number of distinct attack paths identified across all targets.",
    )
    critical_paths: int = Field(
        default=0,
        ge=0,
        description=(
            "Count of paths whose ``overall_severity`` is ``critical``. "
            "A non-zero value should trigger immediate remediation workflows."
        ),
    )
    assets_at_risk: list[NetworkAddress] = Field(
        default_factory=list,
        description=(
            "Deduplicated list of high-value assets that appear as the ``target_asset`` "
            "in at least one discovered attack path."
        ),
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description=(
            "Prioritised, actionable remediation recommendations derived from "
            "the identified paths. Each entry should reference the relevant path "
            "or technique (e.g. 'Patch CVE-2021-44228 on PROD-WEB-01 to break "
            "2 critical paths targeting PROD-DB-01')."
        ),
        examples=[[
            "Patch CVE-2021-44228 on PROD-WEB-01 to break 2 critical paths.",
            "Restrict SMB (port 445) between DMZ and database VLAN.",
        ]],
    )
    executive_summary: str | None = Field(
        default=None,
        description=(
            "Board-level narrative summarising the overall risk posture, "
            "key findings, and top-line remediation priorities. "
            "Suitable for inclusion in a security briefing without technical detail."
        ),
    )
    analyst_notes: str | None = Field(
        default=None,
        description=(
            "Free-form analyst commentary — methodology caveats, "
            "out-of-scope items, and follow-up investigation threads."
        ),
    )
