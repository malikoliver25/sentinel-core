"""
app.tools
~~~~~~~~~
LangChain tool definitions wired into the sentinel-core agent.

Each tool is decorated with ``@tool`` so LangGraph can bind them directly
to the agent's ReAct loop.  Implementations are air-gapped stubs; replace
the function bodies with real integrations (SIEM, EDR, threat-intel APIs)
without changing the tool signatures or schemas.

Adding a new tool:
    1. Define a function and decorate it with ``@tool``.
    2. Write a precise docstring — the LLM uses it to decide when to call it.
    3. Append the function to ``SENTINEL_TOOLS`` at the bottom of this file.
"""

from __future__ import annotations

import datetime
import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Host / endpoint tools
# ---------------------------------------------------------------------------


@tool
def query_host_metadata(hostname: str) -> str:
    """
    Retrieve asset metadata for a given hostname from the CMDB.

    Returns operating system, owner, criticality rating, network zone,
    and last-seen timestamp.  Use this before any other host-level action
    to establish baseline context.

    Args:
        hostname: The fully-qualified or short hostname to look up.

    Returns:
        JSON string containing asset metadata fields.
    """
    logger.info("query_host_metadata called for host=%s", hostname)

    # --- stub implementation ---
    result = {
        "hostname": hostname,
        "os": "RHEL 9.2",
        "owner": "platform-team",
        "criticality": "high",
        "network_zone": "dmz",
        "last_seen": datetime.datetime.utcnow().isoformat() + "Z",
        "note": "stub — replace with CMDB API call",
    }
    return json.dumps(result)


@tool
def isolate_host(hostname: str, reason: str) -> str:
    """
    Quarantine a host by removing it from all network segments.

    This is a **destructive, high-impact action**.  Only call after
    confirming malicious activity through at least one corroborating
    evidence source.  The operation is logged and triggers a SOC alert.

    Args:
        hostname: The target host to isolate.
        reason:   A concise, human-readable justification for isolation.

    Returns:
        JSON string with isolation status and ticket reference.
    """
    logger.warning("isolate_host called for host=%s reason=%s", hostname, reason)

    result = {
        "hostname": hostname,
        "action": "isolate",
        "status": "simulated_success",
        "ticket": "INC-00000",
        "reason": reason,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "note": "stub — replace with NAC / EDR API call",
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Threat intelligence tools
# ---------------------------------------------------------------------------


@tool
def lookup_ioc(indicator: str, indicator_type: str = "auto") -> str:
    """
    Query the threat-intelligence platform for a known indicator of compromise.

    Supports IP addresses, domain names, file hashes (MD5/SHA-1/SHA-256),
    and URLs.  Set ``indicator_type`` to one of ``ip``, ``domain``,
    ``hash``, ``url``, or leave as ``auto`` for automatic detection.

    Args:
        indicator:      The raw IOC value to look up.
        indicator_type: Type hint to speed up classification.

    Returns:
        JSON string with reputation score, tags, first/last seen, and sources.
    """
    logger.info("lookup_ioc called indicator=%s type=%s", indicator, indicator_type)

    result = {
        "indicator": indicator,
        "type": indicator_type,
        "reputation": "malicious",
        "confidence": 0.91,
        "tags": ["apt29", "lateral-movement"],
        "first_seen": "2025-11-03T00:00:00Z",
        "last_seen": datetime.datetime.utcnow().isoformat() + "Z",
        "sources": ["VirusTotal", "MISP"],
        "note": "stub — replace with TI platform API call",
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Log / SIEM tools
# ---------------------------------------------------------------------------


@tool
def search_siem_logs(query: str, lookback_minutes: int = 60) -> str:
    """
    Execute a structured search against the SIEM for recent log events.

    Write the query in the platform's native syntax (e.g. Sigma, KQL, SPL).
    Results are capped at 100 events to prevent context overflow; refine
    the query if more precision is needed.

    Args:
        query:            Search expression in the SIEM's query language.
        lookback_minutes: How far back to search, in minutes (default 60).

    Returns:
        JSON string containing a list of matching log events and hit count.
    """
    logger.info("search_siem_logs query=%r lookback=%d min", query, lookback_minutes)

    result = {
        "query": query,
        "lookback_minutes": lookback_minutes,
        "total_hits": 3,
        "events": [
            {
                "timestamp": "2026-07-14T04:00:01Z",
                "host": "PROD-WEB-01",
                "event_id": 4688,
                "process": "cmd.exe",
                "cmdline": "whoami /all",
            },
            {
                "timestamp": "2026-07-14T04:00:12Z",
                "host": "PROD-WEB-01",
                "event_id": 4624,
                "logon_type": 3,
                "src_ip": "10.0.4.55",
            },
            {
                "timestamp": "2026-07-14T04:01:00Z",
                "host": "PROD-WEB-01",
                "event_id": 4648,
                "target": "DC-01$",
            },
        ],
        "note": "stub — replace with SIEM SDK / Elasticsearch call",
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Reporting tools
# ---------------------------------------------------------------------------


@tool
def create_incident_report(
    title: str,
    severity: str,
    summary: str,
    affected_assets: list[str],
) -> str:
    """
    Create a formal incident report and open a tracking ticket in the ITSM.

    Call this as the **final action** once the investigation is complete and
    findings are consolidated.  Do not call during active triage.

    Args:
        title:            Short, descriptive title for the incident.
        severity:         One of ``low``, ``medium``, ``high``, ``critical``.
        summary:          Multi-sentence narrative of findings and actions taken.
        affected_assets:  List of hostnames, IPs, or resource IDs involved.

    Returns:
        JSON string with the created ticket ID and portal URL.
    """
    logger.info("create_incident_report title=%r severity=%s", title, severity)

    result = {
        "ticket_id": "INC-99999",
        "title": title,
        "severity": severity,
        "status": "open",
        "affected_assets": affected_assets,
        "summary": summary,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "portal_url": "https://itsm.internal/incidents/INC-99999",
        "note": "stub — replace with ITSM API call",
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool registry — imported by agent.py
# ---------------------------------------------------------------------------

SENTINEL_TOOLS: list = [
    query_host_metadata,
    isolate_host,
    lookup_ioc,
    search_siem_logs,
    create_incident_report,
]
