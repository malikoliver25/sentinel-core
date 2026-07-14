"""
app.main
~~~~~~~~
FastAPI application entry point for the sentinel-core orchestration engine.

Start the server
----------------
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
    GET  /health       — Liveness probe for container orchestrators.
    POST /run          — Submit a task to the security orchestration agent.
    POST /analyze      — Submit a NetworkScan and receive an AttackPathReport.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from langchain_core.messages import HumanMessage

from app.agent import build_agent, build_analysis_graph
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    AttackPathReport,
    NetworkScan,
    ScanStatus,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — holds singleton resources initialised at startup
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """
    Compile both LangGraph graphs once at startup:
    - ``agent``          — ReAct LLM agent for POST /run
    - ``analysis_graph`` — Deterministic pipeline for POST /analyze
    """
    logger.info("Sentinel-core starting — compiling graphs …")
    _state["agent"] = build_agent()
    _state["analysis_graph"] = build_analysis_graph()
    logger.info("All graphs ready.")
    yield
    logger.info("Sentinel-core shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OpenAPI tag definitions — controls grouping order in Swagger UI
# ---------------------------------------------------------------------------

_TAGS_METADATA = [
    {
        "name": "ops",
        "description": "Operational probes for container orchestrators and load balancers.",
    },
    {
        "name": "orchestration",
        "description": "LangGraph agent endpoints — submit natural-language security tasks.",
    },
    {
        "name": "analysis",
        "description": (
            "Structured security analysis endpoints. Submit scan data and receive "
            "machine-readable threat intelligence reports."
        ),
    },
]

app = FastAPI(
    title="Sentinel-Core",
    summary="Enterprise-grade, air-gapped security orchestration engine.",
    description=(
        "## Overview\n"
        "Sentinel-Core is an air-gapped security orchestration platform that combines "
        "a **LangGraph ReAct agent** with structured network analysis pipelines.\n\n"
        "## Key capabilities\n"
        "- **`POST /run`** — Natural-language security task submission; the agent reasons, "
        "invokes tools (SIEM, EDR, threat-intel), and returns a structured incident report.\n"
        "- **`POST /analyze`** — Submit a `NetworkScan` payload; receive an `AttackPathReport` "
        "mapping discovered vulnerabilities to MITRE ATT&CK-aligned attack paths.\n"
        "- **`GET /health`** — Liveness probe for Kubernetes / ECS health checks.\n\n"
        "## Authentication\n"
        "All endpoints require a valid `Authorization: Bearer <token>` header in production. "
        "Authentication middleware is not applied in this development build."
    ),
    version="0.1.0",
    contact={
        "name": "Sentinel-Core Engineering",
        "email": "security-platform@sentinel.internal",
    },
    license_info={
        "name": "Proprietary — All rights reserved",
    },
    openapi_tags=_TAGS_METADATA,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["ops"],
    summary="Liveness probe",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> dict:
    """Return 200 OK when the service is alive and the agent is compiled."""
    return {
        "status": "ok",
        "agent_ready": "agent" in _state,
    }


@app.post(
    "/run",
    response_model=AgentResponse,
    tags=["orchestration"],
    summary="Submit a task to the security orchestration agent",
    status_code=status.HTTP_200_OK,
)
async def run_agent(request: AgentRequest) -> AgentResponse:
    """
    Invoke the LangGraph security agent with the supplied task.

    The agent will reason over the task, invoke tools as needed (SIEM queries,
    host isolation, IOC lookups, etc.), and return a structured report.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not initialised — server may still be starting.",
        )

    run_id = str(uuid.uuid4())
    start_ts = time.monotonic()
    logger.info("run_id=%s task=%r severity=%s", run_id, request.task[:80], request.severity)

    # Build the initial system + user message pair
    system_prompt = (
        "You are Sentinel, an enterprise security orchestration AI.\n"
        "Your mission: investigate the supplied security task methodically, "
        "use the available tools to gather evidence and take containment actions "
        "where warranted, and produce a concise incident summary.\n"
        f"Severity: {request.severity.value}. "
        f"Max reasoning iterations: {request.max_iterations}.\n"
        "Context: " + str(request.context)
    )

    messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=request.task),
    ]

    try:
        result = await agent.ainvoke(
            {"messages": messages},
            config={"recursion_limit": request.max_iterations * 2},
        )
    except Exception as exc:
        logger.exception("Agent run %s failed: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {exc}",
        ) from exc

    elapsed = time.monotonic() - start_ts

    # Extract tool calls from the message history
    tool_results: list[ToolResult] = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                tool_results.append(
                    ToolResult(
                        tool_name=tc.get("name", "unknown"),
                        success=True,
                        data=tc.get("args"),
                    )
                )

    # Final assistant message is the summary
    final_messages = result.get("messages", [])
    summary = final_messages[-1].content if final_messages else "No summary produced."

    response = AgentResponse(
        run_id=run_id,
        status=AgentStatus.COMPLETED,
        summary=summary,
        tool_calls=tool_results,
        iterations_used=len([m for m in final_messages if hasattr(m, "tool_calls")]),
        metadata={
            "model": "gpt-4o",
            "elapsed_seconds": round(elapsed, 3),
            "severity": request.severity.value,
        },
    )

    logger.info("run_id=%s completed in %.2fs", run_id, elapsed)
    return response


# ---------------------------------------------------------------------------
# /analyze — NetworkScan → AttackPathReport
# ---------------------------------------------------------------------------


@app.post(
    "/analyze",
    response_model=AttackPathReport,
    tags=["analysis"],
    summary="Analyse a network scan and generate an attack path report",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Request body failed Pydantic validation.",
        },
        status.HTTP_501_NOT_IMPLEMENTED: {
            "description": "Analysis engine not yet implemented.",
        },
    },
)
async def analyze_scan(scan: NetworkScan) -> AttackPathReport:
    """
    Submit a completed ``NetworkScan`` and receive an ``AttackPathReport``.

    ### Processing pipeline
    1. **Ingest** — validate and normalise the scan payload.
    2. **analyzer_node** — extract and de-duplicate all ``Vulnerability`` objects
       from ``scan.hosts``, sorted by CVSS severity.
    3. **tool_interface_node** — simulate adversarial attack chains; for each
       critical/high vulnerability, construct an ``AttackPath`` anchored to the
       affected host (real implementation will call SIEM, IOC, and ATT&CK tools).
    4. **summarizer_node** — assemble the ``AttackPathReport``, rank paths by
       likelihood, derive recommendations, and author the executive summary.

    ### Current status
    The graph is fully wired and stateful.  Node bodies contain documented
    placeholder logic; replace stub sections in ``agent.py`` with real
    tool calls and analysis algorithms.
    """
    graph = _state.get("analysis_graph")
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analysis graph is not initialised — server may still be starting.",
        )

    logger.info(
        "analyze scan_id=%s hosts=%d status=%s",
        scan.scan_id,
        len(scan.hosts),
        scan.status.value,
    )

    try:
        result = await graph.ainvoke(
            {
                "scan": scan,
                "identified_vulnerabilities": [],
                "analyzer_reasoning": "",
                "tool_outputs": [],
                "attack_paths": [],
                "report": None,
                "node_log": [],
            }
        )
    except Exception as exc:
        logger.exception("Analysis graph failed for scan_id=%s: %s", scan.scan_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis graph execution error: {exc}",
        ) from exc

    report: AttackPathReport | None = result.get("report")
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="summarizer_node did not produce a report — check analysis graph logs.",
        )

    logger.info(
        "analyze complete scan_id=%s report_id=%s paths=%d",
        scan.scan_id,
        report.report_id,
        report.total_paths_found,
    )
    return report
