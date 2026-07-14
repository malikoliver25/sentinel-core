```markdown
# Sentinel-core: Security Orchestration Engine

Sentinel-core is an automated, graph-based security orchestration engine designed to ingest network scan data, prioritize vulnerabilities based on real-world exploitability, and synthesize actionable attack path reports.

## Overview
This engine transforms raw scanner output (e.g., Nmap/OpenVAS) into high-fidelity security insights. It uses a graph-based reasoning model to link vulnerabilities across network assets, identifying paths an adversary would likely take to reach high-value targets.

## Quick Start

### Prerequisites
- [uv](https://github.com/astral-sh/uv): The project uses `uv` for dependency and environment management.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/malikoliver25/sentinel-core.git](https://github.com/malikoliver25/sentinel-core.git)
   cd sentinel-core

```

2. Sync the environment:
```bash
uv sync

```



### Running the API

Start the FastAPI server:

```bash
uv run uvicorn app.main:app --reload

```

The API will be available at `http://127.0.0.1:8000`.

## Verification (Smoke Test)

Included a smoke test that simulates a real-world scan ingestion. To verify that the ingestion and reasoning pipeline are fully functional:

1. Ensure the API is running (see above).
2. Run the test script in a separate terminal:
```bash
uv run python smoke_test.py

```


3. The script will output a `200 OK` status and the synthesized `AttackPathReport` JSON to your console.

## Architecture

* **Schemas**: Validated via Pydantic models for complex, nested scan data.
* **Agentic Engine**: Built on a graph-based reasoning model (`analyzer_node` → `tool_interface_node` → `summarizer_node`).
* **Prioritization**: Reasoning logic inherently prioritizes `CVSS > 8.0` and `exploit_available` flags, ensuring high-risk paths are surfaced first.

```

```
