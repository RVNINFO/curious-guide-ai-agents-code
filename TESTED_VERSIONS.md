# Tested Versions & Environment Matrix

This repository verifies all code listings against the pins published in *A Curious Guide to AI Agents* (First Edition).

## Pinned Core Dependencies
* Python: 3.11, 3.12, 3.13
* `langchain`: 1.4.7
* `langchain-core`: 1.7.0
* `langchain-classic`: 1.0.8
* `langgraph`: 1.2.18
* `fastmcp`: 4.1.4
* `fastapi`: 0.162.0
* `uvicorn`: 0.54.0
* `httpx`: 0.28.1
* `pydantic`: 2.14.1
* `pytest`: 9.2.0
* `sqlglot`: >=25.0.0 (used in Listing 5.9.4)
* `numpy`: 2.4.6 (< 3.12) / 2.5.4 (>= 3.12)

## Tested Protocols & Specifications
* **A2A (Agent2Agent Protocol)**: v1.0
  * Tested against `a2a-samples` "helloworld" server (`a2a-sdk` 1.1.0, protocol 1.0).
  * Wire format uses `A2A-Version: 1.0` header, JSON-RPC 2.0 bindings, `supportedInterfaces`, and standard task states (`TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_INPUT_REQUIRED`, `TASK_STATE_AUTH_REQUIRED`, `TASK_STATE_COMPLETED`).
* **MCP (Model Context Protocol)**:
  * Tested with FastMCP 4.1.4 (Listing 3.11.1).

## Live Model Strings
* `anthropic:claude-sonnet-5-5`
* `anthropic:claude-haiku-5-5`
