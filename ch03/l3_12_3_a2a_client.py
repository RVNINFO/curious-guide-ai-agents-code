"""Listing 3.12.3 - A minimal A2A client over raw JSON-RPC (httpx), for A2A protocol v1.0.

Verified against the official a2a-samples "helloworld" server (a2a-sdk 1.1.0, protocol 1.0);
see TESTED_VERSIONS.md and tests/test_a2a_integration.py.

What v1.0 requires on the wire (each of these broke the first draft of this listing):
  * the Agent Card gives its endpoint in `supportedInterfaces[].url`, not a top-level `url`;
  * every request carries an `A2A-Version: 1.0` header;
  * roles are "ROLE_USER" / "ROLE_AGENT", and a part is `{"text": ...}` with no "kind" field;
  * `SendMessage` returns `result.task` OR `result.message` (never a bare task);
  * task states are "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", ... "TASK_STATE_COMPLETED".

The logic is wrapped in functions taking an httpx.Client so tests can inject a mock transport.
Run against a server:  python -m ch03.l3_12_3_a2a_client http://127.0.0.1:9999 [token]
"""
import sys
import time

import httpx

A2A_VERSION = "1.0"
ACTIVE_STATES = {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}
PAUSED_STATES = {"TASK_STATE_INPUT_REQUIRED",
                 "TASK_STATE_AUTH_REQUIRED"}


class TaskPaused(RuntimeError):
    """The agent stopped and is waiting for the caller
    (more input, or credentials)."""

    def __init__(self, state: str, task: dict):
        self.state = state
        self.task = task
        parts = task["status"].get("message", {}).get("parts", [])
        self.question = " ".join(
            p.get("text", "") for p in parts).strip()
        super().__init__(
            f"Task {task['id']} is waiting: {state}."
            f"{self.question}")


def _headers(token: str | None) -> dict:
    h = {"A2A-Version": A2A_VERSION}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _rpc(
    client: httpx.Client,
    url: str,
    token: str | None,
    req_id: str,
    method: str,
    params: dict,
) -> dict:
    res = client.post(
        url,
        headers=_headers(token),
        json={"jsonrpc": "2.0", "id": req_id,
              "method": method, "params": params},
    ).json()
    if "error" in res:
        raise RuntimeError(
            f"A2A error from {method}: {res['error']}")
    return res["result"]


def get_task(client: httpx.Client, url: str,
             token: str | None, task_id: str) -> dict:
    return _rpc(client, url, token, "req-2", "GetTask",
                {"id": task_id})


def endpoint_from_card(card: dict) -> str:
    """Pick the JSON-RPC endpoint the card advertises."""
    for iface in card["supportedInterfaces"]:
        if iface["protocolBinding"] == "JSONRPC":
            return iface["url"]
    raise ValueError("Agent does not offer a JSONRPC interface")


def ask_agent(
    client: httpx.Client,
    base: str,
    token: str | None,
    text: str,
    poll_seconds: float = 2,
    timeout_seconds: float = 60,
    return_immediately: bool = False,
):
    # 1. Discover what the agent can do
    card = client.get(f"{base}/.well-known/agent-card.json").json()
    # e.g. ['check_stock', 'reserve_stock']
    print([s["id"] for s in card["skills"]])
    url = endpoint_from_card(card)

    # 2. Send it a task
    params = {
        "message": {
            "role": "ROLE_USER",
            "parts": [{"text": text}],
            "messageId": "msg-1",
        }
    }
    # don't wait for completion: forces the polling path below
    if return_immediately:
        params["configuration"] = {"returnImmediately": True}
    result = _rpc(client, url, token, "req-1",
                  "SendMessage", params)

    # Direct Message result: no task was created
    if "message" in result:
        return result["message"]["parts"]

    # 3. Poll the task until it finishes, pauses, or timeout
    task = result["task"]
    deadline = time.monotonic() + timeout_seconds

    while task["status"]["state"] in ACTIVE_STATES:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Task {task['id']} still running after "
                f"{timeout_seconds}s"
            )
        time.sleep(poll_seconds)
        task = get_task(client, url, token, task["id"])

    state = task["status"]["state"]
    if state in PAUSED_STATES:
        # caller decides: ask the user, or authenticate
        raise TaskPaused(state, task)

    if state != "TASK_STATE_COMPLETED":
        raise RuntimeError(f"Task ended in state {state}")

    return task.get("artifacts")


if __name__ == "__main__":
    base = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else None

    with httpx.Client() as c:
        print(ask_agent(
            c, base, token,
            "Do you have 500 units of SKU-8842?"))
        print(
            ask_agent(
                c, base, token,
                "Do you have 500 units of SKU-8842?",
                poll_seconds=0.2,
                return_immediately=True,
            )
        )
