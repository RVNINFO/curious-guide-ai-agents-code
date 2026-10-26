"""Listing 3.12.3 - A minimal A2A client over raw JSON-RPC (httpx).

The book's snippet uses module-level httpx calls against supplier.example.com. Here the logic is
wrapped in functions that take an httpx.Client, so tests can inject a mock transport.
Run against a real A2A agent:  python -m ch03.l3_12_3_a2a_client https://host token
"""
import sys
import time

import httpx


def get_task(client: httpx.Client, url: str, task_id: str, token: str) -> dict:
    res = client.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "jsonrpc": "2.0",
            "id": "req-2",
            "method": "GetTask",
            "params": {"id": task_id},
        },
    ).json()
    return res["result"]


def ask_agent(client: httpx.Client, base: str, token: str, text: str, poll_seconds: float = 2):
    # 1. Discover what the agent can do
    card = client.get(f"{base}/.well-known/agent-card.json").json()
    skills = [s["id"] for s in card["skills"]]
    print(skills)  # ['check_stock', 'reserve_stock']

    # 2. Send it a task
    response = client.post(
        card["url"],
        headers={"Authorization": f"Bearer {token}"},
        json={
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                    "messageId": "msg-1",
                }
            },
        },
    ).json()

    result = response.get("result", {})
    if "id" in result:
        # 3. Handle asynchronous Task result
        task_id = result["id"]
        state = result["status"]["state"]
        task = result
        while state in ("submitted", "working"):
            time.sleep(poll_seconds)
            task = get_task(client, card["url"], task_id, token)
            state = task["status"]["state"]
        return task.get("artifacts")
    # Direct Message result
    return result.get("parts")


if __name__ == "__main__":
    base, token = sys.argv[1], sys.argv[2]
    with httpx.Client() as c:
        print(ask_agent(c, base, token, "Do you have 500 units of SKU-8842?"))
