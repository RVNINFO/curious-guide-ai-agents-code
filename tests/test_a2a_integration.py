"""Integration tests against a live A2A v1.0 server (e.g., helloworld from a2a-samples).

Run live tests:
    A2A_SERVER_URL=http://127.0.0.1:9999 pytest tests/test_a2a_integration.py
"""
import os
import httpx
import pytest

from ch03.l3_12_3_a2a_client import ask_agent, endpoint_from_card, get_task

SERVER_URL = os.getenv("A2A_SERVER_URL")

pytestmark = pytest.mark.skipif(
    not SERVER_URL,
    reason="Live A2A server not running (set A2A_SERVER_URL to run)"
)


def test_a2a_server_fetch_agent_card():
    with httpx.Client() as client:
        res = client.get(f"{SERVER_URL}/.well-known/agent-card.json")
        assert res.status_code == 200
        card = res.json()
        assert "supportedInterfaces" in card
        assert "skills" in card


def test_a2a_server_endpoint_from_card():
    with httpx.Client() as client:
        card = client.get(f"{SERVER_URL}/.well-known/agent-card.json").json()
        endpoint = endpoint_from_card(card)
        assert endpoint.startswith("http")


def test_a2a_server_direct_message():
    with httpx.Client() as client:
        res = ask_agent(client, SERVER_URL, None, "Hello", poll_seconds=0.1)
        assert res is not None


def test_a2a_server_async_task():
    with httpx.Client() as client:
        res = ask_agent(client, SERVER_URL, None, "Do you have 500 units of SKU-8842?", poll_seconds=0.1)
        assert res is not None


def test_a2a_server_return_immediately():
    with httpx.Client() as client:
        res = ask_agent(client, SERVER_URL, None, "Do you have 500 units of SKU-8842?",
                        poll_seconds=0.1, return_immediately=True)
        assert res is not None


def test_a2a_server_get_task_by_id():
    with httpx.Client() as client:
        card = client.get(f"{SERVER_URL}/.well-known/agent-card.json").json()
        url = endpoint_from_card(card)
        params = {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": "check"}],
                "messageId": "msg-test",
            },
            "configuration": {"returnImmediately": True}
        }
        res = client.post(url, headers={"A2A-Version": "1.0"},
                          json={"jsonrpc": "2.0", "id": "req-1", "method": "SendMessage", "params": params}).json()
        if "task" in res.get("result", {}):
            task_id = res["result"]["task"]["id"]
            task = get_task(client, url, None, task_id)
            assert task["id"] == task_id
