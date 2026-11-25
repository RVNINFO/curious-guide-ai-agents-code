"""Listing 6.2 - Durable execution: a PostgresSaver checkpointer and a correlation ID.

Needs `pip install langgraph-checkpoint-postgres "psycopg[binary]"` (see requirements-live.txt) and a
Postgres database. `handle` is database-free, so the tests exercise it with the in-memory checkpointer;
`main` needs CHECKPOINT_DB_URI and your own `serve` (an API handler or queue consumer).
"""
# Production persistence for the agent in Section 5.11.1.
# pip install langgraph-checkpoint-postgres "psycopg[binary]"
import os
import uuid

from ch05.l5_11_1_capstone import AppContext, build_agent


def handle(agent, message, user_id, correlation_id=None):
    # Reuse the same ID when the approval for this task arrives.
    correlation_id = correlation_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": correlation_id},
        "metadata": {"correlation_id": correlation_id},
    }
    result = agent.invoke({"messages": [("user", message)]},
                          config=config,
                          context=AppContext(user_id=user_id))
    return result, correlation_id


def serve(agent):
    """Stand-in: your API handler or queue consumer calls handle(agent, ...)."""
    raise NotImplementedError


def main():
    from langgraph.checkpoint.postgres import PostgresSaver

    # Read from the secret store at deploy time, never from code.
    DB_URI = os.environ["CHECKPOINT_DB_URI"]
    # Open the checkpointer once, when the worker starts.
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # creates the tables on first use
        agent = build_agent(checkpointer=checkpointer)
        serve(agent)  # your API handler or queue consumer


if __name__ == "__main__":
    main()
