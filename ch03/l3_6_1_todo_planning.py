"""Listing 3.6.1 - Planning with TodoListMiddleware.

search_docs / run_query / draft_report are stubs for your own tools.
Live:  ANTHROPIC_API_KEY=... python -m ch03.l3_6_1_todo_planning
"""
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"3 documents found for '{query}'."


@tool
def run_query(sql: str) -> str:
    """Run a read-only SQL query."""
    return "42 rows"


@tool
def draft_report(notes: str) -> str:
    """Draft a report from notes."""
    return f"REPORT: {notes}"


def build_agent(model="anthropic:claude-sonnet-5-5"):
    return create_agent(
        model=model,
        tools=[search_docs, run_query, draft_report],
        middleware=[TodoListMiddleware()],
        system_prompt=(
            "For any task with more than three steps, write a todo list "
            "first, then work through it, marking items complete as you go."
        ),
    )


if __name__ == "__main__":
    out = build_agent().invoke({"messages": [("user", "Research churn, query the data, and draft a report.")]})
    print(out["messages"][-1].content)
