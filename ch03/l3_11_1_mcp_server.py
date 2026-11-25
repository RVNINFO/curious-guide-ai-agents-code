"""Listings 3.11.1 - 3.11.3 - A minimal MCP server (FastMCP) with one tool, one resource and one prompt.
Run:  python -m ch03.l3_11_1_mcp_server
(stdio transport by default; point an MCP host/client at this command).
Wire examples for each capability are in data/mcp_*.jsonrpc.json.
"""
# Framework: FastMCP 2.x or later (tested on 4.1)
from fastmcp import FastMCP
mcp = FastMCP("translation-service")

# stand-in for a real service
PHRASEBOOK = {("hello", "Japanese"): "こんにちは"}


# 3.11.1 Tools
@mcp.tool()
def translate_text(text: str, target_language: str) -> str:
    """Translate text into the target language,
    for example 'Japanese'."""
    return PHRASEBOOK.get(
        (text.lower(), target_language),
        f"[no translation for '{text}']")


# 3.11.2 Resources
@mcp.resource(
    "translation://languages",
    description="Languages the service can translate into.")
def supported_languages() -> str:
    return "Japanese, French, Spanish"


# 3.11.3 Prompts
@mcp.prompt()
def translate_formally(text: str, target_language: str) -> str:
    """Ask for a formal translation."""
    return (f"Translate this into {target_language}, "
            f"formally:\n\n{text}")


if __name__ == "__main__":
    mcp.run()   # stdio by default
