"""Listing 3.11.1 - A minimal MCP server (FastMCP). Run:  python -m ch03.l3_11_1_mcp_server
(stdio transport by default; point an MCP host/client at this command).
"""
from fastmcp import FastMCP

mcp = FastMCP("translation-service")
PHRASEBOOK = {("hello", "Japanese"): "こんにちは"}  # stand-in for a real service


@mcp.tool()
def translate_text(text: str, target_language: str) -> str:
    """Translate text into the target language, for example 'Japanese'."""
    return PHRASEBOOK.get((text.lower(), target_language), f"[no translation for '{text}']")


if __name__ == "__main__":
    mcp.run()  # stdio by default
