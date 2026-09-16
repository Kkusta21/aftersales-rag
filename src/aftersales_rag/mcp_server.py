"""MCP server exposing the assistant's tools, so any MCP client (Claude Desktop, agents) can use them."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer  # mcp>=2 (FastMCP was renamed)

from .agent import get_assistant

mcp = MCPServer("norvik-aftersales")


@mcp.tool()
def search_service_docs(query: str, model: str | None = None, k: int = 4) -> list[dict]:
    """Search Norvik service manuals, warranty policy and maintenance schedule.

    Args:
        query: The question or keywords.
        model: Optional vehicle model, "Aster Hybrid" or "Tern EV".
        k: Number of passages to return.
    """
    hits = get_assistant().retriever.search(query, k=k, model=model)
    return [
        {"source": h.chunk.citation(), "text": h.chunk.text, "relevance": round(h.relevance, 3)} for h in hits
    ]


@mcp.tool()
def check_recalls(model: str, year: int | None = None, include_closed: bool = False) -> list[dict]:
    """List recall campaigns for a Norvik model, optionally filtered by model year."""
    return get_assistant().recall_db.lookup(model, year, include_closed)


@mcp.tool()
def ask_assistant(question: str) -> dict:
    """Ask the full after-sales assistant. Returns an answer with citations."""
    r = get_assistant().ask(question)
    return {"answer": r["answer"], "route": r["route"], "citations": r.get("citations", [])}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
