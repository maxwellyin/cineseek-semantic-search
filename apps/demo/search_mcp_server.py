from __future__ import annotations

import json
import os

from fastmcp import FastMCP

from apps.demo import network


DEFAULT_CANDIDATE_K = int(os.environ.get("FLCR_AGENT_CANDIDATE_K", "30"))

mcp = FastMCP("CineSeek Search Tool")
mcp_app = mcp.http_app(path="/mcp")


@mcp.tool
def search_movies(query: str, k: int = DEFAULT_CANDIDATE_K) -> str:
    """Search the CineSeek semantic index and return compressed movie candidates."""
    result = network.direct_recommend(query, k=k)
    payload = {
        "query_used": result["query_used"],
        "recommendations": [],
    }
    for idx, item in enumerate(result["recommendations"], start=1):
        structured = item.get("structured") or {}
        payload["recommendations"].append(
            {
                "rank": idx,
                "title": item["title"],
                "score": round(float(item["score"]), 4),
                "year": structured.get("release_year", ""),
                "genres": structured.get("genres", [])[:4],
                "overview": (structured.get("overview", "") or "")[:220],
                "tags": structured.get("tags", [])[:4],
            }
        )
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="http")
