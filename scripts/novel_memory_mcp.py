"""Read-only MCP server exposing the project's novel memory.

The server is intentionally limited to the configured novel-memory directory.
It is started by ``configs/mcp_servers.json`` over stdio and can be used by
the Codex-style UI or any MCP-compatible client.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Allow launching this file directly (``python scripts/novel_memory_mcp.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from novel_memory_store import NovelMemoryStore  # noqa: E402


def _memory_root() -> Path:
    configured = os.environ.get("NOVEL_MEMORY_PATH", "novel_memory/my_novel")
    root = Path(configured)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _clip(value: Any, limit: int = 12000) -> Any:
    """Bound tool output so an external server cannot fill the prompt."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "\n[结果已截断]"
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + "\n[结果已截断]"


mcp = FastMCP("Novel Memory")


@mcp.tool()
def get_novel_context(query: str = "") -> str:
    """Return bounded world, outline, character and recent-chapter context."""
    store = NovelMemoryStore(_memory_root())
    return _clip(store.build_context(query=query, max_chars=7000))


@mcp.tool()
def list_chapters() -> list[dict[str, Any]]:
    """List saved chapter files and their sizes."""
    root = _memory_root() / "chapters"
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("chapter_*.md")):
        rows.append({"name": path.name, "bytes": path.stat().st_size})
    return rows[-100:]


@mcp.tool()
def search_chapters(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search chapter text for a phrase and return small matching excerpts."""
    query = str(query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 20))
    rows: list[dict[str, str]] = []
    for path in sorted((_memory_root() / "chapters").glob("chapter_*.md"), reverse=True):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        pos = text.lower().find(query.lower())
        if pos < 0:
            continue
        start = max(0, pos - 180)
        excerpt = text[start : pos + len(query) + 320].strip()
        rows.append({"chapter": path.name, "excerpt": excerpt})
        if len(rows) >= limit:
            break
    return rows


@mcp.tool()
def get_character(name: str) -> dict[str, Any] | None:
    """Find one character by name in characters.json."""
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    target = _memory_root() / "characters.json"
    try:
        items = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and str(item.get("name", item.get("姓名", ""))).strip().lower() == wanted:
            return item
    return None


@mcp.tool()
def get_unresolved_foreshadowing() -> list[dict[str, Any]]:
    """Return unresolved foreshadowing entries from the novel memory."""
    target = _memory_root() / "foreshadowing.json"
    try:
        items = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(items, list):
        return []
    closed = {"resolved", "closed", "已回收", "完成"}
    return [item for item in items if isinstance(item, dict) and str(item.get("status", item.get("状态", "未回收"))).lower() not in closed]


if __name__ == "__main__":
    mcp.run(transport="stdio")
