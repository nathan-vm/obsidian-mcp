import os
import re
import shutil
from pathlib import Path

import frontmatter
import httpx
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/vault"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://host.docker.internal:1234/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "obsidian_vault")
PORT = int(os.environ.get("MCP_PORT", 55000))

mcp = FastMCP("obsidian")
qdrant = QdrantClient(url=QDRANT_URL)


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_path(relative: str) -> Path:
    resolved = (VAULT_PATH / relative).resolve()
    if not str(resolved).startswith(str(VAULT_PATH.resolve())):
        raise ValueError(f"Path escapes vault: {relative}")
    return resolved


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


# ── vault tools ───────────────────────────────────────────────────────────────

@mcp.tool()
def list_notes(directory: str = "") -> list[dict]:
    """List all markdown notes in the vault (or a subdirectory).

    Returns relative paths, note names and last-modified timestamps.
    """
    base = safe_path(directory) if directory else VAULT_PATH
    if not base.exists():
        return []
    return [
        {
            "path": str(p.relative_to(VAULT_PATH)),
            "name": p.stem,
            "modified": p.stat().st_mtime,
        }
        for p in sorted(base.rglob("*.md"))
    ]


@mcp.tool()
def read_note(path: str) -> str:
    """Read the full markdown content of a note.

    Args:
        path: Path relative to the vault root (e.g. "Projects/myproject.md")
    """
    return safe_path(path).read_text(encoding="utf-8")


@mcp.tool()
def write_note(path: str, content: str) -> str:
    """Write (or overwrite) a note. Creates parent directories if needed.

    Args:
        path: Path relative to the vault root
        content: Full markdown content to write
    """
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written: {path}"


@mcp.tool()
def create_note(path: str, content: str = "", overwrite: bool = False) -> str:
    """Create a new note. Raises an error if it already exists (unless overwrite=True).

    Args:
        path: Path relative to the vault root
        content: Initial markdown content
        overwrite: Set True to replace an existing note
    """
    p = safe_path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists — use overwrite=True to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Created: {path}"


@mcp.tool()
def delete_note(path: str) -> str:
    """Permanently delete a note.

    Args:
        path: Path relative to the vault root
    """
    p = safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Note not found: {path}")
    p.unlink()
    return f"Deleted: {path}"


@mcp.tool()
def move_note(from_path: str, to_path: str) -> str:
    """Move or rename a note.

    Args:
        from_path: Current path relative to the vault root
        to_path: New path relative to the vault root
    """
    src = safe_path(from_path)
    dst = safe_path(to_path)
    if not src.exists():
        raise FileNotFoundError(f"Note not found: {from_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved: {from_path} → {to_path}"


@mcp.tool()
def get_note_metadata(path: str) -> dict:
    """Return frontmatter, tags (inline + YAML), and wikilinks for a note.

    Args:
        path: Path relative to the vault root
    """
    text = safe_path(path).read_text(encoding="utf-8")
    post = frontmatter.loads(text)
    yaml_tags = post.metadata.get("tags", [])
    if isinstance(yaml_tags, str):
        yaml_tags = [yaml_tags]
    inline_tags = re.findall(r"(?<!\S)#([\w/]+)", post.content)
    wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", post.content)
    return {
        "path": path,
        "frontmatter": dict(post.metadata),
        "tags": list(set(yaml_tags + inline_tags)),
        "wikilinks": list(set(wikilinks)),
        "word_count": len(post.content.split()),
    }


# ── search tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def fulltext_search(query: str, case_sensitive: bool = False) -> list[dict]:
    """Fast full-text (grep) search across all notes.

    Use this as a fallback when semantic search returns no useful results,
    or when searching for an exact term, title, or tag.

    Args:
        query: Text to search for
        case_sensitive: Default False
    """
    flags = 0 if not case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags if not case_sensitive else 0)
    results = []
    for p in VAULT_PATH.rglob("*.md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        matches = [m.start() for m in pattern.finditer(text)]
        if matches:
            snippets = []
            for pos in matches[:3]:
                start = max(0, pos - 100)
                end = min(len(text), pos + 100)
                snippets.append(text[start:end].replace("\n", " ").strip())
            results.append({
                "path": str(p.relative_to(VAULT_PATH)),
                "match_count": len(matches),
                "snippets": snippets,
            })
    return sorted(results, key=lambda r: r["match_count"], reverse=True)


@mcp.tool()
async def search_similar(query: str, n_results: int = 5, tag: str = "") -> list[dict]:
    """Semantic similarity search using local embeddings (LM Studio + Qdrant).

    Falls back to fulltext_search automatically if the collection is empty
    or LM Studio is unreachable.

    Args:
        query: Natural language query describing what you're looking for
        n_results: Number of results (default 5)
        tag: Optional Obsidian tag to filter (e.g. "project/work")
    """
    # Check if collection has vectors
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        if (info.vectors_count or 0) == 0:
            return _fallback_search(query, n_results)
    except Exception:
        return _fallback_search(query, n_results)

    try:
        vector = await embed(query)
    except Exception:
        return _fallback_search(query, n_results)

    filter_condition = None
    if tag:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        filter_condition = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value=tag))]
        )

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=n_results,
        query_filter=filter_condition,
        with_payload=True,
    )
    return [
        {
            "score": round(hit.score, 4),
            "path": hit.payload.get("path"),
            "heading": hit.payload.get("heading", ""),
            "chunk": hit.payload.get("chunk_text", ""),
            "tags": hit.payload.get("tags", []),
        }
        for hit in hits
    ]


def _fallback_search(query: str, n: int) -> list[dict]:
    """Internal: text search used when vector index is not ready."""
    raw = fulltext_search(query)[:n]
    return [
        {
            "score": None,
            "path": r["path"],
            "heading": "",
            "chunk": r["snippets"][0] if r["snippets"] else "",
            "tags": [],
            "note": "fallback — vector index not ready",
        }
        for r in raw
    ]


@mcp.tool()
def indexing_status() -> dict:
    """Return how many note chunks are currently indexed in the vector database."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "vectors_indexed": info.vectors_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"collection": COLLECTION_NAME, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=PORT)
