import re

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    SparseVector,
)

DENSE_NAME = "text-dense"
SPARSE_NAME = "text-sparse"

_bm25: SparseTextEmbedding | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir="/app/models")
    return _bm25


def _rrf(dense_hits, sparse_hits, k: int = 60) -> list[tuple]:
    scores: dict = {}
    payloads: dict = {}
    for rank, hit in enumerate(dense_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        payloads[hit.id] = hit.payload
    for rank, hit in enumerate(sparse_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        if hit.id not in payloads:
            payloads[hit.id] = hit.payload
    return sorted(
        [(id_, scores[id_], payloads[id_]) for id_ in scores],
        key=lambda x: x[1],
        reverse=True,
    )


def _build_filter(tag: str, path: str, directory: str) -> Filter | None:
    must = []
    if tag:
        must.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
    if path:
        must.append(FieldCondition(key="path", match=MatchValue(value=path)))
    if directory:
        must.append(FieldCondition(key="folders", match=MatchValue(value=directory)))
    return Filter(must=must) if must else None


def register(mcp, config, client: QdrantClient | None, embedder):
    vault = config.vault_path
    qdrant = client

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
        for p in vault.rglob("*.md"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            matches = [m.start() for m in pattern.finditer(text)]
            if matches:
                snippets = []
                for pos in matches[:3]:
                    start = max(0, pos - 100)
                    end = min(len(text), pos + 100)
                    snippets.append(text[start:end].replace("\n", " ").strip())
                results.append(
                    {
                        "path": str(p.relative_to(vault)),
                        "match_count": len(matches),
                        "snippets": snippets,
                    }
                )
        return sorted(results, key=lambda r: r["match_count"], reverse=True)

    def _fallback_search(query: str, n: int) -> list[dict]:
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
    async def search_similar(
        query: str,
        n_results: int = 5,
        tag: str = "",
        path: str = "",
        directory: str = "",
    ) -> list[dict]:
        """Hybrid semantic + BM25 search using Qdrant.

        Combines dense vector similarity with BM25 keyword matching via RRF fusion.
        Falls back to fulltext_search if the index is empty or not yet ready.

        Args:
            query: Natural language query
            n_results: Number of results (default 5)
            tag: Filter by Obsidian tag (e.g. "project/work")
            path: Filter by exact note path (e.g. "Work/Projects/note.md")
            directory: Filter by folder or any ancestor folder (e.g. "Work/Projects")
        """
        if qdrant is None:
            return _fallback_search(query, n_results)

        collection = config.active_collection
        try:
            info = qdrant.get_collection(collection)
            count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
            if count == 0:
                return _fallback_search(query, n_results)
        except Exception:
            return _fallback_search(query, n_results)

        try:
            dense_vector = await embedder.aembed_query(query)
        except Exception:
            return _fallback_search(query, n_results)

        filter_condition = _build_filter(tag=tag, path=path, directory=directory)
        fetch = n_results * 3

        try:
            dense_hits = qdrant.search(
                collection_name=collection,
                query_vector=NamedVector(name=DENSE_NAME, vector=dense_vector),
                limit=fetch,
                query_filter=filter_condition,
                with_payload=True,
            )
        except Exception:
            return _fallback_search(query, n_results)

        try:
            bm25_emb = next(_get_bm25().query_embed(query))
            sparse_hits = qdrant.search(
                collection_name=collection,
                query_vector=NamedSparseVector(
                    name=SPARSE_NAME,
                    vector=SparseVector(
                        indices=bm25_emb.indices.tolist(),
                        values=bm25_emb.values.tolist(),
                    ),
                ),
                limit=fetch,
                query_filter=filter_condition,
                with_payload=True,
            )
        except Exception:
            sparse_hits = []

        fused = _rrf(dense_hits, sparse_hits)[:n_results]
        return [
            {
                "score": round(score, 4),
                "path": payload.get("path"),
                "folder": payload.get("folder", ""),
                "filename": payload.get("filename", ""),
                "heading": payload.get("heading", ""),
                "chunk": payload.get("chunk_text", ""),
                "tags": payload.get("tags", []),
            }
            for _, score, payload in fused
        ]

    @mcp.tool()
    def indexing_status() -> dict:
        """Return how many note chunks are currently indexed in the vector database."""
        collection = config.active_collection
        if qdrant is None:
            return {"collection": collection, "status": "standby — indexer lock held by another container"}
        try:
            info = qdrant.get_collection(collection)
            count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None) or {}
            return {
                "collection": collection,
                "points_indexed": count,
                "has_bm25": SPARSE_NAME in sparse_cfg,
                "status": str(info.status),
            }
        except Exception as e:
            return {"collection": collection, "error": str(e)}
