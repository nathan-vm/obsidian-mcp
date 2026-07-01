import logging
import re

from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)

from shared.embedding import FastEmbedder
from shared.qdrant_pool import QdrantPool

log = logging.getLogger(__name__)

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


def register(mcp, config, pool: QdrantPool, embedder: FastEmbedder):
    vault = config.vault_path

    @mcp.tool()
    def fulltext_search(query: str, case_sensitive: bool = False, mode: str = "keywords") -> list[dict]:
        """Fast full-text (grep) search across all notes.

        Use this as a fallback when semantic search returns no useful results,
        or when searching for an exact term, title, or tag.

        Args:
            query: Text to search for
            case_sensitive: Default False
            mode: "exact" matches the full query as a substring,
                  "keywords" splits on whitespace and matches notes containing ALL keywords
        """
        flags = re.IGNORECASE if not case_sensitive else 0
        results = []

        if mode == "exact":
            pattern = re.compile(re.escape(query), flags)
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
        else:
            # keywords mode: all words must be present
            words = query.split()
            if not words:
                return []
            patterns = [re.compile(re.escape(w), flags) for w in words]
            for p in vault.rglob("*.md"):
                text = p.read_text(encoding="utf-8", errors="ignore")
                word_matches = [pat.findall(text) for pat in patterns]
                if all(word_matches):
                    total = sum(len(m) for m in word_matches)
                    # snippet around first occurrence of first keyword
                    first_match = patterns[0].search(text)
                    snippet = ""
                    if first_match:
                        pos = first_match.start()
                        start = max(0, pos - 100)
                        end = min(len(text), pos + 200)
                        snippet = text[start:end].replace("\n", " ").strip()
                    results.append(
                        {
                            "path": str(p.relative_to(vault)),
                            "match_count": total,
                            "snippets": [snippet] if snippet else [],
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
        collection = config.active_collection

        try:
            dense_vector = await embedder.aembed_query(query)
        except Exception:
            return _fallback_search(query, n_results)

        filter_condition = _build_filter(tag=tag, path=path, directory=directory)
        fetch = n_results * 3

        try:
            with pool.session() as client:
                info = client.get_collection(collection)
                count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
                if count == 0:
                    return _fallback_search(query, n_results)

                dense_response = client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    using=DENSE_NAME,
                    limit=fetch,
                    query_filter=filter_condition,
                    with_payload=True,
                )
                dense_hits = dense_response.points

                try:
                    bm25_emb = next(_get_bm25().query_embed(query))
                    sparse_response = client.query_points(
                        collection_name=collection,
                        query=SparseVector(
                            indices=bm25_emb.indices.tolist(),
                            values=bm25_emb.values.tolist(),
                        ),
                        using=SPARSE_NAME,
                        limit=fetch,
                        query_filter=filter_condition,
                        with_payload=True,
                    )
                    sparse_hits = sparse_response.points
                except Exception:
                    sparse_hits = []
        except Exception as e:
            log.warning("Qdrant search unavailable: %s — using fulltext fallback", e)
            return _fallback_search(query, n_results)

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
        try:
            with pool.session() as client:
                info = client.get_collection(collection)
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
