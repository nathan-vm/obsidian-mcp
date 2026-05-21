import re

from qdrant_client import QdrantClient

from shared.embedding import make_embedder


def register(mcp, config):
    vault = config.vault_path
    qdrant = QdrantClient(url=config.qdrant_url)
    embedder = make_embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        lm_studio_url=config.lm_studio_url,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
    )

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
                results.append({
                    "path": str(p.relative_to(vault)),
                    "match_count": len(matches),
                    "snippets": snippets,
                })
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
    async def search_similar(query: str, n_results: int = 5, tag: str = "") -> list[dict]:
        """Semantic similarity search using embeddings (Qdrant).

        Falls back to fulltext_search automatically if the collection is empty
        or the embedding provider is unreachable.

        Args:
            query: Natural language query describing what you're looking for
            n_results: Number of results (default 5)
            tag: Optional Obsidian tag to filter (e.g. "project/work")
        """
        collection = config.active_collection
        try:
            info = qdrant.get_collection(collection)
            if (info.vectors_count or 0) == 0:
                return _fallback_search(query, n_results)
        except Exception:
            return _fallback_search(query, n_results)

        try:
            vector = await embedder.aembed_query(query)
        except Exception:
            return _fallback_search(query, n_results)

        filter_condition = None
        if tag:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            filter_condition = Filter(
                must=[FieldCondition(key="tags", match=MatchValue(value=tag))]
            )

        hits = qdrant.search(
            collection_name=collection,
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

    @mcp.tool()
    def indexing_status() -> dict:
        """Return how many note chunks are currently indexed in the vector database."""
        collection = config.active_collection
        try:
            info = qdrant.get_collection(collection)
            return {
                "collection": collection,
                "vectors_indexed": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"collection": collection, "error": str(e)}
