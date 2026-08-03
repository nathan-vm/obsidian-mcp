import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tools.search import _build_filter, _rrf, register


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


def make_pool(client: MagicMock) -> MagicMock:
    pool = MagicMock()

    @contextmanager
    def _session():
        yield client

    pool.session = _session
    return pool


@pytest.fixture
def search_tools(tmp_path: Path):
    mcp = FakeMCP()
    config = SimpleNamespace(vault_path=tmp_path, active_collection="test_col")
    client = MagicMock()
    pool = make_pool(client)
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock()
    register(mcp, config, pool, embedder)
    return SimpleNamespace(vault=tmp_path, client=client, pool=pool, embedder=embedder, config=config, **mcp.tools)


class TestBuildFilter:
    def test_no_criteria_returns_none(self):
        assert _build_filter(tag="", path="", directory="") is None

    def test_builds_filter_with_all_criteria(self):
        f = _build_filter(tag="work", path="a.md", directory="Sub")
        assert len(f.must) == 3


class TestRRF:
    def test_fuses_and_ranks_by_combined_score(self):
        dense = [SimpleNamespace(id=1, payload={"path": "a.md"}), SimpleNamespace(id=2, payload={"path": "b.md"})]
        sparse = [SimpleNamespace(id=2, payload={"path": "b.md"}), SimpleNamespace(id=3, payload={"path": "c.md"})]
        fused = _rrf(dense, sparse)
        ids = [f[0] for f in fused]
        # id=2 appears in both lists so should rank first
        assert ids[0] == 2
        assert set(ids) == {1, 2, 3}

    def test_empty_inputs(self):
        assert _rrf([], []) == []


class TestFulltextSearch:
    def test_exact_mode_matches_substring(self, search_tools):
        (search_tools.vault / "a.md").write_text("this note mentions FooBar in it")
        (search_tools.vault / "b.md").write_text("no match here")
        results = search_tools.fulltext_search("FooBar", mode="exact")
        assert len(results) == 1
        assert results[0]["path"] == "a.md"
        assert results[0]["match_count"] == 1

    def test_exact_mode_case_insensitive_by_default(self, search_tools):
        (search_tools.vault / "a.md").write_text("FOOBAR here")
        results = search_tools.fulltext_search("foobar", mode="exact")
        assert len(results) == 1

    def test_exact_mode_case_sensitive_excludes_mismatch(self, search_tools):
        (search_tools.vault / "a.md").write_text("FOOBAR here")
        results = search_tools.fulltext_search("foobar", mode="exact", case_sensitive=True)
        assert results == []

    def test_keywords_mode_requires_all_words(self, search_tools):
        (search_tools.vault / "a.md").write_text("alpha beta gamma")
        (search_tools.vault / "b.md").write_text("alpha only")
        results = search_tools.fulltext_search("alpha beta")
        assert [r["path"] for r in results] == ["a.md"]

    def test_keywords_mode_empty_query_returns_empty(self, search_tools):
        (search_tools.vault / "a.md").write_text("content")
        assert search_tools.fulltext_search("   ") == []

    def test_results_sorted_by_match_count_desc(self, search_tools):
        (search_tools.vault / "a.md").write_text("foo")
        (search_tools.vault / "b.md").write_text("foo foo foo")
        results = search_tools.fulltext_search("foo")
        assert [r["path"] for r in results] == ["b.md", "a.md"]


class TestIndexingStatus:
    def test_returns_collection_stats(self, search_tools):
        info = MagicMock(status="green")
        info.points_count = 42
        info.config.params.sparse_vectors = {"text-sparse": object()}
        search_tools.client.get_collection.return_value = info

        result = search_tools.indexing_status()
        assert result["points_indexed"] == 42
        assert result["has_bm25"] is True
        assert result["collection"] == "test_col"

    def test_returns_error_on_exception(self, search_tools):
        search_tools.client.get_collection.side_effect = RuntimeError("boom")
        result = search_tools.indexing_status()
        assert result["collection"] == "test_col"
        assert "boom" in result["error"]


class TestSearchSimilar:
    def test_falls_back_to_fulltext_when_embedding_fails(self, search_tools):
        (search_tools.vault / "a.md").write_text("hello world")
        search_tools.embedder.aembed_query.side_effect = RuntimeError("no embedder")

        results = asyncio.run(search_tools.search_similar("hello"))
        assert results
        assert results[0]["note"] == "fallback — vector index not ready"

    def test_falls_back_when_collection_empty(self, search_tools):
        (search_tools.vault / "a.md").write_text("hello world")
        search_tools.embedder.aembed_query.return_value = [0.1, 0.2]
        info = MagicMock(points_count=0, vectors_count=0)
        search_tools.client.get_collection.return_value = info

        results = asyncio.run(search_tools.search_similar("hello"))
        assert results
        assert results[0]["note"] == "fallback — vector index not ready"

    def test_falls_back_on_qdrant_exception(self, search_tools):
        (search_tools.vault / "a.md").write_text("hello world")
        search_tools.embedder.aembed_query.return_value = [0.1, 0.2]
        search_tools.client.get_collection.side_effect = RuntimeError("qdrant down")

        results = asyncio.run(search_tools.search_similar("hello"))
        assert results
        assert results[0]["note"] == "fallback — vector index not ready"

    def test_returns_fused_hybrid_results(self, search_tools, monkeypatch):
        search_tools.embedder.aembed_query.return_value = [0.1, 0.2]
        info = MagicMock(points_count=5)
        search_tools.client.get_collection.return_value = info

        dense_hit = SimpleNamespace(id=1, payload={"path": "a.md", "chunk_text": "hi", "tags": ["x"]})
        search_tools.client.query_points.return_value = SimpleNamespace(points=[dense_hit])

        # bm25 embedding raises -> sparse_hits falls back to [] internally
        import tools.search as search_mod

        monkeypatch.setattr(search_mod, "_get_bm25", lambda: (_ for _ in ()).throw(RuntimeError("no bm25")))

        results = asyncio.run(search_tools.search_similar("hello", n_results=1))
        assert results[0]["path"] == "a.md"
        assert results[0]["chunk"] == "hi"

    def test_fuses_dense_and_sparse_bm25_hits(self, search_tools, monkeypatch):
        search_tools.embedder.aembed_query.return_value = [0.1, 0.2]
        info = MagicMock(points_count=5)
        search_tools.client.get_collection.return_value = info

        dense_hit = SimpleNamespace(id=1, payload={"path": "a.md", "chunk_text": "dense hit", "tags": []})
        sparse_hit = SimpleNamespace(id=2, payload={"path": "b.md", "chunk_text": "sparse hit", "tags": []})
        search_tools.client.query_points.side_effect = [
            SimpleNamespace(points=[dense_hit]),
            SimpleNamespace(points=[sparse_hit]),
        ]

        bm25_emb = SimpleNamespace(indices=MagicMock(tolist=lambda: [1]), values=MagicMock(tolist=lambda: [0.9]))
        fake_bm25 = MagicMock()
        fake_bm25.query_embed.return_value = iter([bm25_emb])

        import tools.search as search_mod

        monkeypatch.setattr(search_mod, "_get_bm25", lambda: fake_bm25)

        results = asyncio.run(search_tools.search_similar("hello", n_results=5))
        paths = {r["path"] for r in results}
        assert paths == {"a.md", "b.md"}
