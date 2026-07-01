from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def make_mock_client(collections: list[str] | None = None) -> MagicMock:
    """Return a QdrantClient mock with no collections by default."""
    client = MagicMock()
    existing = [MagicMock(name=n) for n in (collections or [])]
    client.get_collections.return_value = MagicMock(collections=existing)
    return client


def make_mock_bm25() -> MagicMock:
    sparse_emb = MagicMock()
    sparse_emb.indices.tolist.return_value = [1, 2, 3]
    sparse_emb.values.tolist.return_value = [0.5, 0.3, 0.2]
    bm25 = MagicMock()
    # side_effect returns a fresh iterator on every call (return_value would be exhausted)
    bm25.embed.side_effect = lambda texts: iter([sparse_emb])
    return bm25


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 4
    return embedder


@pytest.fixture
def mock_client() -> MagicMock:
    return make_mock_client()


@pytest.fixture
def mock_bm25() -> MagicMock:
    return make_mock_bm25()


@pytest.fixture
def qdrant_store(tmp_path: Path, mock_client: MagicMock, mock_embedder: MagicMock, mock_bm25: MagicMock):
    """QdrantStore with all external dependencies mocked out."""
    from store import QdrantStore

    with (
        patch("store.QdrantClient", return_value=mock_client),
        patch("store.SparseTextEmbedding", return_value=mock_bm25),
        patch.object(Path, "mkdir"),
    ):
        store = QdrantStore(
            path=tmp_path / "qdrant",
            collection="test_col",
            embedder=mock_embedder,
            chunk_size=200,
            chunk_overlap=0,
        )

    store.client = mock_client
    store._bm25 = mock_bm25
    return store
