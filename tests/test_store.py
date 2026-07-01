from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest import make_mock_bm25, make_mock_client, make_mock_pool


def _make_store(tmp_path, client, embedder, bm25=None):
    from store import QdrantStore

    if bm25 is None:
        bm25 = make_mock_bm25()

    mock_pool = make_mock_pool(client)

    with patch("store.QdrantPool", return_value=mock_pool), patch("store.SparseTextEmbedding", return_value=bm25):
        store = QdrantStore(
            path=tmp_path / "qdrant",
            collection="test_col",
            embedder=embedder,
            chunk_size=200,
            chunk_overlap=0,
        )

    store._bm25 = bm25
    return store


class TestInit:
    def test_pool_created_with_path(self, tmp_path, mock_embedder):
        client = make_mock_client()
        qdrant_path = tmp_path / "qdrant"
        with (
            patch("store.QdrantPool") as mock_pool_cls,
            patch("store.SparseTextEmbedding"),
        ):
            from store import QdrantStore

            QdrantStore(path=qdrant_path, collection="col", embedder=mock_embedder, chunk_size=200, chunk_overlap=0)

        mock_pool_cls.assert_called_once_with(qdrant_path)

    def test_embedding_dim_starts_uncached(self, qdrant_store):
        assert qdrant_store._embedding_dim is None


class TestGetDim:
    def test_calls_embed_query_once(self, qdrant_store, mock_embedder):
        mock_embedder.embed_query.return_value = [0.0] * 10
        dim = qdrant_store._get_dim()
        assert dim == 10
        mock_embedder.embed_query.assert_called_once_with("probe")

    def test_caches_result(self, qdrant_store, mock_embedder):
        mock_embedder.embed_query.return_value = [0.0] * 8
        qdrant_store._get_dim()
        qdrant_store._get_dim()
        assert mock_embedder.embed_query.call_count == 1


class TestGetBm25:
    def test_lazy_loads_on_first_call(self, tmp_path, mock_client, mock_embedder):
        fresh_bm25 = make_mock_bm25()
        store = _make_store(tmp_path, mock_client, mock_embedder, bm25=fresh_bm25)
        store._bm25 = None  # reset so the cold path runs

        with patch("store.SparseTextEmbedding", return_value=fresh_bm25) as mock_cls:
            result = store._get_bm25()

        mock_cls.assert_called_once_with(model_name="Qdrant/bm25", cache_dir="/app/models")
        assert result is fresh_bm25

    def test_caches_after_first_call(self, qdrant_store):
        with patch("store.SparseTextEmbedding") as mock_cls:
            qdrant_store._get_bm25()
            qdrant_store._get_bm25()
        mock_cls.assert_not_called()  # already cached in fixture


class TestBm25Vector:
    def test_returns_sparse_vector(self, qdrant_store):
        from qdrant_client.models import SparseVector

        vec = qdrant_store._bm25_vector("hello world")
        assert isinstance(vec, SparseVector)
        assert vec.indices == [1, 2, 3]
        assert vec.values == [0.5, 0.3, 0.2]


class TestEnsureCollection:
    def _col_info(self, sparse_vectors, points_count=0):
        info = MagicMock()
        info.config.params.sparse_vectors = sparse_vectors
        info.points_count = points_count
        info.vectors_count = None
        return info

    def test_creates_collection_when_missing(self, tmp_path, mock_embedder):
        client = make_mock_client(collections=[])
        mock_embedder.embed_query.return_value = [0.0] * 4
        store = _make_store(tmp_path, client, mock_embedder)

        result = store.ensure_collection()

        assert result is True
        client.create_collection.assert_called_once()
        assert client.create_payload_index.call_count == 5  # one per _INDEXED_FIELDS

    def test_returns_false_when_collection_has_data(self, tmp_path, mock_embedder):
        client = make_mock_client(collections=["test_col"])
        client.get_collections.return_value.collections[0].name = "test_col"
        client.get_collection.return_value = self._col_info(
            sparse_vectors={"text-sparse": MagicMock()}, points_count=10
        )
        store = _make_store(tmp_path, client, mock_embedder)

        result = store.ensure_collection()

        assert result is False
        client.create_collection.assert_not_called()

    def test_returns_true_when_collection_exists_but_empty(self, tmp_path, mock_embedder):
        client = make_mock_client(collections=["test_col"])
        client.get_collections.return_value.collections[0].name = "test_col"
        client.get_collection.return_value = self._col_info(sparse_vectors={"text-sparse": MagicMock()}, points_count=0)
        store = _make_store(tmp_path, client, mock_embedder)

        result = store.ensure_collection()

        assert result is True

    def test_recreates_collection_missing_sparse_vectors(self, tmp_path, mock_embedder):
        client = make_mock_client(collections=["test_col"])
        client.get_collections.return_value.collections[0].name = "test_col"
        client.get_collection.return_value = self._col_info(sparse_vectors=None)
        mock_embedder.embed_query.return_value = [0.0] * 4
        store = _make_store(tmp_path, client, mock_embedder)

        store.ensure_collection()

        client.delete_collection.assert_called_once_with("test_col")
        client.create_collection.assert_called_once()


class TestDeleteNoteChunks:
    def test_calls_client_delete(self, qdrant_store, mock_client):
        qdrant_store.delete_note_chunks("notes/foo.md")
        mock_client.delete.assert_called_once()

    def test_swallows_exceptions(self, qdrant_store, mock_client):
        mock_client.delete.side_effect = RuntimeError("qdrant down")
        qdrant_store.delete_note_chunks("notes/foo.md")  # must not raise


class TestMoveNoteChunks:
    def test_no_results_returns_early(self, qdrant_store, mock_client):
        mock_client.scroll.return_value = ([], None)
        qdrant_store.move_note_chunks("old.md", "new.md")
        mock_client.upsert.assert_not_called()

    def test_updates_path_and_reindexes(self, qdrant_store, mock_client):
        point = MagicMock()
        point.id = 99
        point.payload = {"path": "old.md", "chunk_text": "hello"}
        point.vector = {"text-dense": [0.1, 0.2]}
        mock_client.scroll.return_value = ([point], None)

        qdrant_store.move_note_chunks("old.md", "new.md")

        mock_client.delete.assert_called()
        mock_client.upsert.assert_called_once()
        upserted = mock_client.upsert.call_args[1]["points"]
        assert upserted[0].payload["path"] == "new.md"

    def test_falls_back_to_delete_on_error(self, qdrant_store, mock_client):
        mock_client.scroll.side_effect = RuntimeError("timeout")
        qdrant_store.move_note_chunks("old.md", "new.md")
        mock_client.delete.assert_called_once()  # fallback delete


class TestIndexFile:
    def test_indexes_markdown_file(self, qdrant_store, mock_client, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "note.md"
        note.write_text("# Hello\n\nContent here.")

        qdrant_store.index_file(note, vault)

        mock_client.upsert.assert_called_once()
        points = mock_client.upsert.call_args[1]["points"]
        assert len(points) >= 1
        assert points[0].payload["path"] == "note.md"

    def test_logs_error_on_exception(self, qdrant_store, mock_client, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        missing = vault / "ghost.md"  # doesn't exist → read_text raises

        qdrant_store.index_file(missing, vault)  # must not raise

        mock_client.upsert.assert_not_called()


class TestFullReindex:
    def test_indexes_all_md_files(self, qdrant_store, mock_client, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "a.md").write_text("# A\n\nContent A.")
        (vault / "b.md").write_text("# B\n\nContent B.")
        (vault / "ignore.txt").write_text("not a note")

        qdrant_store.full_reindex(vault)

        assert mock_client.upsert.call_count == 2

    def test_empty_vault(self, qdrant_store, mock_client, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        qdrant_store.full_reindex(vault)

        mock_client.upsert.assert_not_called()
