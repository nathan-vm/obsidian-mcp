from unittest.mock import MagicMock, patch

import pytest

from shared.qdrant_pool import QdrantPool


class TestQdrantPool:
    def test_session_opens_and_closes_client(self, tmp_path):
        mock_client = MagicMock()
        with patch("shared.qdrant_pool.QdrantClient", return_value=mock_client):
            pool = QdrantPool(path=tmp_path / "qdrant")
            with pool.session() as client:
                assert client is mock_client
            mock_client.close.assert_called_once()

    def test_session_closes_client_on_exception(self, tmp_path):
        mock_client = MagicMock()
        with patch("shared.qdrant_pool.QdrantClient", return_value=mock_client):
            pool = QdrantPool(path=tmp_path / "qdrant")
            with pytest.raises(ValueError):
                with pool.session() as _client:
                    raise ValueError("boom")
            mock_client.close.assert_called_once()

    def test_retries_on_lock_busy(self, tmp_path):
        mock_client = MagicMock()
        call_count = 0

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Storage folder /data/qdrant is already accessed by another instance")
            return mock_client

        with patch("shared.qdrant_pool.QdrantClient", side_effect=_side_effect):
            pool = QdrantPool(path=tmp_path / "qdrant", retries=5, backoff=0.01)
            with pool.session() as client:
                assert client is mock_client
        assert call_count == 3

    def test_raises_after_max_retries(self, tmp_path):
        def _always_locked(**kwargs):
            raise RuntimeError("Storage folder /data/qdrant is already accessed by another instance")

        with patch("shared.qdrant_pool.QdrantClient", side_effect=_always_locked):
            pool = QdrantPool(path=tmp_path / "qdrant", retries=2, backoff=0.01)
            with pytest.raises(RuntimeError, match="already accessed"):
                with pool.session():
                    pass

    def test_non_lock_errors_propagate_immediately(self, tmp_path):
        def _other_error(**kwargs):
            raise RuntimeError("something else entirely")

        with patch("shared.qdrant_pool.QdrantClient", side_effect=_other_error):
            pool = QdrantPool(path=tmp_path / "qdrant", retries=5, backoff=0.01)
            with pytest.raises(RuntimeError, match="something else"):
                with pool.session():
                    pass

    def test_creates_directory(self, tmp_path):
        pool_path = tmp_path / "nested" / "qdrant"
        QdrantPool(path=pool_path)
        assert pool_path.exists()
