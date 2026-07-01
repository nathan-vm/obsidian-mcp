import asyncio
from unittest.mock import MagicMock, patch

from shared.embedding import FastEmbedder


def _make_embedder(model_name: str = "test-model") -> tuple[FastEmbedder, MagicMock]:
    mock_model = MagicMock()
    with patch("shared.embedding.TextEmbedding", return_value=mock_model):
        embedder = FastEmbedder(model_name=model_name, cache_dir="/tmp/test-models")
    return embedder, mock_model


class TestFastEmbedder:
    def test_embed_query_returns_list_of_floats(self):
        embedder, mock_model = _make_embedder()
        mock_model.embed.return_value = iter([[0.1, 0.2, 0.3]])

        result = embedder.embed_query("hello world")

        mock_model.embed.assert_called_once_with(["hello world"])
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_returns_plain_list(self):
        embedder, mock_model = _make_embedder()
        mock_model.embed.return_value = iter([[1.0, 2.0]])

        result = embedder.embed_query("test")

        assert isinstance(result, list)

    def test_aembed_query_returns_same_as_embed_query(self):
        embedder, mock_model = _make_embedder()
        mock_model.embed.return_value = iter([[0.5, 0.6]])

        result = asyncio.run(embedder.aembed_query("async test"))

        assert result == [0.5, 0.6]

    def test_embed_documents_returns_list_of_vectors(self):
        embedder, mock_model = _make_embedder()
        mock_model.embed.return_value = iter([[0.1, 0.2], [0.3, 0.4]])

        result = embedder.embed_documents(["doc one", "doc two"])

        mock_model.embed.assert_called_once_with(["doc one", "doc two"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_documents_empty_input(self):
        embedder, mock_model = _make_embedder()
        mock_model.embed.return_value = iter([])

        result = embedder.embed_documents([])

        assert result == []
