from fastembed import TextEmbedding

_MODEL_CACHE_DIR = "/app/models"


class FastEmbedder:
    """Local ONNX embedder via fastembed — no external API required."""

    def __init__(self, model_name: str, cache_dir: str = _MODEL_CACHE_DIR):
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_query(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._model.embed(texts)]
