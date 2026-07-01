FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY shared/ shared/
COPY indexer/ indexer/
COPY obsidian-mcp/ obsidian-mcp/

# Pre-download embedding models so no network access is needed at runtime.
# Both the dense and BM25 sparse models are baked into the image.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5', cache_dir='/app/models')"
RUN uv run python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25', cache_dir='/app/models')"

ENV VAULT_PATH=/vault
ENV DATA_PATH=/data

VOLUME ["/vault", "/data"]

CMD ["uv", "run", "python", "obsidian-mcp/server.py"]
