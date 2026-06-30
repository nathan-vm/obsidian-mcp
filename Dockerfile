FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY shared/ shared/
COPY indexer/ indexer/
COPY obsidian-mcp/ obsidian-mcp/

# Pre-download the embedding model so no network access is needed at runtime.
# The model is baked into the image; only Qdrant index data lives on the volume.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5', cache_dir='/app/models')"

ENV VAULT_PATH=/vault
ENV DATA_PATH=/data

VOLUME ["/vault", "/data"]

CMD ["uv", "run", "python", "obsidian-mcp/server.py"]
