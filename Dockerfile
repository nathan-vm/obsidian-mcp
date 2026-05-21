FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync --all-groups

COPY obsidian-mcp/ obsidian-mcp/
COPY indexer/ indexer/
