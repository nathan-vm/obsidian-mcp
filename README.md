# obsidian-mcp

A self-hosted MCP (Model Context Protocol) server that gives AI assistants semantic and full-text search over an Obsidian vault. Runs as a single Docker container — no external services required.

## Architecture

```
Obsidian Vault (host)
        │  volume mount
        ▼
┌──────────────────────────────┐
│        obsidian-mcp          │
│                              │
│  ┌──────────┐  ┌──────────┐ │
│  │ indexer  │  │   MCP    │ │
│  │ (thread) │  │  server  │ │
│  └────┬─────┘  └────┬─────┘ │
│       │              │       │
│  ┌────▼──────────────▼─────┐ │
│  │   Qdrant (embedded)     │ │
│  │   fastembed ONNX model  │ │
│  └─────────────────────────┘ │
└──────────────────────────────┘
        │  stdio
        ▼
   Claude / AI client
```

Everything runs inside a single container:
- **Indexer** — background thread that chunks and embeds vault notes into Qdrant on startup, then watches for file changes
- **MCP server** — FastMCP over stdio, exposes tools to the AI client
- **Qdrant (embedded)** — in-process vector DB, no separate service needed
- **fastembed** — local ONNX embedding model, no API key needed

The Qdrant index is stored on a Docker volume and persists across container restarts.

---

## Production

### 1. Build the image

```bash
docker build -t obsidian-mcp .
```

This bakes the `nomic-ai/nomic-embed-text-v1.5` model (~270 MB) into the image so no network access is needed at runtime.

### 2. Configure your MCP client

**Claude Code:**
```bash
claude mcp add obsidian \
  -- docker run -i --rm \
  -e VAULT_PATH=/vault \
  -e DATA_PATH=/data \
  -v /absolute/path/to/your/vault:/vault:ro \
  -v obsidian-mcp-data:/data \
  obsidian-mcp:latest
```

**Claude Desktop / VS Code (`mcp.json`):**
```json
{
  "mcpServers": {
    "obsidian": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "VAULT_PATH=/vault",
        "-e", "DATA_PATH=/data",
        "-v", "/absolute/path/to/your/vault:/vault:ro",
        "-v", "obsidian-mcp-data:/data",
        "obsidian-mcp:latest"
      ]
    }
  }
}
```

The named volume `obsidian-mcp-data` is created automatically by Docker on first run and persists the Qdrant index across container restarts.

---

## Development

Use `docker-compose.yml` + `Dockerfile.dev` for a live-reload dev environment. Source directories are bind-mounted so changes to `.py` files are picked up immediately via `watchmedo`.

### 1. Set your vault path

Copy `env.example` to a local env file and set `VAULT_PATH` to the absolute path of your vault on the host.

### 2. Build the dev image

```bash
docker compose build
```

The model is **not** baked into the dev image — it is downloaded on first run and cached in the `model_cache` Docker volume.

### 3. Configure your MCP client to use docker compose

**Claude Code:**
```bash
claude mcp add obsidian \
  -- docker compose \
  -f /absolute/path/to/obsidian-mcp/docker-compose.yml \
  run --rm obsidian-mcp
```

**Claude Desktop / VS Code (`mcp.json`):**
```json
{
  "mcpServers": {
    "obsidian": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/obsidian-mcp/docker-compose.yml",
        "run", "--rm",
        "obsidian-mcp"
      ]
    }
  }
}
```

Each time the MCP client starts a session it spawns a fresh container. `watchmedo` restarts the server process inside the running container when you edit source files mid-session.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `VAULT_PATH` | — | **Required.** Path inside the container where the vault is mounted |
| `DATA_PATH` | `/data` | Where to store the Qdrant index — mount a volume here for persistence |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Any fastembed-compatible model |
| `COLLECTION_NAME` | `obsidian_vault` | Base name — actual collection is `<name>__<model_slug>` |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `OBSERVER_INTERVAL` | `1.0` | Vault watcher poll interval (seconds) |

---

## MCP Tools

### Notes
| Tool | Description |
|---|---|
| `list_notes` | List all `.md` files in vault or a subdirectory |
| `read_note` | Read full content of a note by relative path |
| `write_note` | Write/overwrite a note (creates parent dirs) |
| `create_note` | Create a note, errors if it already exists unless `overwrite=True` |
| `delete_note` | Permanently delete a note |
| `move_note` | Move or rename a note |
| `get_note_metadata` | Return frontmatter, tags, wikilinks, word count |

### Search
| Tool | Description |
|---|---|
| `search_similar` | Hybrid semantic + BM25 search with RRF fusion. Accepts `tag`, `path`, `directory` filters. Falls back to fulltext if index is not ready |
| `fulltext_search` | Regex grep across all notes — use for exact terms, titles, or tags |
| `indexing_status` | How many chunks are currently in the vector index |

---

## Project Structure

```
obsidian-mcp/
├── Dockerfile           # Production image (model baked in)
├── Dockerfile.dev       # Dev image (model cached in volume, source bind-mounted)
├── docker-compose.yml   # Dev environment
├── pyproject.toml
├── shared/
│   ├── config.py        # Config dataclass loaded from env
│   └── embedding.py     # FastEmbedder wrapping fastembed ONNX
├── indexer/
│   ├── main.py          # Standalone entry point (calls runner.run_indexer)
│   ├── runner.py        # run_indexer() — reindex + watchdog loop
│   ├── chunker.py       # Heading-aware markdown chunker
│   ├── store.py         # QdrantStore — index/move/delete/reindex
│   └── watcher.py       # VaultHandler (watchdog events)
└── obsidian-mcp/
    ├── server.py        # All-in-one entry point: starts indexer thread + MCP stdio server
    ├── utils.py         # safe_path() path traversal guard
    └── tools/
        ├── notes.py     # CRUD tools
        └── search.py    # Semantic + fulltext search tools
```

## Implementation Notes

- **Chunking**: splits notes by markdown headings, then by paragraph up to `CHUNK_SIZE` words. Tags collected from YAML frontmatter and inline `#tags`.
- **Chunk IDs**: stable MD5 hashes of `path:index` — re-indexing upserts cleanly without duplicates.
- **Hybrid search**: combines dense cosine similarity (fastembed) with BM25 sparse vectors, fused via Reciprocal Rank Fusion (RRF).
- **Move handling**: when a note is moved/renamed, existing vectors are fetched and re-inserted with updated path metadata — no re-embedding needed.
- **Path safety**: `safe_path()` rejects any path that escapes the vault root (path traversal protection).
- **Fallback search**: `search_similar` degrades to `fulltext_search` if the index is empty or not yet ready (e.g. during initial indexing).

