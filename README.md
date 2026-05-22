# obsidian-mcp

A self-hosted MCP (Model Context Protocol) server that gives AI assistants semantic and full-text search over an Obsidian vault. Runs entirely locally via Docker.

## Architecture

```
Obsidian Vault (host)
        │
        ▼
  ┌─────────────┐       ┌─────────────────┐
  │   indexer   │──────▶│   Qdrant (DB)   │
  └─────────────┘       └────────┬────────┘
                                 │
  ┌─────────────┐                │
  │  obsidian   │◀───────────────┘
  │  mcp server │
  └──────┬──────┘
         │ SSE (port 55000)
         ▼
    Claude / AI client
```

| Service | Role |
|---|---|
| `qdrant` | Vector database — stores embeddings |
| `indexer` | Reads vault, chunks notes, embeds text, upserts into Qdrant. Watches for file changes live |
| `obsidian-mcp` | FastMCP server — exposes tools to AI clients over SSE |

## Quick Start

1. Create the env:
```
cp .env.example .env
```

2. Set VAULT_PATH (and optionally EMBEDDING_PROVIDER)

3. Run the MCP in **full mode**
```bash
docker compose --profile full up --build -d
```

> **MCP only** (no Qdrant/indexer — fulltext search only):
> ```bash
> docker compose up -d
> ```

## Connecting to Claude

**Claude Code (CLI):**
```bash
claude mcp add obsidian-mcp --transport sse http://localhost:55000/sse
```

**VS Code (>MCP: Open User Configuration - `mcp.json`):**
```json
"mcp": {
  "servers": {
    "obsidian-mcp": {
      "type": "sse",
      "url": "http://localhost:55000/sse"
    }
  }
}
```

## Embedding Providers

Set `EMBEDDING_PROVIDER` to one of:

| Provider | Value | Required variable |
|---|---|---|
| LM Studio (default) | `lm_studio` | `LM_STUDIO_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Gemini | `gemini` | `GEMINI_API_KEY` |

Each `EMBEDDING_MODEL` gets its own Qdrant collection (`obsidian_vault__<model_slug>`), so you can switch models freely without data corruption and test multiple models side-by-side.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `VAULT_PATH` | — | **Required.** Absolute path to your Obsidian vault on the host |
| `EMBEDDING_PROVIDER` | `lm_studio` | `lm_studio` \| `openai` \| `gemini` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model name for the chosen provider |
| `LM_STUDIO_URL` | `http://host.docker.internal:1234/v1` | LM Studio local server |
| `OPENAI_API_KEY` | — | For `provider=openai` (e.g. `text-embedding-3-small`) |
| `GEMINI_API_KEY` | — | For `provider=gemini` (e.g. `models/text-embedding-004`) |
| `QDRANT_URL` | `http://qdrant:6333` | Internal Docker URL — don't change |
| `COLLECTION_NAME` | `obsidian_vault` | Base name — actual collection is `<name>__<model_slug>` |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `MCP_PORT` | `55000` | Port exposed to host |
| `OBSERVER_INTERVAL` | `1.0` | Vault watcher poll interval (seconds) |

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
| `search_similar` | Semantic search via Qdrant embeddings. Accepts optional `tag` filter. Falls back to fulltext automatically |
| `fulltext_search` | Regex grep across all notes — use for exact terms, titles, or tags |
| `indexing_status` | How many chunks are currently in the vector index |

## Project Structure

```
obsidian-mcp/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── shared/
│   ├── config.py          # Config dataclass loaded from env
│   └── embedding.py       # LangChain embedder factory (lm_studio/openai/gemini)
├── indexer/
│   ├── main.py            # Entry point: reindex + start watcher
│   ├── chunker.py         # Heading-aware markdown chunker
│   ├── embedder.py        # Delegates to shared.embedding
│   ├── store.py           # QdrantStore — index/move/delete/reindex
│   └── watcher.py         # VaultHandler (watchdog events)
└── obsidian-mcp/
    ├── server.py          # FastMCP server setup
    ├── embedding.py       # Delegates to shared.embedding
    ├── utils.py           # safe_path() path traversal guard
    └── tools/
        ├── notes.py       # CRUD tools
        └── search.py      # Semantic + fulltext search tools
```

## Implementation Notes

- **Chunking**: splits notes by markdown headings, then by paragraph up to `CHUNK_SIZE` words. Tags collected from YAML frontmatter and inline `#tags`.
- **Chunk IDs**: stable MD5 hashes of `path:index` — re-indexing upserts cleanly without duplicates.
- **Move handling**: when a note is moved/renamed, existing vectors are fetched and re-inserted with updated path metadata — no re-embedding needed.
- **Path safety**: `safe_path()` rejects any path that escapes the vault root (path traversal protection).
- **Fallback search**: `search_similar` degrades to `fulltext_search` if Qdrant is empty or the embedding provider is unreachable.
