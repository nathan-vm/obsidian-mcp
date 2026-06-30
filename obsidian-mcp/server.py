import logging
import sys
import threading
from pathlib import Path

# Make shared/ and indexer/ importable from the repo root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))   # obsidian-mcp/
sys.path.insert(0, str(_ROOT))                    # repo root (shared/)
sys.path.insert(0, str(_ROOT / "indexer"))        # indexer/

from mcp.server.fastmcp import FastMCP

from shared.config import config
from shared.embedding import FastEmbedder
from store import QdrantStore
from runner import run_indexer
from tools.notes import register as register_notes
from tools.search import register as register_search

# Logs must go to stderr — stdout is reserved for MCP stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("data path: %s", config.data_path)
    log.info("vault path: %s", config.vault_path)
    log.info("embedding model: %s", config.embedding_model)

    embedder = FastEmbedder(model_name=config.embedding_model)
    store = QdrantStore(
        path=config.qdrant_path,
        collection=config.active_collection,
        embedder=embedder,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    indexer_thread = threading.Thread(
        target=run_indexer,
        args=(store, config.vault_path, config.observer_interval),
        daemon=True,
        name="indexer",
    )
    indexer_thread.start()
    log.info("indexer started (collection=%s)", config.active_collection)

    mcp = FastMCP("obsidian")
    register_notes(mcp, config)
    register_search(mcp, config, store.client, embedder)

    log.info("MCP server ready (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
