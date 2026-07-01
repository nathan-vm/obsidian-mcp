import fcntl
import logging
import sys
import threading
from pathlib import Path

# Make shared/ and indexer/ importable from the repo root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # obsidian-mcp/
sys.path.insert(0, str(_ROOT))  # repo root (shared/)
sys.path.insert(0, str(_ROOT / "indexer"))  # indexer/

from mcp.server.fastmcp import FastMCP
from runner import run_indexer
from store import QdrantStore
from tools.notes import register as register_notes
from tools.search import register as register_search

from shared.config import config
from shared.embedding import FastEmbedder

# Logs must go to stderr — stdout is reserved for MCP stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def _try_acquire_indexer_lock(data_path: Path):
    """Try to acquire an exclusive non-blocking lock on the indexer lock file.

    Returns the open file descriptor if the lock was acquired, None otherwise.
    The caller must keep the fd alive for the duration of the process.
    """
    lock_path = data_path / "indexer.lock"
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log.info("indexer lock acquired (%s)", lock_path)
        return fd
    except BlockingIOError:
        fd.close()
        log.warning("indexer lock busy — another container is already indexing (%s)", lock_path)
        log.warning("this container will serve MCP tools but skip indexing (search falls back to fulltext)")
        return None


def main() -> None:
    log.info("data path: %s", config.data_path)
    log.info("vault path: %s", config.vault_path)
    log.info("embedding model: %s", config.embedding_model)

    embedder = FastEmbedder(model_name=config.embedding_model)

    # QdrantStore now uses dynamic locking — no permanent client needed
    store = QdrantStore(
        path=config.qdrant_path,
        collection=config.active_collection,
        embedder=embedder,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    log.info("Qdrant pool ready (dynamic locking)")

    # Try to acquire the indexer lock (only one container runs the watcher at a time)
    lock_fd = _try_acquire_indexer_lock(config.data_path)
    if lock_fd is not None:
        indexer_thread = threading.Thread(
            target=run_indexer,
            args=(store, config.vault_path, config.observer_interval),
            daemon=True,
            name="indexer",
        )
        indexer_thread.start()
        log.info("indexer started (collection=%s)", config.active_collection)
    else:
        log.info("running in search-only mode (indexer lock held by another container)")

    mcp = FastMCP("obsidian")
    register_notes(mcp, config)
    register_search(mcp, config, store.pool, embedder)

    log.info("MCP server ready (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
