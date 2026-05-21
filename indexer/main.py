import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from watchdog.observers import Observer

from shared.config import config
from shared.embedding import make_embedder
from store import QdrantStore
from watcher import VaultHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    embedder = make_embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        lm_studio_url=config.lm_studio_url,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
    )

    store = QdrantStore(
        url=config.qdrant_url,
        collection=config.active_collection,
        embedder=embedder,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    store.wait_for_connection()
    store.ensure_collection()
    store.full_reindex(config.vault_path)

    handler = VaultHandler(store, config.vault_path)
    observer = Observer()
    observer.schedule(handler, str(config.vault_path), recursive=True)
    observer.start()
    log.info(
        "watching vault: %s (collection=%s, interval=%.1fs)",
        config.vault_path,
        config.active_collection,
        config.observer_interval,
    )

    try:
        while True:
            time.sleep(config.observer_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
