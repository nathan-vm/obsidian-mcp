import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from watchdog.observers import Observer

from shared.config import config
from store import QdrantStore
from watcher import VaultHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    store = QdrantStore(
        url=config.qdrant_url,
        collection=config.collection_name,
        embedding_dim=config.embedding_dim,
        lm_studio_url=config.lm_studio_url,
        embedding_model=config.embedding_model,
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
    log.info("watching vault: %s (interval=%.1fs)", config.vault_path, config.observer_interval)

    try:
        while True:
            time.sleep(config.observer_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
