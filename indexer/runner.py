import logging
import time
from pathlib import Path

from store import QdrantStore
from watchdog.observers import Observer
from watcher import VaultHandler

log = logging.getLogger(__name__)


def run_indexer(store: QdrantStore, vault_path: Path, observer_interval: float) -> None:
    """Full reindex then watch for changes. Designed to run in a background thread."""
    needs_reindex = store.ensure_collection()
    if needs_reindex:
        store.full_reindex(vault_path)
    else:
        log.info("index already populated — skipping full reindex, starting watcher")

    handler = VaultHandler(store, vault_path)
    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.start()
    log.info("watching vault: %s (interval=%.1fs)", vault_path, observer_interval)

    try:
        while True:
            time.sleep(observer_interval)
    except Exception:
        observer.stop()
    observer.join()
