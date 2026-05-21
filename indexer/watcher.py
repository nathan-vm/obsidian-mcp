import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler

log = logging.getLogger(__name__)


class VaultHandler(FileSystemEventHandler):
    def __init__(self, store, vault_path: Path):
        self.store = store
        self.vault_path = vault_path

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self.store.index_file(Path(event.src_path), self.vault_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self.store.index_file(Path(event.src_path), self.vault_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            rel = str(Path(event.src_path).relative_to(self.vault_path))
            self.store.delete_note_chunks(rel)
            log.info("removed chunks for deleted note: %s", rel)

    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path.endswith(".md"):
                self.store.delete_note_chunks(
                    str(Path(event.src_path).relative_to(self.vault_path))
                )
            if event.dest_path.endswith(".md"):
                self.store.index_file(Path(event.dest_path), self.vault_path)
