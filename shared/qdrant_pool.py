"""Dynamic Qdrant client pool for concurrent container access.

Instead of holding the QdrantClient open permanently (which exclusively locks
the embedded storage), this pool opens/closes the client per operation with
retry logic when the lock is busy.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 0.5  # seconds


class QdrantPool:
    """Context-manager-based access to Qdrant embedded storage with retry."""

    def __init__(self, path: Path, retries: int = _DEFAULT_RETRIES, backoff: float = _DEFAULT_BACKOFF):
        self.path = path
        self.retries = retries
        self.backoff = backoff
        path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def session(self):
        """Open a Qdrant client, yield it, then close to release the lock.

        Retries with exponential backoff if the storage is locked by another process.
        """
        last_err = None
        for attempt in range(self.retries):
            try:
                client = QdrantClient(path=str(self.path))
                try:
                    yield client
                    return
                finally:
                    client.close()
            except RuntimeError as e:
                if "already accessed" in str(e):
                    last_err = e
                    wait = self.backoff * (2**attempt)
                    log.debug("Qdrant lock busy (attempt %d/%d), retrying in %.1fs", attempt + 1, self.retries, wait)
                    time.sleep(wait)
                else:
                    raise
        raise last_err  # type: ignore[misc]
