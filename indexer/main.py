import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import config
from shared.embedding import FastEmbedder
from runner import run_indexer
from store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    embedder = FastEmbedder(model_name=config.embedding_model)
    store = QdrantStore(
        path=config.qdrant_path,
        collection=config.active_collection,
        embedder=embedder,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    run_indexer(store, config.vault_path, config.observer_interval)


if __name__ == "__main__":
    main()
