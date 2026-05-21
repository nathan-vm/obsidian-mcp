import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from chunker import chunk_id, chunk_markdown
from embedder import get_embedding

log = logging.getLogger(__name__)


class QdrantStore:
    def __init__(
        self,
        url: str,
        collection: str,
        embedding_dim: int,
        lm_studio_url: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.lm_studio_url = lm_studio_url
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def wait_for_connection(self, retries: int = 30, delay: int = 2) -> None:
        for i in range(retries):
            try:
                self.client.get_collections()
                log.info("connected to Qdrant")
                return
            except Exception:
                log.info("waiting for Qdrant (%d/%d)...", i + 1, retries)
                time.sleep(delay)
        raise RuntimeError("could not connect to Qdrant")

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            log.info("created collection '%s' (dim=%d)", self.collection, self.embedding_dim)

    def delete_note_chunks(self, rel_path: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[FieldCondition(key="path", match=MatchValue(value=rel_path))]
                ),
            )
        except Exception:
            pass

    def index_file(self, file_path: Path, vault_path: Path) -> None:
        rel = str(file_path.relative_to(vault_path))
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_markdown(text, rel, self.chunk_size, self.chunk_overlap)
            points = []
            for i, chunk in enumerate(chunks):
                context = f"{chunk['path']}\n{chunk['heading']}\n\n{chunk['chunk_text']}"
                vector = get_embedding(context, self.lm_studio_url, self.embedding_model)
                points.append(PointStruct(id=chunk_id(rel, i), vector=vector, payload=chunk))
            self.delete_note_chunks(rel)
            if points:
                self.client.upsert(collection_name=self.collection, points=points)
            log.info("indexed %s (%d chunks)", rel, len(points))
        except Exception as e:
            log.error("failed to index %s: %s", rel, e)

    def full_reindex(self, vault_path: Path) -> None:
        files = list(vault_path.rglob("*.md"))
        log.info("reindexing %d notes...", len(files))
        for f in files:
            self.index_file(f, vault_path)
        log.info("reindex complete")
