import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from chunker import chunk_id, chunk_markdown

log = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, url: str, collection: str, embedder, chunk_size: int, chunk_overlap: int):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_dim: int | None = None

    def _get_dim(self) -> int:
        if self._embedding_dim is None:
            probe = self.embedder.embed_query("probe")
            self._embedding_dim = len(probe)
            log.info("embedding dim detected: %d", self._embedding_dim)
        return self._embedding_dim

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
            dim = self._get_dim()
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("created collection '%s' (dim=%d)", self.collection, dim)

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

    def move_note_chunks(self, old_rel: str, new_rel: str) -> None:
        """Update path for all chunks of a moved note, reusing existing vectors."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="path", match=MatchValue(value=old_rel))]
                ),
                with_vectors=True,
                with_payload=True,
                limit=1000,
            )
            if not results:
                return

            old_ids = [p.id for p in results]
            new_points = []
            for i, point in enumerate(results):
                new_payload = dict(point.payload)
                new_payload["path"] = new_rel
                new_points.append(
                    PointStruct(id=chunk_id(new_rel, i), vector=point.vector, payload=new_payload)
                )

            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=old_ids),
            )
            if new_points:
                self.client.upsert(collection_name=self.collection, points=new_points)
            log.info("moved %d chunks: %s → %s", len(new_points), old_rel, new_rel)
        except Exception as e:
            log.error("move failed (%s → %s): %s — falling back to reindex", old_rel, new_rel, e)
            self.delete_note_chunks(old_rel)

    def index_file(self, file_path: Path, vault_path: Path) -> None:
        rel = str(file_path.relative_to(vault_path))
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_markdown(text, rel, self.chunk_size, self.chunk_overlap)
            points = []
            for i, chunk in enumerate(chunks):
                context = f"{chunk['path']}\n{chunk['heading']}\n\n{chunk['chunk_text']}"
                vector = self.embedder.embed_query(context)
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
