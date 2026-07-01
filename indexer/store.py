import logging
from pathlib import Path

from chunker import chunk_id, chunk_markdown
from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from shared.qdrant_pool import QdrantPool

log = logging.getLogger(__name__)

DENSE_NAME = "text-dense"
SPARSE_NAME = "text-sparse"

_INDEXED_FIELDS = ("path", "folder", "folders", "filename", "tags")


class QdrantStore:
    def __init__(self, path: Path, collection: str, embedder, chunk_size: int, chunk_overlap: int):
        self.pool = QdrantPool(path)
        self.collection = collection
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_dim: int | None = None
        self._bm25: SparseTextEmbedding | None = None

    def _get_dim(self) -> int:
        if self._embedding_dim is None:
            probe = self.embedder.embed_query("probe")
            self._embedding_dim = len(probe)
            log.info("embedding dim detected: %d", self._embedding_dim)
        return self._embedding_dim

    def _get_bm25(self) -> SparseTextEmbedding:
        if self._bm25 is None:
            self._bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir="/app/models")
        return self._bm25

    def _bm25_vector(self, text: str) -> SparseVector:
        emb = next(self._get_bm25().embed([text]))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def ensure_collection(self) -> bool:
        """Ensure the collection exists and is configured correctly.

        Returns True if a full reindex is needed (collection was created or recreated),
        False if the collection already has data and can be used as-is.
        """
        with self.pool.session() as client:
            existing = {c.name for c in client.get_collections().collections}
            if self.collection in existing:
                info = client.get_collection(self.collection)
                sparse_cfg = getattr(info.config.params, "sparse_vectors", None) or {}
                if SPARSE_NAME not in sparse_cfg:
                    log.info("collection missing BM25 sparse vectors — recreating (full reindex needed)")
                    client.delete_collection(self.collection)
                else:
                    count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
                    if count > 0:
                        log.info("collection '%s' already has %d chunks — skipping full reindex", self.collection, count)
                        return False
                    log.info("collection '%s' exists but is empty — reindexing", self.collection)
                    return True

            dim = self._get_dim()
            client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE_NAME: VectorParams(size=dim, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))},
            )
            for field in _INDEXED_FIELDS:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            log.info("created collection '%s' (dim=%d, BM25 sparse, payload indexes)", self.collection, dim)
            return True

    def delete_note_chunks(self, rel_path: str) -> None:
        try:
            with self.pool.session() as client:
                client.delete(
                    collection_name=self.collection,
                    points_selector=Filter(must=[FieldCondition(key="path", match=MatchValue(value=rel_path))]),
                )
        except Exception:
            pass

    def move_note_chunks(self, old_rel: str, new_rel: str) -> None:
        """Update path for all chunks of a moved note, reusing existing vectors."""
        try:
            with self.pool.session() as client:
                results, _ = client.scroll(
                    collection_name=self.collection,
                    scroll_filter=Filter(must=[FieldCondition(key="path", match=MatchValue(value=old_rel))]),
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
                    new_points.append(PointStruct(id=chunk_id(new_rel, i), vector=point.vector, payload=new_payload))

                client.delete(
                    collection_name=self.collection,
                    points_selector=PointIdsList(points=old_ids),
                )
                if new_points:
                    client.upsert(collection_name=self.collection, points=new_points)
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
                dense = self.embedder.embed_query(context)
                sparse = self._bm25_vector(context)
                points.append(
                    PointStruct(
                        id=chunk_id(rel, i),
                        vector={DENSE_NAME: dense, SPARSE_NAME: sparse},
                        payload=chunk,
                    )
                )
            with self.pool.session() as client:
                client.delete(
                    collection_name=self.collection,
                    points_selector=Filter(must=[FieldCondition(key="path", match=MatchValue(value=rel))]),
                )
                if points:
                    client.upsert(collection_name=self.collection, points=points)
            log.info("indexed %s (%d chunks)", rel, len(points))
        except Exception as e:
            log.error("failed to index %s: %s", rel, e)

    def full_reindex(self, vault_path: Path) -> None:
        files = list(vault_path.rglob("*.md"))
        log.info("reindexing %d notes...", len(files))
        for f in files:
            self.index_file(f, vault_path)
        log.info("reindex complete")
