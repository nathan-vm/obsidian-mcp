import hashlib
import logging
import os
import re
import time
from pathlib import Path

import frontmatter
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/vault"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://host.docker.internal:1234/v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "obsidian_vault")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 768))

qdrant = QdrantClient(url=QDRANT_URL)


def get_embedding(text: str) -> list[float]:
    r = httpx.post(
        f"{LM_STUDIO_URL}/embeddings",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def chunk_markdown(text: str, rel_path: str) -> list[dict]:
    post = frontmatter.loads(text)
    content = post.content

    yaml_tags = post.metadata.get("tags", [])
    if isinstance(yaml_tags, str):
        yaml_tags = [yaml_tags]
    inline_tags = re.findall(r"(?<!\S)#([\w/]+)", content)
    all_tags = list(set(yaml_tags + inline_tags))

    # Split by headings, tracking breadcrumb context
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    last_pos = 0
    heading_stack: list[str] = []

    for m in heading_re.finditer(content):
        body = content[last_pos : m.start()].strip()
        if body:
            sections.append((" > ".join(heading_stack), body))
        level = len(m.group(1))
        heading_stack = heading_stack[: level - 1] + [m.group(2).strip()]
        last_pos = m.end()

    tail = content[last_pos:].strip()
    if tail:
        sections.append((" > ".join(heading_stack), tail))

    # Split large sections by paragraph with overlap
    chunks: list[dict] = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        current: list[str] = []
        current_words = 0
        for para in paragraphs:
            words = len(para.split())
            if current_words + words > CHUNK_SIZE and current:
                chunks.append(
                    {
                        "path": rel_path,
                        "heading": heading,
                        "chunk_text": "\n\n".join(current),
                        "tags": all_tags,
                    }
                )
                current = current[-1:] if CHUNK_OVERLAP > 0 else []
                current_words = len(current[0].split()) if current else 0
            current.append(para)
            current_words += words
        if current:
            chunks.append(
                {
                    "path": rel_path,
                    "heading": heading,
                    "chunk_text": "\n\n".join(current),
                    "tags": all_tags,
                }
            )

    if not chunks:
        chunks = [
            {"path": rel_path, "heading": "", "chunk_text": text[:2000], "tags": all_tags}
        ]
    return chunks


def chunk_id(rel_path: str, index: int) -> int:
    h = int(hashlib.md5(f"{rel_path}:{index}".encode()).hexdigest(), 16)
    return h % (2**63)


def delete_note_chunks(rel_path: str) -> None:
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="path", match=MatchValue(value=rel_path))]
            ),
        )
    except Exception:
        pass


def index_file(file_path: Path) -> None:
    rel = str(file_path.relative_to(VAULT_PATH))
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_markdown(text, rel)
        points = []
        for i, chunk in enumerate(chunks):
            context = f"{chunk['path']}\n{chunk['heading']}\n\n{chunk['chunk_text']}"
            vector = get_embedding(context)
            points.append(PointStruct(id=chunk_id(rel, i), vector=vector, payload=chunk))
        delete_note_chunks(rel)
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        log.info("indexed %s (%d chunks)", rel, len(points))
    except Exception as e:
        log.error("failed to index %s: %s", rel, e)


def ensure_collection() -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info("created collection '%s' (dim=%d)", COLLECTION_NAME, EMBEDDING_DIM)


def full_reindex() -> None:
    files = list(VAULT_PATH.rglob("*.md"))
    log.info("reindexing %d notes...", len(files))
    for f in files:
        index_file(f)
    log.info("reindex complete")


class VaultHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            index_file(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            index_file(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            rel = str(Path(event.src_path).relative_to(VAULT_PATH))
            delete_note_chunks(rel)
            log.info("removed chunks for deleted note: %s", rel)

    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path.endswith(".md"):
                delete_note_chunks(str(Path(event.src_path).relative_to(VAULT_PATH)))
            if event.dest_path.endswith(".md"):
                index_file(Path(event.dest_path))


def wait_for_qdrant(retries: int = 30, delay: int = 2) -> None:
    for i in range(retries):
        try:
            qdrant.get_collections()
            log.info("connected to Qdrant")
            return
        except Exception:
            log.info("waiting for Qdrant (%d/%d)...", i + 1, retries)
            time.sleep(delay)
    raise RuntimeError("could not connect to Qdrant")


if __name__ == "__main__":
    wait_for_qdrant()
    ensure_collection()
    full_reindex()

    observer = Observer()
    observer.schedule(VaultHandler(), str(VAULT_PATH), recursive=True)
    observer.start()
    log.info("watching vault: %s", VAULT_PATH)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
