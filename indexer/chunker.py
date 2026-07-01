import hashlib
import re
from pathlib import Path

import frontmatter


def chunk_id(rel_path: str, index: int) -> int:
    h = int(hashlib.md5(f"{rel_path}:{index}".encode()).hexdigest(), 16)
    return h % (2**63)


def chunk_markdown(text: str, rel_path: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    post = frontmatter.loads(text)
    content = post.content

    yaml_tags = post.metadata.get("tags", [])
    if isinstance(yaml_tags, str):
        yaml_tags = [yaml_tags]
    inline_tags = re.findall(r"(?<!\S)#([\w/]+)", content)
    all_tags = list(set(yaml_tags + inline_tags))

    p = Path(rel_path)
    filename = p.stem
    folder = str(p.parent) if str(p.parent) != "." else ""
    # All ancestor directories for hierarchical filtering
    ancestors = []
    current = p.parent
    while str(current) not in (".", ""):
        ancestors.append(str(current))
        current = current.parent

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

    chunks: list[dict] = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        current: list[str] = []
        current_words = 0
        for para in paragraphs:
            words = len(para.split())
            if current_words + words > chunk_size and current:
                chunks.append(
                    {
                        "path": rel_path,
                        "folder": folder,
                        "folders": ancestors,
                        "filename": filename,
                        "heading": heading,
                        "chunk_text": "\n\n".join(current),
                        "tags": all_tags,
                    }
                )
                current = current[-1:] if chunk_overlap > 0 else []
                current_words = len(current[0].split()) if current else 0
            current.append(para)
            current_words += words
        if current:
            chunks.append(
                {
                    "path": rel_path,
                    "folder": folder,
                    "folders": ancestors,
                    "filename": filename,
                    "heading": heading,
                    "chunk_text": "\n\n".join(current),
                    "tags": all_tags,
                }
            )

    if not chunks:
        chunks = [
            {
                "path": rel_path,
                "folder": folder,
                "folders": ancestors,
                "filename": filename,
                "heading": "",
                "chunk_text": text[:2000],
                "tags": all_tags,
            }
        ]
    return chunks
