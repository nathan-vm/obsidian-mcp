import os
import re
import shutil

import frontmatter
from utils import safe_path


def register(mcp, config):
    vault = config.vault_path

    @mcp.tool()
    def list_notes(directory: str = "") -> list[dict]:
        """List all markdown notes in the vault (or a subdirectory).

        Returns relative paths, note names and last-modified timestamps.
        """
        base = safe_path(vault, directory) if directory else vault
        if not base.exists():
            return []
        return [
            {
                "path": str(p.relative_to(vault)),
                "name": p.stem,
                "modified": p.stat().st_mtime,
            }
            for p in sorted(base.rglob("*.md"))
        ]

    @mcp.tool()
    def read_note(path: str) -> str:
        """Read the full markdown content of a note.

        Args:
            path: Path relative to the vault root (e.g. "Projects/myproject.md")
        """
        return safe_path(vault, path).read_text(encoding="utf-8")

    @mcp.tool()
    def write_note(path: str, content: str) -> str:
        """Write (or overwrite) a note. Creates parent directories if needed.

        Args:
            path: Path relative to the vault root
            content: Full markdown content to write
        """
        p = safe_path(vault, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
        return f"Written: {path}"

    @mcp.tool()
    def create_note(path: str, content: str = "", overwrite: bool = False) -> str:
        """Create a new note. Raises an error if it already exists (unless overwrite=True).

        Args:
            path: Path relative to the vault root
            content: Initial markdown content
            overwrite: Set True to replace an existing note
        """
        p = safe_path(vault, path)
        if p.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists — use overwrite=True to replace it")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
        return f"Created: {path}"

    @mcp.tool()
    def delete_note(path: str) -> str:
        """Permanently delete a note.

        Args:
            path: Path relative to the vault root
        """
        p = safe_path(vault, path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {path}")
        p.unlink()
        return f"Deleted: {path}"

    @mcp.tool()
    def move_note(from_path: str, to_path: str) -> str:
        """Move or rename a note.

        Args:
            from_path: Current path relative to the vault root
            to_path: New path relative to the vault root
        """
        src = safe_path(vault, from_path)
        dst = safe_path(vault, to_path)
        if not src.exists():
            raise FileNotFoundError(f"Note not found: {from_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved: {from_path} → {to_path}"

    @mcp.tool()
    def get_note_metadata(path: str) -> dict:
        """Return frontmatter, tags (inline + YAML), and wikilinks for a note.

        Args:
            path: Path relative to the vault root
        """
        text = safe_path(vault, path).read_text(encoding="utf-8")
        post = frontmatter.loads(text)
        yaml_tags = post.metadata.get("tags", [])
        if isinstance(yaml_tags, str):
            yaml_tags = [yaml_tags]
        inline_tags = re.findall(r"(?<!\S)#([\w/]+)", post.content)
        wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", post.content)
        return {
            "path": path,
            "frontmatter": dict(post.metadata),
            "tags": list(set(yaml_tags + inline_tags)),
            "wikilinks": list(set(wikilinks)),
            "word_count": len(post.content.split()),
        }
