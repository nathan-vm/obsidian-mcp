import os
import re
import shutil

import frontmatter
from utils import safe_path

# Hard cap on lines returned by read_note in a single call, regardless of
# the requested limit — keeps a single huge note from blowing the context budget.
MAX_READ_LINES = 2000


def _prune_empty_dirs(start, vault) -> None:
    """Remove start and its ancestors, stopping at the vault root or the first non-empty dir."""
    current = start.resolve()
    vault = vault.resolve()
    while current != vault and vault in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


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
    def read_note(path: str, offset: int = 0, limit: int | None = None) -> dict:
        """Read a note's markdown content, paginated by line.

        Notes longer than MAX_READ_LINES are truncated even without an explicit
        limit — check `truncated` and re-call with a larger `offset` to keep reading.

        Args:
            path: Path relative to the vault root (e.g. "Projects/myproject.md")
            offset: 0-indexed line to start reading from
            limit: Max lines to return (capped at MAX_READ_LINES)
        """
        text = safe_path(vault, path).read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        effective_limit = min(limit, MAX_READ_LINES) if limit is not None else MAX_READ_LINES
        selected = lines[offset : offset + effective_limit]
        return {
            "path": path,
            "content": "".join(selected),
            "total_lines": total_lines,
            "offset": offset,
            "returned_lines": len(selected),
            "truncated": offset + len(selected) < total_lines,
        }

    @mcp.tool()
    def write_note(path: str, content: str) -> str:
        """Write (or overwrite) a note. Creates parent directories if needed.

        For small, targeted changes to an existing note, prefer update_note —
        it avoids resending the whole note body.

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
    def update_note(path: str, edits: list[dict]) -> str:
        """Apply one or more targeted string replacements to an existing note.

        Each edit is {"old_string": str, "new_string": str, "replace_all": bool (optional, default False)}.
        old_string must match the note's current text exactly, including whitespace and
        indentation. It must be unique within the note unless replace_all=True. Edits are
        applied in order, each against the result of the previous one.

        Args:
            path: Path relative to the vault root
            edits: List of edit objects, applied in order
        """
        p = safe_path(vault, path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {path}")
        content = p.read_text(encoding="utf-8")

        for i, edit in enumerate(edits):
            old = edit["old_string"]
            new = edit["new_string"]
            replace_all = edit.get("replace_all", False)

            if old == "":
                raise ValueError(f"Edit {i}: old_string must not be empty")
            if old == new:
                raise ValueError(f"Edit {i}: old_string and new_string are identical")

            count = content.count(old)
            if count == 0:
                raise ValueError(f"Edit {i}: old_string not found in note")
            if count > 1 and not replace_all:
                raise ValueError(
                    f"Edit {i}: old_string matches {count} locations — "
                    "add more surrounding context to make it unique, or set replace_all=True"
                )

            content = content.replace(old, new) if replace_all else content.replace(old, new, 1)

        tmp = p.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
        return f"Updated: {path} ({len(edits)} edit(s) applied)"

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

        After moving, prunes the source's parent directory (and its ancestors,
        up to the vault root) if they were left empty.

        Args:
            from_path: Current path relative to the vault root
            to_path: New path relative to the vault root
        """
        src = safe_path(vault, from_path)
        dst = safe_path(vault, to_path)
        if not src.exists():
            raise FileNotFoundError(f"Note not found: {from_path}")
        src_parent = src.parent
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        _prune_empty_dirs(src_parent, vault)
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
