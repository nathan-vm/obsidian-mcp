from pathlib import Path

import pytest
from utils import safe_path


def test_safe_path_resolves_within_vault(tmp_path: Path):
    result = safe_path(tmp_path, "Projects/note.md")
    assert result == (tmp_path / "Projects/note.md").resolve()


def test_safe_path_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="Path escapes vault"):
        safe_path(tmp_path, "../outside.md")


def test_safe_path_rejects_absolute_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="Path escapes vault"):
        safe_path(tmp_path, "../../etc/passwd")
