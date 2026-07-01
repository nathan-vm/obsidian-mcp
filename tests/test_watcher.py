from pathlib import Path
from unittest.mock import MagicMock

import pytest

from watcher import VaultHandler


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture
def mock_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler(mock_store: MagicMock, vault_path: Path) -> VaultHandler:
    return VaultHandler(store=mock_store, vault_path=vault_path)


def _event(src: str, is_dir: bool = False, dest: str | None = None) -> MagicMock:
    e = MagicMock()
    e.src_path = src
    e.is_directory = is_dir
    if dest is not None:
        e.dest_path = dest
    return e


class TestOnCreated:
    def test_md_file_triggers_index(self, handler, mock_store, vault_path):
        path = str(vault_path / "note.md")
        handler.on_created(_event(path))
        mock_store.index_file.assert_called_once_with(Path(path), vault_path)

    def test_non_md_file_is_ignored(self, handler, mock_store, vault_path):
        handler.on_created(_event(str(vault_path / "file.txt")))
        mock_store.index_file.assert_not_called()

    def test_directory_is_ignored(self, handler, mock_store, vault_path):
        handler.on_created(_event(str(vault_path / "subdir"), is_dir=True))
        mock_store.index_file.assert_not_called()


class TestOnModified:
    def test_md_file_triggers_index(self, handler, mock_store, vault_path):
        path = str(vault_path / "note.md")
        handler.on_modified(_event(path))
        mock_store.index_file.assert_called_once_with(Path(path), vault_path)

    def test_non_md_file_is_ignored(self, handler, mock_store, vault_path):
        handler.on_modified(_event(str(vault_path / "image.png")))
        mock_store.index_file.assert_not_called()

    def test_directory_is_ignored(self, handler, mock_store, vault_path):
        handler.on_modified(_event(str(vault_path / "dir"), is_dir=True))
        mock_store.index_file.assert_not_called()


class TestOnDeleted:
    def test_md_file_removes_chunks(self, handler, mock_store, vault_path):
        path = str(vault_path / "note.md")
        handler.on_deleted(_event(path))
        mock_store.delete_note_chunks.assert_called_once_with("note.md")

    def test_non_md_file_is_ignored(self, handler, mock_store, vault_path):
        handler.on_deleted(_event(str(vault_path / "doc.pdf")))
        mock_store.delete_note_chunks.assert_not_called()

    def test_directory_is_ignored(self, handler, mock_store, vault_path):
        handler.on_deleted(_event(str(vault_path / "folder"), is_dir=True))
        mock_store.delete_note_chunks.assert_not_called()


class TestOnMoved:
    def test_md_to_md_moves_chunks(self, handler, mock_store, vault_path):
        src = str(vault_path / "old.md")
        dst = str(vault_path / "new.md")
        handler.on_moved(_event(src, dest=dst))
        mock_store.move_note_chunks.assert_called_once_with("old.md", "new.md")

    def test_md_to_non_md_deletes_chunks(self, handler, mock_store, vault_path):
        src = str(vault_path / "note.md")
        dst = str(vault_path / "note.txt")
        handler.on_moved(_event(src, dest=dst))
        mock_store.delete_note_chunks.assert_called_once_with("note.md")
        mock_store.move_note_chunks.assert_not_called()

    def test_non_md_to_md_indexes_file(self, handler, mock_store, vault_path):
        src = str(vault_path / "draft.txt")
        dst = str(vault_path / "note.md")
        handler.on_moved(_event(src, dest=dst))
        mock_store.index_file.assert_called_once_with(Path(dst), vault_path)
        mock_store.move_note_chunks.assert_not_called()

    def test_directory_move_is_ignored(self, handler, mock_store, vault_path):
        src = str(vault_path / "dir_a")
        dst = str(vault_path / "dir_b")
        handler.on_moved(_event(src, is_dir=True, dest=dst))
        mock_store.move_note_chunks.assert_not_called()
        mock_store.index_file.assert_not_called()
        mock_store.delete_note_chunks.assert_not_called()
