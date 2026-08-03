from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.notes import register


class FakeMCP:
    """Captures functions passed to @mcp.tool() by name, bypassing FastMCP."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def notes(tmp_path: Path):
    mcp = FakeMCP()
    config = SimpleNamespace(vault_path=tmp_path)
    register(mcp, config)
    return SimpleNamespace(vault=tmp_path, **mcp.tools)


class TestListNotes:
    def test_empty_vault(self, notes):
        assert notes.list_notes() == []

    def test_lists_markdown_files_recursively(self, notes):
        (notes.vault / "a.md").write_text("a")
        (notes.vault / "Sub").mkdir()
        (notes.vault / "Sub/b.md").write_text("b")
        (notes.vault / "not_markdown.txt").write_text("x")

        result = notes.list_notes()
        paths = {r["path"] for r in result}
        assert paths == {"a.md", "Sub/b.md"}

    def test_scoped_to_directory(self, notes):
        (notes.vault / "Sub").mkdir()
        (notes.vault / "Sub/b.md").write_text("b")
        (notes.vault / "a.md").write_text("a")

        result = notes.list_notes(directory="Sub")
        assert [r["path"] for r in result] == ["Sub/b.md"]

    def test_missing_directory_returns_empty(self, notes):
        assert notes.list_notes(directory="Nope") == []


class TestReadNote:
    def test_reads_full_content_when_small(self, notes):
        (notes.vault / "n.md").write_text("line1\nline2\nline3\n")
        result = notes.read_note("n.md")
        assert result["content"] == "line1\nline2\nline3\n"
        assert result["total_lines"] == 3
        assert result["truncated"] is False

    def test_pagination_offset_and_limit(self, notes):
        (notes.vault / "n.md").write_text("line1\nline2\nline3\nline4\n")
        result = notes.read_note("n.md", offset=1, limit=2)
        assert result["content"] == "line2\nline3\n"
        assert result["returned_lines"] == 2
        assert result["truncated"] is True

    def test_limit_capped_at_max_read_lines(self, notes, monkeypatch):
        import tools.notes as notes_mod

        monkeypatch.setattr(notes_mod, "MAX_READ_LINES", 2)
        (notes.vault / "n.md").write_text("l1\nl2\nl3\nl4\n")
        result = notes.read_note("n.md", limit=100)
        assert result["returned_lines"] == 2
        assert result["truncated"] is True

    def test_offset_past_end_returns_empty(self, notes):
        (notes.vault / "n.md").write_text("line1\n")
        result = notes.read_note("n.md", offset=10)
        assert result["content"] == ""
        assert result["returned_lines"] == 0


class TestWriteNote:
    def test_writes_new_file_creating_parents(self, notes):
        result = notes.write_note("Sub/n.md", "hello")
        assert (notes.vault / "Sub/n.md").read_text() == "hello"
        assert result == "Written: Sub/n.md"

    def test_overwrites_existing_file(self, notes):
        (notes.vault / "n.md").write_text("old")
        notes.write_note("n.md", "new")
        assert (notes.vault / "n.md").read_text() == "new"


class TestDeleteNote:
    def test_deletes_existing_file(self, notes):
        (notes.vault / "n.md").write_text("x")
        result = notes.delete_note("n.md")
        assert not (notes.vault / "n.md").exists()
        assert result == "Deleted: n.md"

    def test_raises_if_missing(self, notes):
        with pytest.raises(FileNotFoundError):
            notes.delete_note("nope.md")


class TestMoveNote:
    def test_moves_file(self, notes):
        (notes.vault / "a.md").write_text("content")
        result = notes.move_note("a.md", "Sub/b.md")
        assert not (notes.vault / "a.md").exists()
        assert (notes.vault / "Sub/b.md").read_text() == "content"
        assert "a.md" in result and "Sub/b.md" in result

    def test_raises_if_source_missing(self, notes):
        with pytest.raises(FileNotFoundError):
            notes.move_note("nope.md", "b.md")

    def test_prunes_empty_source_dir_after_move(self, notes):
        (notes.vault / "Sub").mkdir()
        (notes.vault / "Sub/a.md").write_text("content")
        notes.move_note("Sub/a.md", "b.md")
        assert not (notes.vault / "Sub").exists()

    def test_prunes_empty_ancestor_dirs_up_to_vault_root(self, notes):
        (notes.vault / "A/B/C").mkdir(parents=True)
        (notes.vault / "A/B/C/a.md").write_text("content")
        notes.move_note("A/B/C/a.md", "b.md")
        assert not (notes.vault / "A").exists()

    def test_does_not_prune_dir_with_remaining_files(self, notes):
        (notes.vault / "Sub").mkdir()
        (notes.vault / "Sub/a.md").write_text("content")
        (notes.vault / "Sub/keep.md").write_text("keep me")
        notes.move_note("Sub/a.md", "b.md")
        assert (notes.vault / "Sub").exists()
        assert (notes.vault / "Sub/keep.md").exists()

    def test_does_not_prune_vault_root_itself(self, notes):
        (notes.vault / "a.md").write_text("content")
        notes.move_note("a.md", "b.md")
        assert notes.vault.exists()

    def test_move_within_same_dir_does_not_delete_it(self, notes):
        (notes.vault / "Sub").mkdir()
        (notes.vault / "Sub/a.md").write_text("content")
        notes.move_note("Sub/a.md", "Sub/b.md")
        assert (notes.vault / "Sub").exists()
        assert (notes.vault / "Sub/b.md").read_text() == "content"


class TestGetNoteMetadata:
    def test_extracts_frontmatter_tags_and_wikilinks(self, notes):
        (notes.vault / "n.md").write_text(
            "---\n"
            "tags:\n"
            "  - project/work\n"
            "---\n\n"
            "Body mentions #inline/tag and links to [[Other Note]] and [[Other Note|label]].\n"
        )
        result = notes.get_note_metadata("n.md")
        assert result["frontmatter"]["tags"] == ["project/work"]
        assert set(result["tags"]) == {"project/work", "inline/tag"}
        assert result["wikilinks"] == ["Other Note"]
        assert result["word_count"] > 0

    def test_string_frontmatter_tag_normalized_to_list(self, notes):
        (notes.vault / "n.md").write_text("---\ntags: solo-tag\n---\nbody\n")
        result = notes.get_note_metadata("n.md")
        assert "solo-tag" in result["tags"]

    def test_no_frontmatter_or_links(self, notes):
        (notes.vault / "n.md").write_text("just plain text")
        result = notes.get_note_metadata("n.md")
        assert result["frontmatter"] == {}
        assert result["tags"] == []
        assert result["wikilinks"] == []


class TestUpdateNote:
    def test_single_unique_match(self, notes):
        (notes.vault / "n.md").write_text("hello world")
        result = notes.update_note("n.md", [{"old_string": "world", "new_string": "there"}])
        assert (notes.vault / "n.md").read_text() == "hello there"
        assert "1 edit(s)" in result

    def test_multiple_edits_applied_in_sequence(self, notes):
        (notes.vault / "n.md").write_text("aaa bbb ccc")
        notes.update_note(
            "n.md",
            [
                {"old_string": "aaa", "new_string": "AAA"},
                {"old_string": "ccc", "new_string": "CCC"},
            ],
        )
        assert (notes.vault / "n.md").read_text() == "AAA bbb CCC"

    def test_second_edit_can_target_text_introduced_by_first(self, notes):
        (notes.vault / "n.md").write_text("original")
        notes.update_note(
            "n.md",
            [
                {"old_string": "original", "new_string": "intermediate"},
                {"old_string": "intermediate", "new_string": "final"},
            ],
        )
        assert (notes.vault / "n.md").read_text() == "final"

    def test_raises_if_note_missing(self, notes):
        with pytest.raises(FileNotFoundError):
            notes.update_note("nope.md", [{"old_string": "a", "new_string": "b"}])

    def test_raises_if_old_string_not_found(self, notes):
        (notes.vault / "n.md").write_text("hello world")
        with pytest.raises(ValueError, match="not found"):
            notes.update_note("n.md", [{"old_string": "missing", "new_string": "x"}])

    def test_raises_if_ambiguous_without_replace_all(self, notes):
        (notes.vault / "n.md").write_text("foo bar foo")
        with pytest.raises(ValueError, match="matches 2 locations"):
            notes.update_note("n.md", [{"old_string": "foo", "new_string": "x"}])

    def test_replace_all_replaces_every_occurrence(self, notes):
        (notes.vault / "n.md").write_text("foo bar foo")
        notes.update_note("n.md", [{"old_string": "foo", "new_string": "baz", "replace_all": True}])
        assert (notes.vault / "n.md").read_text() == "baz bar baz"

    def test_raises_if_old_and_new_identical(self, notes):
        (notes.vault / "n.md").write_text("hello world")
        with pytest.raises(ValueError, match="identical"):
            notes.update_note("n.md", [{"old_string": "world", "new_string": "world"}])

    def test_raises_if_old_string_empty(self, notes):
        (notes.vault / "n.md").write_text("hello world")
        with pytest.raises(ValueError, match="must not be empty"):
            notes.update_note("n.md", [{"old_string": "", "new_string": "x"}])

    def test_file_not_left_partially_written_on_mid_batch_failure(self, notes):
        (notes.vault / "n.md").write_text("hello world")
        with pytest.raises(ValueError):
            notes.update_note(
                "n.md",
                [
                    {"old_string": "hello", "new_string": "hi"},
                    {"old_string": "missing", "new_string": "x"},
                ],
            )
        # first edit's in-memory effect never got written to disk
        assert (notes.vault / "n.md").read_text() == "hello world"
