import tomllib

from indexer.chunker import chunk_id, chunk_markdown


def test_chunk_id_is_deterministic():
    assert chunk_id("notes/foo.md", 0) == chunk_id("notes/foo.md", 0)
    assert chunk_id("notes/foo.md", 0) != chunk_id("notes/foo.md", 1)
    assert chunk_id("notes/foo.md", 0) != chunk_id("notes/bar.md", 0)


def test_chunk_id_fits_in_63_bits():
    assert 0 <= chunk_id("a.md", 0) < 2**63


def test_chunk_markdown_basic():
    text = "# Hello\n\nThis is a paragraph."
    chunks = chunk_markdown(text, "notes/foo.md", chunk_size=200, chunk_overlap=0)
    assert len(chunks) >= 1
    assert chunks[0]["path"] == "notes/foo.md"
    assert chunks[0]["filename"] == "foo"
    assert chunks[0]["folder"] == "notes"
    assert "Hello" in chunks[0]["heading"]


def test_chunk_markdown_tags_from_frontmatter():
    text = "---\ntags:\n  - dev/python\n---\n\n# Note\n\nContent."
    chunks = chunk_markdown(text, "test.md", chunk_size=200, chunk_overlap=0)
    assert "dev/python" in chunks[0]["tags"]


def test_chunk_markdown_inline_tags():
    # The inline-tag regex captures [\w/]+ so hyphens terminate the match;
    # #inline-tag captures "inline" and #dev/python captures "dev/python".
    text = "# Note\n\nContent with #inline and #dev/python tags."
    chunks = chunk_markdown(text, "test.md", chunk_size=200, chunk_overlap=0)
    assert "inline" in chunks[0]["tags"]
    assert "dev/python" in chunks[0]["tags"]


def test_chunk_markdown_ancestor_folders():
    chunks = chunk_markdown("# T\n\nBody", "a/b/c.md", chunk_size=200, chunk_overlap=0)
    assert "a/b" in chunks[0]["folders"]
    assert "a" in chunks[0]["folders"]


def test_chunk_markdown_splits_large_content():
    paras = [f"Paragraph {i} with enough words to fill space." for i in range(20)]
    text = "# Section\n\n" + "\n\n".join(paras)
    chunks = chunk_markdown(text, "big.md", chunk_size=10, chunk_overlap=0)
    assert len(chunks) > 1


def test_chunk_markdown_empty_file_fallback():
    chunks = chunk_markdown("", "empty.md", chunk_size=200, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == ""


def test_version_is_semver():
    import re

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", version), f"Version {version!r} is not semver"
