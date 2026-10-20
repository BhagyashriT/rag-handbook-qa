from rag_handbook_qa.chunking import chunk_markdown


def test_header_splitting_keeps_breadcrumbs():
    text = "# Title\n\n## Alpha\n\nalpha body\n\n### Beta\n\nbeta body\n"
    chunks = chunk_markdown(text, "f.md")
    by_section = {c.section: c.text for c in chunks}
    assert "Title > Alpha" in by_section
    assert "Title > Alpha > Beta" in by_section
    assert "alpha body" in by_section["Title > Alpha"]
    assert all(c.file == "f.md" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_oversized_section_is_windowed_with_overlap():
    body = "word " * 1000  # ~5000 chars
    chunks = chunk_markdown(f"# T\n\n## S\n\n{body}", "f.md", max_chars=1000, overlap=200)
    windowed = [c for c in chunks if c.section == "T > S"]
    assert len(windowed) > 1
    assert all(len(c.text) <= 1000 for c in windowed)
    # overlap: the tail of one window repeats at the head of the next
    assert windowed[0].text[-100:] in windowed[1].text


def test_overlap_must_be_smaller_than_max_chars():
    try:
        chunk_markdown("x", "f.md", max_chars=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
