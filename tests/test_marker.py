from voxtex.io.marker import fallback_location, find_marker, nearby_text


def test_find_marker(sample_thesis):
    loc = find_marker(
        sample_thesis, ["main.tex", "chapters/introduction.tex", "chapters/methodology.tex"]
    )
    assert loc and loc.found
    assert loc.rel == "chapters/methodology.tex"
    assert loc.before.rstrip().endswith("\\end{table}")


def test_no_marker_falls_back_to_last_file(sample_thesis):
    assert find_marker(sample_thesis, ["chapters/introduction.tex"]) is None
    loc = fallback_location(sample_thesis, ["main.tex", "chapters/introduction.tex"])
    assert loc.rel == "chapters/introduction.tex" and not loc.found


def test_nearby_text_cuts_at_line_and_drops_comments():
    before = "\n".join(f"line {i} " + "x" * 50 for i in range(40)) + "\n% a comment\nlast line"
    out = nearby_text(before, 200)
    assert len(out) <= 200
    assert out.endswith("last line")
    assert "% a comment" not in out
    assert out.startswith("line ")
