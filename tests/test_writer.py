import json

import pytest

from voxtex.io.history import History
from voxtex.io.writer import Writer, read_file

BODY = "\\chapter{Results}\n\nSome text.\n\n%> HERE\n\nAfter the marker.\n"


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "thesis"
    root.mkdir()
    (root / "ch.tex").write_text(BODY, encoding="utf-8")
    return root


def test_insert_above_marker_backup_and_history(proj):
    w = Writer(proj, History(proj, "s1"))
    f = proj / "ch.tex"
    span = w.insert_at_marker(f, "First block.", "b1")
    w.insert_at_marker(f, "Second block.", "b2")
    assert read_file(f) == (
        "\\chapter{Results}\n\nSome text.\n\nFirst block.\n\nSecond block.\n\n%> HERE\n\n"
        "After the marker.\n"
    )
    assert span.text == "First block."
    backups = list((proj / ".voxtex" / "backups").iterdir())
    assert [b.name for b in backups] == ["s1-ch.tex"]  # once per session and file
    assert backups[0].read_text() == BODY
    entries = [json.loads(x) for x in (proj / ".voxtex" / "history.jsonl").read_text().splitlines()]
    assert [e["edit"]["block"] for e in entries] == ["b1", "b2"]
    assert not [p for p in proj.iterdir() if p.is_file() and p.name.endswith(".voxtex")]  # no temps


def test_marker_added_when_missing(proj):
    f = proj / "plain.tex"
    f.write_text("Only text.", encoding="utf-8")
    w = Writer(proj)
    w.insert_at_marker(f, "New.", "b1")
    assert read_file(f) == "Only text.\n\nNew.\n\n%> HERE\n"


def test_crlf_and_bom_preserved(proj):
    f = proj / "win.tex"
    f.write_bytes("\ufeffIntro.\r\n\r\n%> HERE\r\n".encode())
    w = Writer(proj)
    span = w.insert_at_marker(f, "Line one\nline two", "b1")
    raw = f.read_bytes()
    assert raw.startswith("\ufeff".encode())
    assert raw.decode() == "\ufeffIntro.\r\n\r\nLine one\r\nline two\r\n\r\n%> HERE\r\n"
    new = w.replace_span(span, "Changed\nagain", "replace", "b1")
    assert new is not None
    assert "Changed\r\nagain\r\n" in f.read_bytes().decode()


def test_replace_and_delete_survive_author_edits_elsewhere(proj):
    f = proj / "ch.tex"
    w = Writer(proj)
    span = w.insert_at_marker(f, "Score 0.63.", "b1")
    f.write_text("% the author added a line\n" + read_file(f), encoding="utf-8")
    span = w.replace_span(span, "Score 0.61.", "replace", "b1")
    assert "Score 0.61." in read_file(f) and "0.63" not in read_file(f)
    assert w.replace_span(span, "", "delete", "b1") is not None
    assert read_file(f) == "% the author added a line\n" + BODY


def test_replace_returns_none_when_author_changed_the_block(proj):
    f = proj / "ch.tex"
    w = Writer(proj)
    span = w.insert_at_marker(f, "Score 0.63.", "b1")
    f.write_text(read_file(f).replace("Score 0.63.", "Score 0.63, roughly."), encoding="utf-8")
    assert w.replace_span(span, "Score 0.61.", "replace", "b1") is None


def test_undo_restores_exact_content(proj):
    f = proj / "ch.tex"
    w = Writer(proj)
    s1 = w.insert_at_marker(f, "One.", "b1")
    w.insert_at_marker(f, "Two.", "b2")
    w.replace_span(s1, "Uno.", "replace", "b1")
    after_three = read_file(f)
    done, problem = w.undo(1)
    assert problem is None and [e.kind for e in done] == ["replace"]
    assert "One." in read_file(f) and "Uno." not in read_file(f)
    done, _ = Writer(proj).undo(5)  # a new session can undo older writes
    assert [e.block for e in done] == ["b2", "b1"]
    assert read_file(f) == BODY
    assert after_three != BODY
    assert Writer(proj).undo(1) == ([], None)


def test_undo_after_author_edits_and_when_impossible(proj):
    f = proj / "ch.tex"
    w = Writer(proj)
    w.insert_at_marker(f, "Mine.", "b1")
    f.write_text("Author's new first line.\n" + read_file(f), encoding="utf-8")
    assert w.undo(1)[1] is None
    assert read_file(f) == "Author's new first line.\n" + BODY

    w.insert_at_marker(f, "Mine again.", "b2")
    f.write_text(read_file(f).replace("Mine again.", "Rewritten by author."), encoding="utf-8")
    done, problem = w.undo(1)
    assert done == [] and "changed too much" in problem
