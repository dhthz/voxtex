from pathlib import Path

from rich.console import Console

from voxtex.engine.buffer import Block
from voxtex.engine.decide import Delete, Insert, Replace, UncertainInsert
from voxtex.engine.engine import Step
from voxtex.io.writer import TextSpan
from voxtex.ui import UI, view_of, word_changes


def _file(tmp_path: Path, text: str) -> Path:
    f = tmp_path / "ch.tex"
    f.write_text(text, encoding="utf-8")
    return f


def test_word_changes_marks_only_the_changed_number():
    old, new = "reaches 0.71 on the test", "reaches 0.73 on the test"
    marks_old, marks_new = word_changes(old, new)
    assert [old[i:j] for i, j in marks_old] == ["71"]
    assert [new[i:j] for i, j in marks_new] == ["73"]


def test_insert_view_has_line_numbers_and_context(tmp_path):
    f = _file(tmp_path, "a\nb\nNew text.\n\n%> HERE\n")
    step = Step(Insert(action="insert", latex="New text."), Block("b1", "New text."))
    v = view_of(step, "", TextSpan(f, 4, "New text."), "ch.tex")
    assert v.verb == "Write" and v.summary == "Added 1 line at line 3 (b1)"
    assert [(ln.number, ln.sign) for ln in v.lines] == [
        (1, " "),
        (2, " "),
        (3, "+"),
        (4, " "),
        (5, " "),
    ]


def test_replace_view_pairs_old_and_new(tmp_path):
    f = _file(tmp_path, "x\nIt is 0.73.\n")
    step = Step(
        Replace(action="replace", block_id="b1", latex="It is 0.73."), Block("b1", "It is 0.73.")
    )
    v = view_of(step, "It is 0.71.", TextSpan(f, 2, "It is 0.73."), "ch.tex")
    assert v.verb == "Update"
    removed = next(ln for ln in v.lines if ln.sign == "-")
    assert [removed.text[i:j] for i, j in removed.marks] == ["71"]


def test_delete_and_dry_run_views(tmp_path):
    f = _file(tmp_path, "x\n\n%> HERE\n")
    step = Step(Delete(action="delete", block_id="b1"), Block("b1", "Gone."))
    v = view_of(step, "Gone.", TextSpan(f, 2, ""), "ch.tex")
    assert v.summary == "Removed b1 (1 line)"
    note = Block("b2", "See X.", note="which year?")
    v = view_of(
        Step(UncertainInsert(action="uncertain_insert", latex="See X.", note="which year?"), note),
        "",
        None,
        "",
    )
    assert v.status == "check" and "would add 2 lines" in v.summary


def test_status_shows_speech_in_a_box():
    con = Console(record=True, width=60)
    con.print(UI(con).status("Listening", "groq-gptoss", speech="hello there", hints="ctrl+c"))
    out = con.export_text()
    assert "Listening…" in out and "> hello there" in out and "╭" in out


def test_long_blocks_are_cut(tmp_path):
    body = "\n".join(f"line {i}" for i in range(30))
    f = _file(tmp_path, body + "\n")
    step = Step(Insert(action="insert", latex=body), Block("b1", body))
    con = Console(record=True, width=80)
    UI(con).write(view_of(step, "", TextSpan(f, 0, body), "ch.tex"))
    out = con.export_text()
    assert "more lines" in out and "line 29" not in out
