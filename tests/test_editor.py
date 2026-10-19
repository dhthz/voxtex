import pytest

from voxtex.editor.highlight import char_to_byte, latex_spans
from voxtex.editor.place import advance, find_unique, place_block


def test_advance():
    assert advance((2, 3), "ab") == (2, 5)
    assert advance((2, 3), "ab\ncd\nxyz") == (4, 3)


def test_on_an_empty_line_the_block_is_its_own_paragraph():
    lines = ["First paragraph.", "", "", "Second."]
    p = place_block(lines, (1, 0), "New text.")
    assert p.at == (1, 0) and p.text == "\nNew text."  # blank lines kept on both sides
    assert (p.start, p.end) == ((2, 0), (2, 9))
    # A single blank line between paragraphs: blank lines are added around the new one.
    p = place_block(["A.", "", "B."], (1, 0), "New.")
    assert p.text == "\nNew.\n"


def test_empty_line_right_after_text_gets_a_blank_line_before():
    lines = ["First paragraph.", ""]
    p = place_block(lines, (1, 0), "New.")
    assert p.text == "\nNew."
    assert p.start == (2, 0)


def test_a_sentence_goes_at_the_cursor_inside_a_paragraph():
    lines = ["One. Three."]
    p = place_block(lines, (0, 4), "Two.")
    assert p.at == (0, 4) and p.text == " Two."
    lines = ["One."]
    assert place_block(lines, (0, 4), "Two.").text == " Two."


def test_a_table_goes_after_the_paragraph():
    lines = ["Para line one", "line two", "", "Next"]
    table = "\\begin{table}\n\\end{table}"
    p = place_block(lines, (0, 3), table)
    assert p.at == (1, 8) and p.text == "\n\n" + table
    assert p.start == (3, 0) and p.end == (4, 11)


def test_find_unique():
    assert find_unique("a b a", "b") == 2
    assert find_unique("a b a", "a") is None
    assert find_unique("abc", "") is None


def test_latex_spans():
    line = r"See \cite{x} and $a+b$ % note"
    names = {line[s:e]: n for s, e, n in latex_spans(line)}
    assert names[r"\cite"] == "latex.ref"
    assert names["$a+b$"] == "latex.math"
    assert names["% note"] == "latex.comment"
    assert latex_spans("% voxtex-check: which year?") == [(0, 27, "voxtex.check")]
    assert latex_spans(r"50\% done") == [(2, 4, "latex.command")]


def test_char_to_byte_handles_greek():
    assert char_to_byte("aβc") == [0, 1, 3, 4]


# --- the app, without audio ----------------------------------------------------------------


class FakeWhisper:
    """Hears whatever the test says next."""

    name = "fake"

    def __init__(self):
        self.next = ""

    def transcribe(self, audio, prompt=""):
        from voxtex.audio.stt import Transcript

        return Transcript(self.next, 0.1, len(audio) / 16000, "fake", "el")


class ScriptedLLM:
    def __init__(self):
        self.replies: list = []
        self.seen: list = []

    def __call__(self, p, key, messages, json_mode, timeout):
        import json

        from voxtex.providers import Reply

        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(json.dumps(reply), input_tokens=500, output_tokens=50)


@pytest.fixture
def editor(sample_thesis, tmp_path):
    from voxtex.config import ProviderConfig, load_config
    from voxtex.editor.app import EditorApp
    from voxtex.project.index import load_index
    from voxtex.providers import Chain

    idx, _ = load_index(sample_thesis)
    config = load_config(idx.root_path, global_dir=tmp_path / "noconfig")
    llm = ScriptedLLM()
    local = ProviderConfig(name="local", base_url="http://localhost:11434/v1", model="m")
    path = sample_thesis / "chapters" / "methodology.tex"
    app = EditorApp(idx, config, Chain([local], send=llm), path, audio=False)
    app.transcriber = FakeWhisper()
    return app, path, llm


async def talk(app, pilot, said: str) -> None:
    """One press-to-talk round: a second of 'audio' that the fake hears as `said`."""
    import numpy as np

    app.transcriber.next = said
    app.process(np.full(16000, 0.1, dtype=np.float32))
    for _ in range(200):
        await pilot.pause(0.01)
        if app.phase == "ready":
            return
    raise AssertionError(f"stuck in {app.phase}")


async def test_dictation_is_written_at_the_cursor(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))  # an empty last line
        llm.replies.append({"action": "insert", "latex": "It reaches 0.71."})
        await talk(app, pilot, "it reaches 0.71")
        assert "It reaches 0.71." in path.read_text()  # saved right away
        system = llm.seen[0][0]["content"]
        assert "Keep the speaker's words" in system
        assert "DICTATION:\nit reaches 0.71" in llm.seen[0][1]["content"]


async def test_a_selection_is_changed_as_asked(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        llm.replies.append({"action": "insert", "latex": "It reaches 0.71."})
        await talk(app, pilot, "it reaches 0.71")
        row = next(i for i, ln in enumerate(area.document.lines) if "0.71" in ln)
        col = area.document.lines[row].index("reaches")
        area.selection = ((row, col), (row, col + len("reaches")))
        llm.replies.append({"action": "replace", "block_id": "sel", "latex": "attains"})
        await talk(app, pilot, "πιο επίσημο")
        assert "It attains 0.71." in path.read_text()
        assert "INSTRUCTION:\nπιο επίσημο" in llm.seen[1][1]["content"]
        assert "SELECTED TEXT [sel]" in llm.seen[1][1]["content"]


async def test_nothing_said_is_ever_dropped(editor):
    from voxtex.providers import Unavailable

    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        llm.replies.append({"action": "ignore"})  # the model refuses: the words still go in
        await talk(app, pilot, "is this working")
        assert "is this working" in path.read_text()
        llm.replies += [Unavailable("down")] * 3  # no model at all: written as heard
        await talk(app, pilot, "50% of cases")
        assert r"50\% of cases" in path.read_text()


async def test_doubts_become_a_toast_not_a_latex_comment(editor):
    app, path, llm = editor
    async with app.run_test(notifications=True) as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        llm.replies.append(
            {"action": "uncertain_insert", "latex": "It reaches 0.71.", "note": "0.71 or 0.17?"}
        )
        await talk(app, pilot, "it reaches 0.71")
        assert "voxtex-check" not in path.read_text()
        assert any("0.71 or 0.17?" in n.message for n in app._notifications)
        assert app.area.fresh[-1][3] == "voxtex.unsure"


def test_pill_never_changes_size():
    from voxtex.editor import pill

    for state in pill.BORDER:
        for levels in ([], [0.0], [1.0] * 20, [0.2, 0.9, 0.4]):
            for t in (0.0, 0.37, 5.1):
                assert len(pill.pill_content(state, levels, t).plain) == pill.WIDTH


def test_pill_wave_grows_from_the_middle():
    from voxtex.editor import pill

    wave = pill.pill_content(pill.HEARING, [0.0] * 8 + [1.0], 0).plain.split(" ")
    assert wave[len(wave) // 2] == "█"
    assert wave[0] == wave[-1] == pill.LEVELS[1]
