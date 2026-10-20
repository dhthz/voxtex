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


async def test_cite_here_keeps_the_selected_text(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        llm.replies.append({"action": "insert", "latex": "Legal models do well."})
        await talk(app, pilot, "legal models do well")
        row = next(i for i, ln in enumerate(area.document.lines) if "do well." in ln)
        area.selection = ((row, 0), (row, len(area.document.lines[row])))
        # The model forgets the text and returns only the citation.
        llm.replies.append(
            {"action": "replace", "block_id": "sel", "latex": "\\parencite{chalkidis2019}"}
        )
        await talk(app, pilot, "cite Chalkidis here")
        assert "Legal models do well \\parencite{chalkidis2019}." in path.read_text()
        # "undo" works with a selection too, without asking the model.
        area.selection = ((row, 0), (row, 5))
        await talk(app, pilot, "undo")
        assert "Legal models do well." in path.read_text()
        assert len(llm.seen) == 2


async def test_noise_is_not_written_and_says_what_was_heard(editor):
    app, path, llm = editor
    async with app.run_test(notifications=True) as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        before = path.read_text()
        await talk(app, pilot, "Andes-Lee, 커버 at core.")  # Whisper's noise: no model call
        llm.replies.append({"action": "unclear", "note": "random letters"})
        await talk(app, pilot, "M-A-U-A-O-E")  # the model can't read it as words
        assert path.read_text() == before
        assert len(llm.seen) == 1
        assert sum("Didn't catch that" in n.message for n in app._notifications) == 2
        # A long recording is real speech, even if the model calls it unclear.
        llm.replies.append({"action": "unclear"})
        await talk(app, pilot, "the system stores triples in a relational database we built")
        assert "relational database" in path.read_text()


async def test_saying_undo_undoes(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        llm.replies.append({"action": "insert", "latex": "It reaches 0.71."})
        await talk(app, pilot, "it reaches 0.71")
        assert "0.71" in path.read_text()
        await talk(app, pilot, "Undo the last change.")
        assert "0.71" not in path.read_text()
        assert len(llm.seen) == 1


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


async def test_esc_closes_the_file_menu_and_enter_opens_a_file(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        await pilot.press("ctrl+o")
        await pilot.pause(0.1)
        assert len(app.screen_stack) == 2
        await pilot.press("escape")  # never trapped: esc always goes back
        await pilot.pause(0.1)
        assert len(app.screen_stack) == 1
        await pilot.press("ctrl+o")
        await pilot.press(*"intro")  # type to filter
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert app.rel == "chapters/introduction.tex"


def test_file_tree_lists_every_tex_file_by_folder(tmp_path):
    from voxtex.editor.menus import tex_files

    for f in ("main.tex", "ch/a.tex", "ch/b.tex", ".voxtex/x.tex", "build/y.tex", "ch/c.bib"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    assert tex_files(tmp_path) == ["main.tex", "ch/a.tex", "ch/b.tex"]


async def test_unfiltered_skips_the_ai(editor):
    app, path, llm = editor
    async with app.run_test() as pilot:
        area = app.area
        area.move_cursor((len(area.document.lines) - 1, 0))
        await pilot.press("ctrl+t")
        await talk(app, pilot, "it reaches 50% , cite chalkidis")
        assert r"it reaches 50\% , cite chalkidis" in path.read_text()
        assert llm.seen == []
    assert app.unfiltered  # remembered for this thesis
    from voxtex.editor.app import _load_state

    assert _load_state(app.root)["unfiltered"] is True


def test_usage_panel_shows_only_what_is_in_use():
    from types import SimpleNamespace

    from rich.console import Console

    from voxtex.editor.usage import GpuSample, Usage, render, spark
    from voxtex.providers import Chain

    assert spark([1, 2, 4, 8], 6) == "  ▁▂▄█"
    console = Console(width=34, record=True)
    u = Usage()
    console.print(render(u, None, 34))
    assert "nothing to show yet" in console.export_text()
    chain = Chain([SimpleNamespace(name="groq")])
    chain.active, chain.totals.calls = "groq", 3
    u.latency.append(0.8)
    u.gpu = GpuSample("RTX 3060", 40, 4096, 12288)  # a GPU, but nothing uses it
    console.print(render(u, chain, 34))
    text = console.export_text()
    assert "groq" in text and "latency" in text and "gpu" not in text
    u.gpu_used = True
    console.print(render(u, chain, 34))
    assert "RTX 3060" in console.export_text()


async def test_theme_picker_previews_and_keeps_the_choice(editor):
    from voxtex.editor.app import _load_settings
    from voxtex.editor.palette import THEMES

    app, path, llm = editor
    assert app.theme_name == "charcoal" and len(THEMES) > 10
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.press("down")
        await pilot.pause(0.1)
        previewed = app.theme_name
        assert previewed != "charcoal"  # moving previews the theme
        await pilot.press("escape")  # esc goes back to the one you had
        await pilot.pause(0.1)
        assert app.theme_name == "charcoal" and len(app.screen_stack) == 1
        await pilot.press("ctrl+p", "down", "enter")
        await pilot.pause(0.1)
        assert app.theme_name == previewed
    assert _load_settings()["theme"] == previewed  # kept for every thesis


def test_every_theme_has_a_full_palette():
    from textual.color import Color

    from voxtex.editor.palette import THEMES

    for name, (theme, pal) in THEMES.items():
        for value in pal.css().values():
            Color.parse(value)  # raises on a bad colour
        assert theme.variables["vx-accent"] == pal.accent, name


async def test_typing_saves_itself_after_a_pause(editor, monkeypatch):
    from voxtex.editor import app as editor_app

    monkeypatch.setattr(editor_app, "AUTOSAVE_S", 0.5)
    app, path, llm = editor
    async with app.run_test(notifications=True) as pilot:
        await pilot.press(*"typed by hand")
        assert "typed by hand" not in path.read_text()  # not on every keystroke
        await pilot.pause(0.9)
        assert "typed by hand" in path.read_text()
        await pilot.press("x", "ctrl+s")
        await pilot.pause(0.05)
        assert "typed by handx" in path.read_text()
        assert any(n.message.startswith("Saved") for n in app._notifications)


@pytest.mark.parametrize(
    "latex, words",
    [
        (r"Το \en{Sesame} αποθηκεύει δεδομένα \en{RDF} \cite{Broekstra02}.", 5),
        (r"\section{Related Work}\label{sec:rw} See Figure~\ref{fig:a} and $x^2$ here.", 6),
        (r"\textbf{bold words} and \emph{this} \\ next % not this", 5),
        ("\\begin{equation}\n E = mc^2 \\label{eq:1}\n\\end{equation}\nDone.", 1),
        (
            r"\begin{figure}[!ht]\centering\includegraphics[width=5cm]{a.png}"
            r"\caption{Αρχιτεκτονική Απλού Κόμβου}\label{f}\end{figure}",
            3,
        ),
        (r"50\% of state-of-the-art models don't", 5),
        (
            "\\documentclass{book}\\usepackage{x}\\title{T}\n"
            "\\begin{document}Only this.\\end{document}",
            2,
        ),
    ],
)
def test_word_count_is_what_the_pdf_shows(latex, words):
    from voxtex.editor.words import count

    assert count(latex).words == words


async def test_side_panes_open_on_their_own(editor):
    app, path, llm = editor
    async with app.run_test(size=(150, 40)) as pilot:
        assert app.show_usage and app.show_words  # wide enough: both open
        await pilot.press("ctrl+g")  # usage closes; words takes the column
        await pilot.pause(0.2)
        assert not app.query_one("#usage").display and app.query_one("#words").display
        app.area.select_line(0)
        await pilot.pause(0.2)
        from rich.console import Console

        console = Console(width=40, record=True)
        console.print(app.query_one("#words").content)
        assert "selection" in console.export_text()
        await pilot.press("ctrl+n")
        await pilot.pause(0.2)
        assert not app.query_one("#side").display  # no pane open: no column
    from voxtex.editor.app import _load_state

    assert _load_state(app.root)["usage"] is False and _load_state(app.root)["words"] is False


async def test_a_table_is_edited_in_its_pane(editor):
    from test_tables import BOOKTABS

    app, path, llm = editor
    path.write_text(BOOKTABS, encoding="utf-8")  # the editor opens this file
    async with app.run_test(size=(150, 40)) as pilot:
        row = next(i for i, ln in enumerate(app.area.document.lines) if "legal-bert" in ln)
        app.area.move_cursor((row, 6))
        await pilot.pause(0.2)
        assert app.query_one("#table").display  # the pane opens by itself
        await pilot.press("ctrl+l", "down", "right", "enter")  # row 2, column 2
        await pilot.pause(0.1)
        app.query_one("#cell").value = "0.73"
        await pilot.press("enter", "r")  # change the cell, then add a row after it
        await pilot.pause(0.3)
        await pilot.press("escape")  # back to the text
        await pilot.pause(0.1)
        assert app.area.has_focus
        text = app.area.text
        assert r"\textbf{legal-bert} & 0.73" in text and "0.71" not in text
        assert len(table_rows(text)) == 4  # header, two models, the new row
        assert r"\caption{Results \& more}" in text  # outside the tabular: untouched
        app.area.move_cursor((0, 0))
        await pilot.pause(0.2)
        assert not app.query_one("#table").display  # out of the table: the pane goes


def table_rows(text):
    from voxtex.editor.tables import find_table, parse

    start, end = find_table(text, text.index("legal-bert"))
    return parse(text[start:end]).rows
