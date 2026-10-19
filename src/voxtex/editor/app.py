"""`voxtex` / `voxtex edit`: the chapter open in a terminal editor, with voice.

It works like Wispr Flow: press ctrl+space, talk, press it again. What you said is
written at the cursor, formatted as LaTeX but in your words. With text selected, what
you say is an instruction for it ("make this more formal", "add the Broekstra citation").
The microphone records only between the two presses, so typing and clicks never count.
Voice edits are part of the editor's undo (ctrl+z), and the file is saved after every
voice edit, on ctrl+s and on quit (with a backup before the first save).
"""

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.text_area import TextAreaTheme

from voxtex.app import build_chain, open_project
from voxtex.audio.turns import is_garbled, spoken_undo
from voxtex.config import Config
from voxtex.editor import pill
from voxtex.editor.highlight import char_to_byte, latex_spans
from voxtex.editor.place import Location, advance, place_block
from voxtex.engine.buffer import Buffer
from voxtex.engine.decide import Ignore, Insert, Replace, UncertainInsert
from voxtex.engine.engine import Engine, Step
from voxtex.io.writer import Writer, read_file, write_atomic
from voxtex.project.index import STATE_DIR, ProjectIndex
from voxtex.providers import Chain
from voxtex.ui import ACCENT, session_summary

MIN_RECORDING_S = 0.4
MAX_RECORDING_S = 300.0
SILENT_PEAK = 0.01  # a recording never louder than this heard nothing
FRESH_SECONDS = 4.0  # how long newly written text stays tinted
CHECK_SECONDS = 20.0  # text voxtex is unsure about stays tinted longer
BOM = "﻿"
TEXT_COLOR = "#d4d4d4"
FAINT = "#5a5a5a"

THEME = TextAreaTheme(
    name="voxtex",
    base_style=Style(color=TEXT_COLOR),
    gutter_style=Style(color="#3c3c3c"),
    cursor_line_gutter_style=Style(color="#8a8a8a"),
    cursor_line_style=Style(bgcolor="#1a1a1a"),
    cursor_style=Style(color="#111111", bgcolor="#e6e6e6"),
    selection_style=Style(bgcolor="#4a3328"),
    bracket_matching_style=Style(bold=True),
    syntax_styles={
        "latex.comment": Style(color="#6a737d", italic=True),
        "latex.command": Style(color="#569cd6"),
        "latex.structure": Style(color="#c586c0", bold=True),
        "latex.ref": Style(color="#4ec9b0"),
        "latex.math": Style(color="#dcdcaa"),
        "latex.brace": Style(color="#808080"),
        "voxtex.check": Style(color="#e5c07b", bgcolor="#3a3220"),
        "voxtex.new": Style(bgcolor="#1d3b26"),
        "voxtex.unsure": Style(bgcolor="#3a3220"),
    },
)


class LatexArea(TextArea):
    """A TextArea with LaTeX colouring and a tint on text voxtex just wrote."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fresh: list[tuple[Location, Location, float, str]] = []

    def mark_fresh(self, start: Location, end: Location, unsure: bool = False) -> None:
        seconds = CHECK_SECONDS if unsure else FRESH_SECONDS
        name = "voxtex.unsure" if unsure else "voxtex.new"
        self.fresh.append((start, end, time.monotonic() + seconds, name))
        self._build_highlight_map()
        self.refresh()

    def expire_fresh(self) -> None:
        now = time.monotonic()
        if any(f[2] <= now for f in self.fresh):
            self.fresh = [f for f in self.fresh if f[2] > now]
            self._build_highlight_map()
            self.refresh()

    def _build_highlight_map(self) -> None:  # TextArea hook (textual 8)
        self._line_cache.clear()
        highlights = self._highlights
        highlights.clear()
        lines = self.document.lines
        for row, line in enumerate(lines):
            spans = latex_spans(line)
            if spans:
                b = char_to_byte(line)
                highlights[row].extend((b[s], b[e], name) for s, e, name in spans)
        for (sr, sc), (er, ec), _, name in getattr(self, "fresh", []):
            for row in range(sr, min(er, len(lines) - 1) + 1):
                line = lines[row]
                b = char_to_byte(line)
                s = min(sc if row == sr else 0, len(line))
                e = min(ec if row == er else len(line), len(line))
                if e > s:
                    highlights[row].append((b[s], b[e], name))


class FilePicker(ModalScreen[str | None]):
    DEFAULT_CSS = """
    FilePicker { align: center middle; background: black 40%; }
    FilePicker > OptionList {
        width: 64; max-height: 60%; padding: 1 2; border: round #3a3a3a;
        background: #141414;
    }
    FilePicker > OptionList > .option-list--option-highlighted { background: #2a2a2a; }
    """
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, files: list[str]):
        super().__init__()
        self.files = files

    def compose(self) -> ComposeResult:
        yield OptionList(*self.files)

    @on(OptionList.OptionSelected)
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.files[event.option_index])


class EditorApp(App):
    TITLE = "voxtex"
    ENABLE_COMMAND_PALETTE = False
    # Every row has a fixed height: what changes is the content, never the layout.
    CSS = """
    Screen { layout: vertical; background: ansi_default; }
    #top { height: 1; margin: 0 2; }
    #page { height: 1fr; margin-top: 1; }
    LatexArea {
        width: 100%; max-width: 104; height: 1fr; border: none; padding: 0;
        background: ansi_default;
        scrollbar-size-vertical: 1; scrollbar-color: #2e2e2e; scrollbar-color-hover: #4a4a4a;
        scrollbar-background: ansi_default; scrollbar-background-hover: ansi_default;
    }
    LatexArea:focus { border: none; }
    #heard { height: 1; margin: 1 4 0 4; text-align: center; }
    #dock { height: 3; }
    #pill { width: auto; height: 3; padding: 0 2; border: round #3a3a3a; }
    ToastRack { dock: top; align: right top; margin-top: 2; }
    Toast { width: 52; max-width: 60%; background: #1a1a1a; padding: 0 1; border-left: none; }
    Toast.-information { border-left: outer #4a4a4a; }
    Toast.-warning { border-left: outer #c9a050; }
    Toast.-error { border-left: outer #c25a5a; }
    """
    BINDINGS = [
        Binding("ctrl+space,ctrl+@,ctrl+r", "talk", "Talk", priority=True),
        Binding("escape", "cancel", "Cancel recording", priority=True),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+o", "open_file", "Open chapter"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        idx: ProjectIndex,
        config: Config,
        chain: Chain,
        path: Path,
        private: bool = False,
        stt_mode: str | None = None,
        mic: str | None = None,
        audio: bool = True,
    ):
        super().__init__(ansi_color=True)
        self.idx, self.config, self.chain = idx, config, chain
        self.root = idx.root_path
        self.path = path
        self.private, self.stt_mode, self.mic_name = private, stt_mode, mic
        self.audio = audio
        self.engine = Engine(idx, chain, config, None)
        self.engine.blocks_in_location = True  # earlier blocks are in the editor text
        self.writer = Writer(self.root)
        self.bom = False
        self.saved_text = ""
        self.saved_mtime = 0.0
        self.writes = 0
        self.levels: list[float] = []
        # loading → ready → recording → transcribing → writing → ready  (or no-mic)
        self.phase = "loading" if audio else "ready"
        self.recording_since = 0.0
        self.selection_range: tuple[Location, Location] | None = None
        self.speech_label = ""
        self.microphone = self.transcriber = self.fallback = None
        self.prompt = ""
        self.said = ""

    # --- layout ----------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Center(id="page"):
            yield LatexArea(
                id="editor", soft_wrap=True, show_line_numbers=True, tab_behavior="indent"
            )
        yield Static(id="heard")
        with Center(id="dock"):
            yield Static(id="pill")

    @property
    def area(self) -> LatexArea:
        return self.query_one(LatexArea)

    def on_mount(self) -> None:
        self.area.register_theme(THEME)
        self.area.theme = "voxtex"
        self.load(self.path)
        self.set_interval(0.1, self.refresh_chrome)
        self.set_interval(1 / 20, self.animate)
        self.set_interval(1.0, self.tick)
        if self.audio:
            self.run_worker(self.start_audio, thread=True)
        self.refresh_chrome()

    # --- files -----------------------------------------------------------------------------

    @property
    def rel(self) -> str:
        return self.writer.rel(self.path)

    def load(self, path: Path) -> None:
        self.path = path
        text = read_file(path)
        self.bom = text.startswith(BOM)
        text = text.removeprefix(BOM)
        self.area.load_text(text)
        self.saved_text = text
        self.saved_mtime = path.stat().st_mtime
        self.engine.buffer = Buffer()
        row = _remembered_row(self.root, self.rel)
        lines = self.area.document.lines
        if row is None or row >= len(lines):
            row = len(lines) - 1
        self.area.move_cursor((row, len(lines[row])), center=True)
        self.area.focus()

    def save(self) -> None:
        text = self.area.text
        if text == self.saved_text and self.path.exists():
            return
        self.writer.backup(self.path)
        write_atomic(self.path, (BOM if self.bom else "") + text)
        self.saved_text = text
        self.saved_mtime = self.path.stat().st_mtime
        _remember_row(self.root, self.rel, self.area.cursor_location[0])

    def action_save(self) -> None:
        self.save()

    def action_open_file(self) -> None:
        def chosen(rel: str | None) -> None:
            if rel:
                self.save()
                self.load(self.root / rel)

        self.push_screen(FilePicker(self.idx.body_files), chosen)

    async def action_quit(self) -> None:
        self.save()
        self.stop_audio()
        self.exit()

    def tick(self) -> None:
        self.area.expire_fresh()
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self.saved_mtime:
            return
        self.saved_mtime = mtime
        on_disk = read_file(self.path).removeprefix(BOM)
        if on_disk == self.saved_text:
            return
        if self.area.text == self.saved_text:  # no unsaved changes here: take theirs
            cursor = self.area.cursor_location
            self.area.load_text(on_disk)
            self.saved_text = on_disk
            self.area.move_cursor(cursor)
            self.note(f"{self.rel} changed on disk and was reloaded")
        else:
            self.note(f"{self.rel} changed on disk too; ctrl+s keeps this version", "warning")

    # --- audio -----------------------------------------------------------------------------

    def start_audio(self) -> None:
        """Worker thread: load speech-to-text and open the microphone."""
        from voxtex.audio.capture import MicError, Segmenter, open_microphone
        from voxtex.audio.stt import SttChoice, build_transcriber, choose_stt
        from voxtex.listenmode import stt_languages, whisper_prompt

        cfg = self.config
        choice = choose_stt(cfg, self.private, self.stt_mode)
        self.speech_label = choice.label
        transcriber = build_transcriber(choice, cfg, stt_languages(self.idx))
        if choice.kind != "local":
            self.fallback = build_transcriber(SttChoice("local", "small", "cpu"), cfg)
        else:
            transcriber.load()
        self.transcriber, self.prompt = transcriber, whisper_prompt(self.idx)
        try:
            # Press-to-talk: nothing needs cutting into utterances.
            mic = open_microphone(Segmenter(lambda frame: False), self.mic_name or cfg.mic)
            mic.start()
        except MicError as e:
            self.call_from_thread(self.note, f"{e}\nvoxtex listen --list-mics", "error")
            self.phase = "no-mic"
            return
        self.microphone = mic
        self.phase = "ready"

    def stop_audio(self) -> None:
        if self.microphone is not None:
            self.microphone.stop()
            self.microphone = None
        for t in (self.transcriber, self.fallback):
            if hasattr(t, "unload"):
                t.unload()
        self.transcriber = self.fallback = None

    # --- talking ---------------------------------------------------------------------------

    def action_talk(self) -> None:
        if self.phase == "recording":
            self.stop_recording()
        elif self.phase == "ready" and self.microphone is not None:
            self.microphone.recording = []
            self.recording_since = time.monotonic()
            self.phase = "recording"
        elif self.phase in ("transcribing", "writing"):
            self.note("Still writing what you said last; one moment.")
        elif self.phase == "loading":
            self.note("Speech recognition is still loading; one moment.")
        elif self.phase == "no-mic":
            self.note("No microphone. voxtex listen --list-mics shows them.", "error")

    def action_cancel(self) -> None:
        if self.phase == "recording":
            self.microphone.recording = None
            self.phase = "ready"

    def stop_recording(self) -> None:
        frames, self.microphone.recording = self.microphone.recording or [], None
        self.phase = "ready"
        audio = recorded_audio(frames)
        if len(audio) < MIN_RECORDING_S * 16000:
            self.note("Too short. Press ctrl+space, talk, then press it again.")
            return
        if float(np.abs(audio).max()) < SILENT_PEAK:
            self.note("Didn't hear anything. Is the microphone muted?", "warning")
            return
        self.process(audio)

    def process(self, audio: np.ndarray) -> None:
        """Snapshot the selection on the UI thread; transcribe and write in a worker."""
        area = self.area
        selected = area.selected_text
        sel = area.selection
        start, end = min(sel.start, sel.end), max(sel.start, sel.end)
        self.selection_range = (start, end) if selected.strip() else None
        cursor = start if self.selection_range else area.cursor_location
        before = area.document.get_text_range((0, 0), cursor)
        self.phase = "transcribing"

        def work() -> None:
            try:
                said = self.transcribe(audio)
            except Exception as e:  # noqa: BLE001 - shown to the user, then carry on
                self.call_from_thread(self.finish, None, f"Speech recognition failed: {e}")
                return
            if not said:
                self.call_from_thread(self.finish, None, "")
                return
            if is_garbled(said):  # Whisper's noise, not anything said
                unheard = f'Didn\'t catch that ("{_tail(said, 50)}"). Say it again.'
                self.call_from_thread(self.finish, None, "", unheard)
                return
            undos = spoken_undo(said)
            if undos and not self.selection_range:
                self.call_from_thread(self.voice_undo, undos)
                return
            self.said, self.phase = said, "writing"
            try:
                if self.selection_range:
                    step = self.engine.command(said, selected, before)
                else:
                    step = self.engine.dictate(said, before)
            except Exception as e:  # noqa: BLE001 - never lose what was said
                if self.selection_range:
                    error = f"{e}\nThe selection was not changed."
                    self.call_from_thread(self.finish, None, error)
                    return
                step = self.engine.as_heard(said, str(e))
            self.call_from_thread(self.finish, step, None)

        self.run_worker(work, thread=True)

    def voice_undo(self, times: int) -> None:
        for _ in range(times):
            self.area.undo()
        self.phase, self.said = "ready", ""
        self.note("Undone" if times == 1 else f"Undone {times} times")
        self.save()

    def transcribe(self, audio: np.ndarray) -> str:
        """Worker thread. The whole recording at once, so Whisper hears full sentences."""
        from voxtex.audio.turns import is_filler
        from voxtex.listenmode import _log

        try:
            t = self.transcriber.transcribe(audio, self.prompt)
        except Exception as e:
            if self.fallback is None:
                raise
            self.call_from_thread(self.note, f"Switched to {self.fallback.name} ({e})")
            self.transcriber, self.fallback = self.fallback, None
            t = self.transcriber.transcribe(audio, self.prompt)
        _log(self.root, {
            "ts": round(time.time(), 3), "kind": "stt", "backend": t.backend,
            "language": t.language, "audio_s": round(t.audio_seconds, 2),
            "latency_ms": round(t.seconds * 1000), "text": t.text,
        })  # fmt: skip
        return "" if is_filler(t.text, self.prompt) else t.text

    def finish(self, step: Step | None, error: str | None, unheard: str = "") -> None:
        self.phase = "ready"
        self.said = ""
        if error:
            self.note(error, "error")
        elif step is None:
            self.note(unheard or "Didn't catch any words.", "warning" if unheard else "information")
        else:
            self.apply_step(step)

    # --- writing ---------------------------------------------------------------------------

    def apply_step(self, step: Step) -> None:
        """Put the result into the editor text (and save)."""
        d, b, area = step.decision, step.block, self.area
        for n in step.notices:
            self.note(n)
        if isinstance(d, Ignore) or b is None:
            return
        # Doubts go to a toast and a yellow tint, never into the LaTeX as comments.
        unsure = isinstance(d, UncertainInsert) or bool(b.note)
        if unsure:
            self.note(f"Check this: {b.note}" if b.note else "Not sure about this; check it.",
                      "warning", timeout=10)  # fmt: skip
        text = _text(b)
        if isinstance(d, Insert | UncertainInsert):
            self._insert(text, unsure=unsure)
        elif isinstance(d, Replace) and self.selection_range:
            start, end = self.selection_range
            old = step.previous.latex if step.previous else ""
            if area.get_text_range(start, end) == old:
                self._replace(start, end, text, unsure)
            else:
                self._insert(text, "The selection changed meanwhile", unsure)
        if step.report:
            for w in step.report.warnings:
                self.note(w.message, "warning")
        self.writes += 1
        self.save()

    def _insert(self, text: str, why: str = "", unsure: bool = False) -> None:
        area = self.area
        p = place_block(list(area.document.lines), area.cursor_location, text)
        area.insert(p.text, p.at)
        area.mark_fresh(p.start, p.end, unsure)
        area.move_cursor(p.end)
        area.scroll_cursor_visible()
        if why:
            self.note(f"{why}; wrote the new version at the cursor.", "warning")

    def _replace(self, start: Location, end: Location, text: str, unsure: bool = False) -> None:
        area = self.area
        area.replace(text, start, end)
        area.mark_fresh(start, advance(start, text), unsure)
        area.move_cursor(advance(start, text))
        area.scroll_cursor_visible()

    # --- chrome ----------------------------------------------------------------------------

    def note(self, message: str, severity: str = "information", timeout: float = 0) -> None:
        """A small toast that fades by itself; the editor text is the real feedback."""
        timeout = timeout or (6 if severity == "error" else 4)
        self.notify(message, severity=severity, timeout=timeout)

    def pill_state(self) -> str:
        return {
            "loading": pill.LOADING,
            "recording": pill.HEARING,
            "transcribing": pill.WORKING,
            "writing": pill.WORKING,
            "no-mic": pill.NO_MIC,
        }.get(self.phase, pill.IDLE)

    def animate(self) -> None:
        mic = self.microphone
        self.levels = (self.levels + [mic.level if mic is not None else 0.0])[-pill.BARS :]
        if self.phase == "recording" and time.monotonic() - self.recording_since > MAX_RECORDING_S:
            self.stop_recording()
        state = self.pill_state()
        try:
            widget = self.query_one("#pill", Static)
        except NoMatches:  # closing down
            return
        widget.styles.border = ("round", pill.BORDER[state])
        widget.update(pill.pill_content(state, self.levels, time.monotonic()))

    def refresh_chrome(self) -> None:
        width = self.size.width - 4
        folder, _, name = self.rel.rpartition("/")
        dirty = self.area.text != self.saved_text
        top = Table.grid(expand=True)
        top.add_column(no_wrap=True)
        top.add_column(justify="right", no_wrap=True)
        keys = Text("^space talk  ^o chapters  ^z undo  ^q quit", style=FAINT)
        top.add_row(
            Text.assemble(
                ("✻ ", ACCENT), (folder + "/" if folder else "", FAINT), (name, TEXT_COLOR),
                (" •" if dirty else "  ", ACCENT),
            ),
            keys if width > 70 else "",
        )  # fmt: skip
        self.query_one("#top", Static).update(top)

        if self.phase == "recording":
            secs = int(time.monotonic() - self.recording_since)
            line = Text.assemble(
                ("● ", "#e06c6c"), (f"{secs // 60}:{secs % 60:02d}", TEXT_COLOR),
                ("   ctrl+space to finish · esc to cancel", FAINT),
            )  # fmt: skip
        elif self.said:
            line = Text(_tail(self.said, width - 8), style="#9a9a9a italic")
        elif self.phase == "ready" and self.area.selected_text.strip():
            line = Text("ctrl+space and say what to change in the selection", style=FAINT)
        elif self.phase == "ready":
            line = Text("ctrl+space to talk", style="#3f3f3f")
        else:
            line = Text("")
        self.query_one("#heard", Static).update(line)


def recorded_audio(frames: list[np.ndarray]) -> np.ndarray:
    """Raw 16-bit microphone frames as float32 audio for Whisper."""
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).astype(np.float32) / 32768.0


def _text(block) -> str:
    """What goes into the chapter for a block: its LaTeX only."""
    return block.latex.rstrip()


def _tail(text: str, width: int) -> str:
    """The end of what was said, on one line."""
    text = " ".join(text.split())
    return text if len(text) <= width else "…" + text[-(width - 1) :]


# --- remembering where you were ----------------------------------------------------------------


def _state_path(root: Path) -> Path:
    return root / STATE_DIR / "editor.json"


def _load_state(root: Path) -> dict:
    try:
        return json.loads(_state_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remembered_row(root: Path, rel: str) -> int | None:
    return _load_state(root).get("rows", {}).get(rel)


def _remember_row(root: Path, rel: str, row: int) -> None:
    state = _load_state(root)
    state.setdefault("rows", {})[rel] = row
    state["last_file"] = rel
    try:
        _state_path(root).parent.mkdir(parents=True, exist_ok=True)
        _state_path(root).write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError:
        pass


def pick_file(idx: ProjectIndex, wanted: Path | None) -> Path | None:
    """The file to open: the one asked for, else the last one edited, else the chapter
    changed most recently."""
    root = idx.root_path
    if wanted is not None:
        p = wanted if wanted.is_absolute() else Path.cwd() / wanted
        if not p.is_file():
            p = root / wanted
        return p if p.is_file() else None
    last = _load_state(root).get("last_file")
    if last and (root / last).is_file():
        return root / last
    chapters = [f for f in idx.body_files if f != idx.main] or idx.body_files
    files = [root / f for f in chapters if (root / f).is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def run_editor(
    console: Console,
    path: Path,
    file: Path | None,
    private: bool,
    provider: str | None,
    stt_mode: str | None,
    mic: str | None,
    verbose: bool = False,
) -> int:
    if file is None and path.is_file():  # `voxtex chapters/x.tex`
        path, file = path.parent, path
    opened = open_project(console, path)
    if opened is None:
        return 1
    idx, config = opened
    chain = build_chain(console, idx, config, private, provider)
    if chain is None:
        return 1
    target = pick_file(idx, file)
    if target is None:
        console.print(f"[red]✗[/red] no such file: {file}")
        return 1
    app = EditorApp(idx, config, chain, target, private, stt_mode, mic)
    app.run()
    app.stop_audio()
    summary = session_summary(chain, app.writes, verbose=verbose, undo_hint=False)
    console.print(f"[grey50]{summary}[/grey50]")
    return 0
