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
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Static, TextArea
from textual.widgets.text_area import TextAreaTheme

from voxtex import config as voxtex_config
from voxtex.app import build_chain, open_project
from voxtex.audio.turns import is_garbled, spoken_undo
from voxtex.config import Config
from voxtex.editor import palette, pill, usage, words
from voxtex.editor.highlight import char_to_byte, latex_spans
from voxtex.editor.menus import FilePicker, ThemePicker, hints
from voxtex.editor.place import Location, advance, place_block
from voxtex.editor.tablepane import TablePane
from voxtex.editor.tables import ends_with_break, find_table, parse, render
from voxtex.engine.buffer import Buffer
from voxtex.engine.decide import Ignore, Insert, Replace, UncertainInsert
from voxtex.engine.engine import Engine, Step
from voxtex.io.writer import Writer, read_file, write_atomic
from voxtex.project.index import STATE_DIR, ProjectIndex
from voxtex.providers import Chain
from voxtex.ui import session_summary

MIN_RECORDING_S = 0.4
MAX_RECORDING_S = 300.0
SILENT_PEAK = 0.01  # a recording never louder than this heard nothing
FRESH_SECONDS = 4.0  # how long newly written text stays tinted
CHECK_SECONDS = 20.0  # text voxtex is unsure about stays tinted longer
BOM = "﻿"
AUTOSAVE_S = 1.5  # typed edits are saved this long after the last keystroke
PANES_AUTO_WIDTH = 130  # the side panes open by themselves on terminals this wide


def text_area_theme(pal: palette.Palette) -> TextAreaTheme:
    """LaTeX colouring for the text area, from the palette."""
    return TextAreaTheme(
        name="voxtex",
        base_style=Style(color=pal.text),
        gutter_style=Style(color=pal.surface1),
        cursor_line_gutter_style=Style(color=pal.overlay1),
        cursor_line_style=Style(bgcolor=pal.surface0),
        cursor_style=Style(color="black", bgcolor=pal.text),
        selection_style=Style(bgcolor=pal.surface1),
        bracket_matching_style=Style(bold=True),
        syntax_styles={
            "latex.comment": Style(color=pal.overlay0, italic=True),
            "latex.command": Style(color=pal.command),
            "latex.structure": Style(color=pal.mauve, bold=True),
            "latex.ref": Style(color=pal.teal),
            "latex.math": Style(color=pal.yellow),
            "latex.brace": Style(color=pal.overlay1),
            "voxtex.check": Style(color=pal.yellow, bgcolor="#3a3220"),
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


class EditorApp(App):
    TITLE = "voxtex"
    ENABLE_COMMAND_PALETTE = False
    # Every row has a fixed height: what changes is the content, never the layout.
    CSS = """
    Screen { layout: vertical; background: ansi_default; }
    #top {
        height: 2; padding: 0 1; background: ansi_default;
        border-bottom: solid $vx-surface0;
    }
    #body { width: 100%; height: 1fr; }
    #page { width: 1fr; height: 1fr; margin-top: 1; }
    #side { width: 38; height: 1fr; }
    #table { height: auto; max-height: 70%; }
    .pane {
        height: 1fr; padding: 0 1; background: ansi_default;
        border: round $vx-surface1; border-title-color: $vx-text;
        border-title-style: bold; border-subtitle-color: $vx-overlay0;
    }
    LatexArea {
        width: 100%; max-width: 104; height: 1fr; border: none; padding: 0;
        background: ansi_default;
        scrollbar-size-vertical: 1; scrollbar-color: $vx-surface0;
        scrollbar-color-hover: $vx-surface1;
        scrollbar-background: ansi_default; scrollbar-background-hover: ansi_default;
    }
    LatexArea:focus { border: none; }
    #heard { height: 1; margin: 1 4 0 4; text-align: center; }
    #dock { height: 3; }
    #pill { width: auto; height: 3; padding: 0 2; border: round $vx-surface1; }
    ToastRack { dock: top; align: right top; margin-top: 3; }
    Toast {
        width: 52; max-width: 60%; background: ansi_default; color: $vx-text; padding: 0 1;
        border: round $vx-surface1;
    }
    Toast.-information { border-left: outer $vx-accent; }
    Toast.-warning { border-left: outer $vx-yellow; }
    Toast.-error { border-left: outer $vx-red; }
    """
    BINDINGS = [
        Binding("ctrl+space,ctrl+@,ctrl+r", "talk", "Talk", priority=True),
        Binding("escape", "cancel", "Cancel recording", priority=True),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+o", "open_file", "Open a file"),
        Binding("ctrl+t", "toggle_unfiltered", "Unfiltered / AI formatting"),
        Binding("ctrl+p", "pick_theme", "Theme"),
        Binding("ctrl+g", "toggle_usage", "Usage pane"),
        Binding("ctrl+n", "toggle_words", "Words pane"),
        Binding("ctrl+l", "edit_table", "Edit the table"),
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
        for theme, _ in palette.THEMES.values():
            self.register_theme(theme)
        self.theme_name = _load_settings().get("theme", palette.DEFAULT)
        if self.theme_name not in palette.THEMES:
            self.theme_name = palette.DEFAULT
        self.pal = palette.palette(self.theme_name)
        self.theme = self.theme_name
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
        self.autosave = None  # the pending save after typing
        self.download: tuple[float, float] | None = None  # speech model: MB done, MB in all
        state = _load_state(self.root)
        self.unfiltered = bool(state.get("unfiltered", False))  # words as heard, no AI
        # Side panes, toggled one by one like btop's; None: open if there's room.
        self.show_usage: bool | None = state.get("usage")
        self.show_words: bool | None = state.get("words")
        self.usage = usage.Usage()
        self.counts_stale = self.table_stale = True
        self.words_at_open = 0
        self.table_span: tuple[int, int] | None = None  # the table around the cursor

    # --- layout ----------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Horizontal(id="body"):
            with Center(id="page"):
                yield LatexArea(
                    id="editor", soft_wrap=True, show_line_numbers=True, tab_behavior="indent"
                )
            with Vertical(id="side"):
                yield TablePane(id="table", classes="pane")
                yield Static(id="usage", classes="pane")
                yield Static(id="words", classes="pane")
        yield Static(id="heard")
        with Center(id="dock"):
            yield Static(id="pill")

    @property
    def area(self) -> LatexArea:
        return self.query_one(LatexArea)

    def on_mount(self) -> None:
        self.area.register_theme(text_area_theme(self.pal))
        self.area.theme = "voxtex"
        self.load(self.path)
        wide = self.size.width >= PANES_AUTO_WIDTH
        self.show_usage = wide if self.show_usage is None else self.show_usage
        self.show_words = wide if self.show_words is None else self.show_words
        table = self.query_one("#table", TablePane)
        table.border_title, table.border_subtitle = "table", "ctrl+l"
        for pane, (title, key) in {
            "usage": ("usage", "ctrl+g"),
            "words": ("words", "ctrl+n"),
        }.items():
            widget = self.query_one(f"#{pane}", Static)
            widget.border_title, widget.border_subtitle = title, key
        if any(_on_this_machine(p) for p in self.chain.usable()):
            self.usage.gpu_used = True  # the local model (Ollama) may run on the GPU
            self.usage.start_gpu()  # shows nothing without an NVIDIA GPU
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
        self.words_at_open = words.count(text).words
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
        self.note(f"Saved {self.rel}")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Typed edits save themselves after a short pause, like voice edits do. Some
        terminals (VS Code's, for one) keep ctrl+s for themselves."""
        if self.autosave is not None:
            self.autosave.stop()
        self.autosave = self.set_timer(AUTOSAVE_S, self.save)
        self.counts_stale = self.table_stale = True

    def on_text_area_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        self.counts_stale = self.table_stale = True

    def action_open_file(self) -> None:
        def chosen(rel: str | None) -> None:
            if rel:
                self.save()
                self.load(self.root / rel)

        self.push_screen(FilePicker(self.root, self.idx.body_files, self.rel), chosen)

    def action_toggle_unfiltered(self) -> None:
        self.unfiltered = not self.unfiltered
        _remember(self.root, unfiltered=self.unfiltered)
        if self.unfiltered:
            self.note("Unfiltered: what you say goes in exactly as heard, no AI.")
        else:
            self.note("AI formatting: your words, with punctuation, citations and LaTeX.")

    def action_pick_theme(self) -> None:
        def chosen(name: str | None) -> None:
            if name:
                self.apply_theme(name)
                _save_settings(_load_settings() | {"theme": name})

        self.push_screen(ThemePicker(self.theme_name), chosen)

    def apply_theme(self, name: str) -> None:
        """Switch every colour on screen: CSS through the theme, Rich text through the
        palette, and the LaTeX colouring."""
        self.theme_name, self.pal = name, palette.palette(name)
        self.theme = name
        self.area.register_theme(text_area_theme(self.pal))
        self.area.theme = "voxtex"
        self.area._build_highlight_map()
        self.area.refresh()

    def action_toggle_usage(self) -> None:
        self.show_usage = not self.show_usage
        _remember(self.root, usage=self.show_usage)

    def action_toggle_words(self) -> None:
        self.show_words = not self.show_words
        _remember(self.root, words=self.show_words)
        self.counts_stale = True

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        # Esc only cancels a recording; otherwise it reaches menus and the editor.
        if action == "cancel":
            return self.phase == "recording"
        return True

    async def action_quit(self) -> None:
        self.save()
        self.stop_audio()
        self.usage.stop()
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
            transcriber.progress = self.downloading
            transcriber.load()
            self.download = None
            if choice.device == "cuda":
                self.usage.gpu_used = True
                self.usage.start_gpu()
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

    def downloading(self, done_mb: float, total_mb: float) -> None:
        """Worker thread: the speech model is downloading (first run only)."""
        self.download = (done_mb, total_mb)

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
            if undos:
                self.call_from_thread(self.voice_undo, undos)
                return
            self.said, self.phase = said, "writing"
            if self.unfiltered:
                step = self.engine.verbatim(said, selected if self.selection_range else "")
                self.call_from_thread(self.finish, step, None)
                return
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
        self.usage.speech.append(t.seconds)
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
        if step.provider:
            self.usage.tokens_in.append(step.input_tokens)
            self.usage.latency.append(step.latency_s)
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
        widget.styles.border = ("round", pill.border(state, self.pal))
        widget.update(pill.pill_content(state, self.levels, time.monotonic(), self.pal))

    # --- tables ----------------------------------------------------------------------------

    def find_table(self) -> None:
        """Show the table pane while the cursor is in a table (or the pane is in use)."""
        pane = self.query_one("#table", TablePane)
        if pane.has_focus_within:
            return  # editing there: the cursor in the text doesn't move meanwhile
        doc, text = self.area.document, self.area.text
        found = find_table(text, doc.get_index_from_location(self.area.cursor_location))
        table = parse(text[found[0] : found[1]]) if found else None
        self.table_span = found if table else None
        pane.display = table is not None
        if table is not None:
            pane.show(table, text[found[0] : found[1]])

    def action_edit_table(self) -> None:
        pane = self.query_one("#table", TablePane)
        if self.table_span is None:
            self.note("Put the cursor in a table first.")
            return
        pane.grid.focus()

    def on_table_pane_edited(self, event: TablePane.Edited) -> None:
        """Write the edited table back into the chapter (one undoable change)."""
        pane = self.query_one("#table", TablePane)
        if self.table_span is None:
            return
        start, end = self.table_span
        area, doc = self.area, self.area.document
        if area.text[start:end] != pane.source:
            self.note("The table changed in the text meanwhile; edit it again.", "warning")
            pane.source = ""
            return
        new = render(event.table, ends_with_break(pane.source))
        area.replace(new, doc.get_location_from_index(start), doc.get_location_from_index(end))
        area.move_cursor(doc.get_location_from_index(start))
        self.table_span = (start, start + len(new))
        pane.show(event.table, new)

    def refresh_chrome(self) -> None:
        try:
            top, heard, used, counted = (
                self.query_one(f"#{n}", Static) for n in ("top", "heard", "usage", "words")
            )
            side = self.query_one("#side")
        except NoMatches:  # closing down
            return
        top.update(self.top_bar())
        # The open panes share the side column; with none open it disappears. A table
        # under the cursor opens its own pane and widens the column to fit it.
        if self.table_stale:
            self.table_stale = False
            self.find_table()
        in_table = self.table_span is not None
        used.display, counted.display = bool(self.show_usage), bool(self.show_words)
        side.display = bool(self.show_usage or self.show_words or in_table)
        side.styles.width = max(38, min(80, int(self.size.width * 0.45))) if in_table else 38
        if self.show_usage:
            width = used.content_region.width or 32
            used.update(usage.render(self.usage, self.chain, width, self.pal))
        if self.show_words and self.counts_stale:
            self.counts_stale = False
            selected = self.area.selected_text
            sel = words.count(selected) if selected.strip() else None
            counted.update(words.render(sel, words.count(self.area.text), self.words_at_open,
                                        self.pal))  # fmt: skip
        heard.update(self.heard_line(self.size.width - 8))

    def top_bar(self) -> Table:
        """`voxtex  folder › chapter.tex ●` on the left; the writing mode and the AI on
        the right."""
        pal = self.pal
        folder, _, name = self.rel.rpartition("/")
        dirty = self.area.text != self.saved_text
        path: list[tuple[str, str]] = []
        for part in folder.split("/") if folder else []:
            path += [(part, pal.overlay1), (" › ", pal.surface1)]
        left = Text.assemble(
            ("voxtex", f"bold {pal.accent}"), ("   ", ""), *path, (name, f"bold {pal.text}"),
            (" ●" if dirty else "", pal.yellow),
        )  # fmt: skip
        if self.unfiltered:
            right = Text.assemble((" unfiltered ", f"bold {pal.contrast} on {pal.peach}"))
        else:
            right = Text.assemble((" ai format ", f"{pal.text} on {pal.surface0}"))
            if self.chain.active:
                right.append(f"  {self.chain.active}", style=pal.overlay0)
        bar = Table.grid(expand=True)
        bar.add_column(no_wrap=True, overflow="ellipsis")
        bar.add_column(justify="right", no_wrap=True)
        bar.add_row(left, right)
        return bar

    def heard_line(self, width: int) -> Text:
        pal = self.pal
        if self.phase == "recording":
            secs = int(time.monotonic() - self.recording_since)
            return Text.assemble(
                ("● ", pal.red), (f"{secs // 60}:{secs % 60:02d}", pal.text),
                ("   ", ""), hints(("^space", "finish"), ("esc", "cancel"), pal=pal),
            )  # fmt: skip
        if self.said:
            return Text(_tail(self.said, width), style=f"{pal.subtext} italic")
        if self.phase == "loading" and self.download:
            done, total = self.download
            return Text.assemble(
                ("downloading the speech model (first run only)  ", pal.overlay1),
                usage.meter(done / total if total else 0, 24, pal),
                (f"  {done:,.0f} / {total:,.0f} MB", pal.text),
            )  # fmt: skip
        if self.phase == "loading":
            return Text("loading speech recognition…", style=pal.overlay0)
        if self.phase == "ready" and self.query_one("#table", TablePane).has_focus_within:
            return hints(("↑↓←→", "move"), ("↵", "edit cell"), ("r/c", "add row/column"),
                         ("d/x", "delete row/column"), ("esc", "back"), pal=pal)  # fmt: skip
        if self.phase == "ready" and self.table_span is not None:
            return hints(("^l", "edit the table"), ("^space", "talk"), pal=pal)
        if self.phase == "ready" and self.area.selected_text.strip():
            what = "replace the selection" if self.unfiltered else "say what to change"
            return hints(("^space", what), pal=pal)
        if self.phase == "ready":
            mode = "ai format" if self.unfiltered else "unfiltered"
            keys = [("^space", "talk"), ("^o", "files"), ("^t", mode), ("^g", "usage"),
                    ("^n", "words"), ("^p", "theme"), ("^q", "quit")]  # fmt: skip
            return hints(*keys, pal=pal)
        return Text("")


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
    _save_state(root, state)


def _remember(root: Path, **settings) -> None:
    """Editor settings (raw words, the usage panel) kept per thesis."""
    _save_state(root, _load_state(root) | settings)


def _on_this_machine(provider) -> bool:
    return any(h in provider.base_url for h in ("localhost", "127.0.0.1"))


def _settings_path() -> Path:
    return voxtex_config.GLOBAL_CONFIG / "editor.json"


def _load_settings() -> dict:
    """Editor settings for every thesis (the theme)."""
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(settings: dict) -> None:
    try:
        _settings_path().parent.mkdir(parents=True, exist_ok=True)
        _settings_path().write_text(json.dumps(settings, indent=1), encoding="utf-8")
    except OSError:
        pass


def _save_state(root: Path, state: dict) -> None:
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
