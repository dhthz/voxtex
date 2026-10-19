"""Menus in herdr's style: a panel with a bold title, a faint subtitle, an `esc close`
button and a footer of key hints. Esc or a click outside always closes them. Colours come
from the theme (`$vx-*`); the background stays the terminal's own."""

import os
from pathlib import Path

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from voxtex.editor.palette import THEMES, VOXTEX, Palette

SKIP_DIRS = {"build", "out", "node_modules", "__pycache__"}
MAX_FILES = 500


def hints(*pairs: tuple[str, str], pal: Palette = VOXTEX) -> Text:
    """`key label` pairs: the key bright, the label faint, like herdr's footers."""
    parts: list[tuple[str, str]] = []
    for i, (key, label) in enumerate(pairs):
        if i:
            parts.append(("  ·  ", pal.overlay0))
        parts += [(key, pal.text), (" " + label, pal.overlay0)]
    return Text.assemble(*parts)


def tex_files(root: Path) -> list[str]:
    """Every .tex file under the thesis folder (not in hidden or build folders), sorted."""
    found: list[str] = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        rel = Path(folder).relative_to(root)
        found += [(rel / f).as_posix() for f in sorted(files) if f.endswith(".tex")]
        if len(found) >= MAX_FILES:
            break
    return found[:MAX_FILES]


class Menu(ModalScreen):
    """The panel every menu shares, with a list of choices (`self.shown[i]` is the choice
    behind option i; "" for headings)."""

    DEFAULT_CSS = """
    Menu { align: center middle; background: ansi_default; }
    Menu > #panel {
        width: 68; max-width: 90%; height: auto; max-height: 80%;
        background: ansi_default; border: round $vx-surface1; padding: 0 1;
    }
    Menu #head { height: 2; margin-bottom: 1; }
    Menu #titles { width: 1fr; height: 2; }
    Menu #close {
        width: auto; height: 1; padding: 0 1; background: $vx-accent; color: $vx-contrast;
        text-style: bold;
    }
    Menu #filter {
        height: 1; border: none; padding: 0; background: ansi_default; color: $vx-text;
    }
    Menu #filter > .input--placeholder { color: $vx-overlay0; }
    Menu #filter:focus { border: none; }
    Menu OptionList {
        height: auto; max-height: 24; margin-top: 1; border: none; padding: 0;
        background: ansi_default; scrollbar-size-vertical: 1;
    }
    Menu OptionList:focus { border: none; }
    Menu OptionList > .option-list--option-highlighted,
    Menu OptionList:focus > .option-list--option-highlighted {
        background: $vx-surface0; color: $vx-text; text-style: none;
    }
    Menu #footer { height: 1; margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("down", "move(1)", show=False),
        Binding("up", "move(-1)", show=False),
        Binding("enter", "choose", show=False),
    ]

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.title_text, self.subtitle = title, subtitle
        self.shown: list[str] = []

    @property
    def pal(self) -> Palette:
        return getattr(self.app, "pal", VOXTEX)

    def head(self) -> ComposeResult:
        pal = self.pal
        with Horizontal(id="head"):
            yield Static(
                Text.assemble((self.title_text, f"bold {pal.text}"), "\n",
                              (self.subtitle, pal.overlay1)),
                id="titles",
            )  # fmt: skip
            yield Static("esc close", id="close")

    def footer(self, *pairs: tuple[str, str]) -> Static:
        return Static(hints(*pairs, pal=self.pal), id="footer")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self or getattr(event.widget, "id", None) == "close":
            self.action_close()

    def action_move(self, delta: int) -> None:
        menu = self.query_one(OptionList)
        i = menu.highlighted if menu.highlighted is not None else -1
        while 0 <= i + delta < len(self.shown):
            i += delta
            if self.shown[i]:
                menu.highlighted = i
                return

    def action_choose(self) -> None:
        i = self.query_one(OptionList).highlighted
        if i is not None and self.shown[i]:
            self.dismiss(self.shown[i])

    @on(OptionList.OptionSelected)
    def chosen(self, event: OptionList.OptionSelected) -> None:
        if self.shown[event.option_index]:
            self.dismiss(self.shown[event.option_index])

    def highlight(self, wanted: str) -> None:
        menu = self.query_one(OptionList)
        first = next((i for i, f in enumerate(self.shown) if f == wanted), None)
        if first is None:
            first = next((i for i, f in enumerate(self.shown) if f), None)
        menu.highlighted = first


class FilePicker(Menu):
    """The thesis's .tex files as a tree by folder. Type to filter; ↵ opens."""

    def __init__(self, root: Path, in_thesis: list[str], current: str):
        super().__init__("open a file", root.name)
        self.files = tex_files(root)
        self.in_thesis = set(in_thesis)
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield from self.head()
            yield Input(placeholder="type to filter", id="filter")
            yield OptionList(id="files")
            yield self.footer(("↑↓", "move"), ("↵", "open"), ("esc", "close"))

    def on_mount(self) -> None:
        self.fill("")
        self.query_one("#filter", Input).focus()

    @on(Input.Changed, "#filter")
    def filtered(self, event: Input.Changed) -> None:
        self.fill(event.value)

    @on(Input.Submitted, "#filter")
    def submitted(self) -> None:
        self.action_choose()

    def fill(self, query: str) -> None:
        pal = self.pal
        words = query.lower().split()
        files = [f for f in self.files if all(w in f.lower() for w in words)]
        options: list[Option] = []
        self.shown = []
        folder = None
        for f in files:
            parent, _, name = f.rpartition("/")
            if parent != folder:
                folder = parent
                options.append(Option(Text(parent or ".", style=pal.overlay0), disabled=True))
                self.shown.append("")
            options.append(Option(self.row(f, name)))
            self.shown.append(f)
        if not files:
            options.append(Option(Text("no .tex file matches", style=pal.overlay0), disabled=True))
            self.shown.append("")
        menu = self.query_one(OptionList)
        menu.clear_options()
        menu.add_options(options)
        self.highlight(self.current)

    def row(self, path: str, name: str) -> Text:
        pal = self.pal
        mark = ("● ", pal.accent) if path == self.current else ("  ", "")
        note = "" if path in self.in_thesis else "  not in the thesis"
        style = f"bold {pal.text}" if path == self.current else pal.text
        return Text.assemble("  ", mark, (name, style), (note, pal.overlay0))


class ThemePicker(Menu):
    """Every theme, previewed as you move; ↵ keeps it, esc goes back to the one you had."""

    def __init__(self, current: str):
        super().__init__("theme", "the background stays your terminal's")
        self.current = current
        self.shown = list(THEMES)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield from self.head()
            yield OptionList(*[Option(self.row(n)) for n in self.shown], id="themes")
            yield self.footer(("↑↓", "preview"), ("↵", "keep"), ("esc", "close"))

    def row(self, name: str) -> Text:
        pal = THEMES[name][1]
        swatch = [("● ", c) for c in (pal.accent, pal.mauve, pal.green, pal.yellow, pal.red)]
        mark = ("● ", self.pal.accent) if name == self.current else ("  ", "")
        return Text.assemble("  ", mark, (name.ljust(24), self.pal.text), *swatch)

    def on_mount(self) -> None:
        self.highlight(self.current)
        self.query_one(OptionList).focus()

    @on(OptionList.OptionHighlighted)
    def preview(self, event: OptionList.OptionHighlighted) -> None:
        self.app.apply_theme(self.shown[event.option_index])

    def action_close(self) -> None:
        self.app.apply_theme(self.current)
        self.dismiss(None)
