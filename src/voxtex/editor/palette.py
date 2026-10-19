"""The editor's colours. `charcoal` (the default) is neutral greys with a dark accent and
vivid colours for LaTeX and graphs; `voxtex` is Catppuccin Mocha with a blue accent, as
herdr uses it; every Textual theme (nord, gruvbox, dracula, tokyo-night, solarized, ...)
works too. The background is always the
terminal's own, so the editor stays black on a black terminal whatever the theme.

Widgets get the colours as CSS variables (`$vx-text`, `$vx-accent`, ...); code that draws
Rich text reads them from a `Palette`."""

from dataclasses import asdict, dataclass, replace

from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

DEFAULT = "charcoal"


@dataclass(frozen=True)
class Palette:
    text: str
    subtext: str
    overlay1: str  # faint text: labels, folders
    overlay0: str  # fainter: hints, headings
    surface1: str  # borders, selection
    surface0: str  # cursor line, quiet borders
    accent: str  # focus, the active thing
    mauve: str
    teal: str
    green: str
    yellow: str
    peach: str
    red: str
    command: str  # LaTeX commands
    contrast: str = "black"  # text on an accent-coloured button

    def css(self) -> dict[str, str]:
        return {f"vx-{name}": value for name, value in asdict(self).items()}


VOXTEX = Palette(
    text="#cdd6f4", subtext="#a6adc8", overlay1="#7f849c", overlay0="#6c7086",
    surface1="#45475a", surface0="#313244", accent="#89b4fa", mauve="#cba6f7",
    teal="#94e2d5", green="#a6e3a1", yellow="#f9e2af", peach="#fab387", red="#f38ba8",
    command="#89b4fa",
)  # fmt: skip


CHARCOAL = Palette(
    text="#e0e0e0", subtext="#bdbdbd", overlay1="#8f8f8f", overlay0="#727272",
    surface1="#474747", surface0="#2e2e2e", accent="#5c5c5c", mauve="#c792ea",
    teal="#4fd6be", green="#8fd47e", yellow="#f2c86e", peach="#f7a072", red="#f2707a",
    command="#7aa7ff", contrast="#f0f0f0",
)  # fmt: skip


def from_theme(theme: Theme) -> Palette:
    """A palette from a Textual theme: its colours, and faint shades of its text towards
    its background."""
    background = Color.parse(theme.background or ("#000000" if theme.dark else "#ffffff"))
    text = Color.parse(theme.foreground or ("#e0e0e0" if theme.dark else "#202020"))

    def shade(amount: float) -> str:
        return background.blend(text, amount).hex

    primary = theme.primary
    return Palette(
        text=text.hex, subtext=shade(0.8), overlay1=shade(0.6), overlay0=shade(0.48),
        surface1=shade(0.3), surface0=shade(0.16),
        accent=primary, mauve=theme.secondary or primary, teal=theme.accent or primary,
        green=theme.success or "#a6e3a1", yellow=theme.warning or "#f9e2af",
        peach=theme.accent or theme.warning or primary, red=theme.error or "#f38ba8",
        command=primary,
    )  # fmt: skip


def _own(name: str, pal: Palette) -> Theme:
    return Theme(
        name=name, primary=pal.accent, secondary=pal.mauve, accent=pal.peach,
        foreground=pal.text, background="#000000", surface=pal.surface0, panel=pal.surface1,
        success=pal.green, warning=pal.yellow, error=pal.red, dark=True,
    )  # fmt: skip


def _themes() -> dict[str, tuple[Theme, Palette]]:
    found = {
        "charcoal": (_own("charcoal", CHARCOAL), CHARCOAL),
        "voxtex": (_own("voxtex", VOXTEX), VOXTEX),
    }
    for name, theme in sorted(BUILTIN_THEMES.items()):
        if not name.startswith(("ansi", "textual-ansi")):  # those have no colours of their own
            found[name] = (theme, from_theme(theme))
    return {
        name: (replace(theme, variables={**theme.variables, **pal.css()}), pal)
        for name, (theme, pal) in found.items()
    }


THEMES = _themes()


def palette(name: str) -> Palette:
    return THEMES.get(name, THEMES[DEFAULT])[1]
