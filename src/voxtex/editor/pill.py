"""The voice pill at the bottom of the editor, after Wispr Flow's: a small fixed-size
capsule that shows dots while it listens, a live waveform while you talk and a moving
shimmer while it works. It never changes size, so nothing on screen jumps."""

from rich.text import Text

from voxtex.editor.palette import VOXTEX, Palette

BARS = 9  # waveform bars, one space apart
WIDTH = BARS * 2 - 1
LEVELS = " ▁▂▃▄▅▆▇█"
DOT = "·"

IDLE, HEARING, WORKING, LOADING, OFF, NO_MIC = (
    "idle", "hearing", "working", "loading", "off", "no-mic"
)  # fmt: skip

BORDER = {  # the palette colour of the pill's border in each state
    IDLE: "surface1",
    HEARING: "red",
    WORKING: "accent",
    LOADING: "surface1",
    OFF: "surface0",
    NO_MIC: "red",
}


def border(state: str, pal: Palette = VOXTEX) -> str:
    return getattr(pal, BORDER[state])


def bar(level: float) -> str:
    """A bar for a loudness of 0..1; quiet speech still shows a little."""
    level = min(1.0, max(0.0, level)) ** 0.6
    return LEVELS[1 + min(7, int(level * 8))]


def _spaced(cells: list[tuple[str, str]]) -> Text:
    parts: list[tuple[str, str]] = []
    for i, cell in enumerate(cells):
        if i:
            parts.append((" ", ""))
        parts.append(cell)
    return Text.assemble(*parts)


def _shimmer(t: float, color: str, dim: str) -> Text:
    """Dots with a bright spot that sweeps back and forth."""
    period = 2 * (BARS - 1)
    pos = int(t * 12) % period
    head = pos if pos < BARS else period - pos
    cells = []
    for i in range(BARS):
        d = abs(i - head)
        cells.append(("●", color) if d == 0 else (DOT, color if d == 1 else dim))
    return _spaced(cells)


def _label(text: str, style: str) -> Text:
    return Text(text.center(WIDTH), style=style)


def pill_content(state: str, levels: list[float], t: float, pal: Palette = VOXTEX) -> Text:
    """The inside of the pill: always WIDTH cells wide."""
    if state == HEARING:
        recent = ([0.0] * BARS + levels)[-BARS:]
        # Newest in the middle, older towards both ends: the wave grows from the centre.
        newest = recent[: -(BARS // 2 + 2) : -1]
        shape = newest[:0:-1] + newest
        return _spaced([(bar(v), pal.text) for v in shape])
    if state == WORKING:
        return _shimmer(t, pal.accent, pal.surface1)
    if state == LOADING:
        return _shimmer(t, pal.overlay0, pal.surface1)
    if state == OFF:
        return _label("mic off", pal.overlay0)
    if state == NO_MIC:
        return _label("no mic", pal.red)
    return _spaced([(DOT, pal.overlay0)] * BARS)
