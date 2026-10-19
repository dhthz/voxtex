"""The voice pill at the bottom of the editor, after Wispr Flow's: a small fixed-size
capsule that shows dots while it listens, a live waveform while you talk and a moving
shimmer while it works. It never changes size, so nothing on screen jumps."""

from rich.text import Text

from voxtex.ui import ACCENT

BARS = 9  # waveform bars, one space apart
WIDTH = BARS * 2 - 1
LEVELS = " ▁▂▃▄▅▆▇█"
DOT = "·"

IDLE, HEARING, WORKING, LOADING, OFF, NO_MIC = (
    "idle", "hearing", "working", "loading", "off", "no-mic"
)  # fmt: skip

BORDER = {
    IDLE: "#3a3a3a",
    HEARING: "#8a8a8a",
    WORKING: ACCENT,
    LOADING: "#3a3a3a",
    OFF: "#2a2a2a",
    NO_MIC: "#6b3a3a",
}


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


def _shimmer(t: float, color: str) -> Text:
    """Dots with a bright spot that sweeps back and forth."""
    period = 2 * (BARS - 1)
    pos = int(t * 12) % period
    head = pos if pos < BARS else period - pos
    cells = []
    for i in range(BARS):
        d = abs(i - head)
        cells.append(("●", color) if d == 0 else (DOT, color if d == 1 else "#4a4a4a"))
    return _spaced(cells)


def _label(text: str, style: str) -> Text:
    return Text(text.center(WIDTH), style=style)


def pill_content(state: str, levels: list[float], t: float) -> Text:
    """The inside of the pill: always WIDTH cells wide."""
    if state == HEARING:
        recent = ([0.0] * BARS + levels)[-BARS:]
        # Newest in the middle, older towards both ends: the wave grows from the centre.
        newest = recent[: -(BARS // 2 + 2) : -1]
        shape = newest[:0:-1] + newest
        return _spaced([(bar(v), "#ededed") for v in shape])
    if state == WORKING:
        return _shimmer(t, ACCENT)
    if state == LOADING:
        return _shimmer(t, "#6a6a6a")
    if state == OFF:
        return _label("mic off", "#5a5a5a")
    if state == NO_MIC:
        return _label("no mic", "#b06060")
    return _spaced([(DOT, "#555555")] * BARS)
