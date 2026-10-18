"""Locate the `%> HERE` marker that says where new text goes."""

import re
from dataclasses import dataclass
from pathlib import Path

from voxtex.project.latex import read_text

MARKER_RE = re.compile(r"^[ \t]*%>[ \t]*HERE\b.*$", re.M)
MARKER = "%> HERE"


@dataclass
class MarkerLocation:
    file: Path
    rel: str
    line: int  # 1-based line of the marker; 0 if appending at end of file (no marker)
    before: str  # file text before the marker

    @property
    def found(self) -> bool:
        return self.line > 0


def find_marker(root: Path, files: list[str]) -> MarkerLocation | None:
    """First marker in `files` (relative paths, in document order)."""
    for rel in files:
        p = root / rel
        try:
            text = read_text(p)
        except OSError:
            continue
        m = MARKER_RE.search(text)
        if m:
            line = text.count("\n", 0, m.start()) + 1
            return MarkerLocation(file=p, rel=rel, line=line, before=text[: m.start()])
    return None


def fallback_location(root: Path, files: list[str]) -> MarkerLocation | None:
    """No marker: the end of the last content file."""
    for rel in reversed(files):
        p = root / rel
        if p.is_file():
            return MarkerLocation(file=p, rel=rel, line=0, before=read_text(p))
    return None


def nearby_text(before: str, max_chars: int = 800) -> str:
    """The last ~max_chars of text before the cursor, cut at a line start, comments dropped."""
    lines = [ln for ln in before.splitlines() if not ln.lstrip().startswith("%")]
    out: list[str] = []
    size = 0
    for ln in reversed(lines):
        if size + len(ln) + 1 > max_chars and out:
            break
        out.append(ln)
        size += len(ln) + 1
    return "\n".join(reversed(out)).strip("\n")[-max_chars:]
