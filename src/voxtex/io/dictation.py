"""Find `%>` dictation lines typed by an external dictation app (watch mode)."""

import re
from dataclasses import dataclass

# A dictation line; the `%> HERE` marker is not one.
_LINE = re.compile(r"^[ \t]*%>(?![ \t]*HERE\b)[ \t]?(.*?)[ \t]*$")


@dataclass
class Dictation:
    start: int  # offset of the first %> line
    end: int  # offset just past the last %> line (including its newline)
    line: int  # 1-based line number of the first %> line
    transcript: str

    def raw(self, content: str) -> str:
        return content[self.start : self.end]


def find_dictations(content: str) -> list[Dictation]:
    """Consecutive `%>` lines form one utterance; anything else (a blank line too) ends it."""
    out: list[Dictation] = []
    current: list[str] = []
    start = line_no = 0
    offset = 0
    for i, line in enumerate(content.splitlines(keepends=True), 1):
        m = _LINE.match(line.rstrip("\r\n"))
        if m:
            if not current:
                start, line_no = offset, i
            current.append(m.group(1).strip())
        elif current:
            _flush(out, current, start, offset, line_no)
            current = []
        offset += len(line)
    if current:
        _flush(out, current, start, offset, line_no)
    return out


def _flush(out: list[Dictation], parts: list[str], start: int, end: int, line: int) -> None:
    text = " ".join(p for p in parts if p).strip()
    if text:
        out.append(Dictation(start, end, line, text))
