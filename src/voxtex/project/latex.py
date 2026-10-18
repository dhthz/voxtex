"""Small, dependency-free helpers for scanning LaTeX source.

These are deliberately not a full parser: they handle comments, balanced
brace groups and optional arguments well enough to read real-world preambles.
"""

import re
from pathlib import Path

_COMMENT = re.compile(r"(?<!\\)%.*")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments(text: str) -> str:
    """Remove `%` comments (but not escaped `\\%`), keeping line structure."""
    return _COMMENT.sub("", text)


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def read_group(text: str, pos: int, open_: str = "{", close: str = "}") -> tuple[str, int] | None:
    """Read a balanced group starting at `pos` (after whitespace).

    Returns (inner content, index after the closing delimiter), or None if
    there is no group at `pos` or it is unbalanced.
    """
    pos = skip_ws(text, pos)
    if pos >= len(text) or text[pos] != open_:
        return None
    depth = 0
    i = pos
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    return None


def read_optional(text: str, pos: int) -> tuple[str, int] | None:
    """Read a `[...]` optional argument (bracket-balanced, braces respected)."""
    pos = skip_ws(text, pos)
    if pos >= len(text) or text[pos] != "[":
        return None
    depth = 0
    i = pos
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            g = read_group(text, i)
            if g is None:
                return None
            i = g[1]
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    return None


def read_name_arg(text: str, pos: int) -> tuple[str, int] | None:
    """Read a command name given either as `{\\foo}` or bare `\\foo`."""
    pos = skip_ws(text, pos)
    g = read_group(text, pos)
    if g is not None:
        return g[0].strip(), g[1]
    m = re.compile(r"\\(?:[A-Za-z@]+|.)").match(text, pos)
    if m:
        return m.group(0), m.end()
    return None


def find_commands(text: str, names: str) -> list[re.Match]:
    """Find `\\name` or `\\name*` occurrences where `names` is a regex alternation."""
    return list(re.finditer(rf"\\({names})(\*?)(?![A-Za-z@])", text))


def split_csv(arg: str) -> list[str]:
    return [p.strip() for p in arg.split(",") if p.strip()]


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1
