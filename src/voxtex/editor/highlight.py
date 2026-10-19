"""LaTeX syntax colouring for the editor, by regular expressions (no parser needed)."""

import re

_TOKEN = re.compile(
    r"(?P<comment>(?<!\\)%.*$)"
    r"|(?P<command>\\(?:[A-Za-z@]+\*?|.))"
    r"|(?P<math>\$\$?(?:[^$\\]|\\.)*\$?\$)"
    r"|(?P<brace>[{}\[\]])"
)
_STRUCTURE = {
    "begin", "end", "chapter", "section", "subsection", "subsubsection", "paragraph",
    "item", "caption", "centering",
}  # fmt: skip
_REFS = {"cite", "parencite", "textcite", "autocite", "citep", "citet", "ref", "cref", "Cref",
         "eqref", "autoref", "label", "pageref", "nameref"}  # fmt: skip
CHECK_PREFIX = "% voxtex-check"


def latex_spans(line: str) -> list[tuple[int, int, str]]:
    """(start, end, style name) character ranges for one line."""
    if line.lstrip().startswith(CHECK_PREFIX):
        return [(0, len(line), "voxtex.check")]
    out = []
    for m in _TOKEN.finditer(line):
        kind = m.lastgroup
        if kind == "command":
            name = m.group().lstrip("\\").rstrip("*")
            if name in _STRUCTURE:
                kind = "structure"
            elif name in _REFS:
                kind = "ref"
        out.append((m.start(), m.end(), f"latex.{kind}"))
    return out


def char_to_byte(line: str) -> list[int]:
    """Byte offset (UTF-8) of every character position, plus the end."""
    offsets, total = [], 0
    for ch in line:
        offsets.append(total)
        total += len(ch.encode("utf-8"))
    offsets.append(total)
    return offsets
