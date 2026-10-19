"""Where dictated text goes in the document, and finding earlier blocks again.

Pure functions on lists of lines and (row, column) locations, so they're easy to test.
"""

from dataclasses import dataclass

Location = tuple[int, int]


@dataclass
class Placement:
    at: Location  # where to insert
    text: str  # what to insert, separators included
    start: Location  # where the block itself starts
    end: Location  # where it ends


def advance(loc: Location, text: str) -> Location:
    """The location after inserting `text` at `loc`."""
    row, col = loc
    if "\n" not in text:
        return row, col + len(text)
    return row + text.count("\n"), len(text.rsplit("\n", 1)[1])


def _is_paragraph(latex: str) -> bool:
    """Environments, several lines or a check note: these get their own paragraph."""
    return "\n" in latex.strip() or latex.lstrip().startswith("\\begin")


def place_block(lines: list[str], cursor: Location, latex: str) -> Placement:
    """Where to put `latex` given the cursor:

    - on an empty line: right there, as its own paragraph (blank lines kept around it)
    - inside a paragraph, for a sentence: at the cursor, with spaces as needed
    - inside a paragraph, for a table, figure or several paragraphs: after that paragraph
    """
    lines = lines or [""]
    row, col = cursor
    row = min(row, len(lines) - 1)
    line = lines[row]
    col = min(col, len(line))

    def blank(r: int) -> bool:
        return 0 <= r < len(lines) and not lines[r].strip()

    if not line.strip():
        prefix = "\n" if row > 0 and not blank(row - 1) else ""
        suffix = "\n" if row + 1 < len(lines) and not blank(row + 1) else ""
        at = (row, 0)
        text = prefix + latex + suffix
    elif _is_paragraph(latex):
        end = row
        while end + 1 < len(lines) and lines[end + 1].strip():
            end += 1
        prefix = "\n\n"
        at = (end, len(lines[end]))
        text = prefix + latex
    else:
        before = line[col - 1] if col > 0 else ""
        after = line[col] if col < len(line) else ""
        prefix = " " if before and not before.isspace() else ""
        suffix = " " if after and not after.isspace() else ""
        at = (row, col)
        text = prefix + latex + suffix
    start = advance(at, prefix)
    return Placement(at, text, start, advance(start, latex))


def find_unique(text: str, needle: str) -> int | None:
    """Offset of `needle` in `text` if it occurs exactly once (else the author has
    changed or duplicated it, and voxtex shouldn't guess)."""
    if not needle:
        return None
    first = text.find(needle)
    if first < 0 or text.find(needle, first + 1) >= 0:
        return None
    return first
