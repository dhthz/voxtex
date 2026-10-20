"""Tables as cells: find the `tabular` around the cursor, read it into rows of cells, and
write it back as LaTeX with aligned columns. The column spec, the rules between rows
(\\hline, \\toprule, \\cline{2-3}, ...) and anything after a row's `\\\\` are kept."""

import re
from dataclasses import dataclass, field

# Environment → how many {...} arguments come before the body.
ENVS = {"tabular": 1, "tabular*": 2, "tabularx": 2, "tabulary": 2, "longtable": 1}
_BEGIN = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in ENVS) + r")\}")
_RULE = re.compile(
    r"\s*(\\(?:hline|toprule|midrule|bottomrule|addlinespace|endhead|endfirsthead|endfoot"
    r"|endlastfoot)(?![A-Za-z])(?:\[[^\]]*\])?|\\c(?:mid)?(?:line|rule)(?:\([^)]*\))?\{[^}]*\})"
)
_MULTICOLUMN = re.compile(r"\\multicolumn\{(\d+)\}")


@dataclass
class Table:
    head: str  # \begin{tabular}{|c|c|} exactly as written
    env: str
    rows: list[list[str]]  # each cell's LaTeX
    rules: dict[int, list[str]] = field(default_factory=dict)  # rules before row i
    ends: list[str] = field(default_factory=list)  # after each row's \\ ("" or "[2pt]")
    indent: str = ""  # before \end{...}
    inner: str = "  "  # before each row and rule

    @property
    def columns(self) -> int:
        return max((span(r) for r in self.rows), default=0)

    def header_rows(self) -> int:
        """Rows above the first rule that follows a row (the table's head), 0 if none."""
        for i in range(1, len(self.rows)):
            if self.rules.get(i):
                return i
        return 0


def span(row: list[str]) -> int:
    """Columns a row covers, counting \\multicolumn{n}."""
    return sum(int(m.group(1)) if (m := _MULTICOLUMN.match(c.strip())) else 1 for c in row)


def find_table(text: str, index: int) -> tuple[int, int] | None:
    """The (start, end) of the table environment around `index`, if any."""
    for m in _BEGIN.finditer(text):
        if m.start() > index:
            break
        end = _matching_end(text, m.end(), m.group(1))
        if end is not None and index <= end:
            return m.start(), end
    return None


def _matching_end(text: str, pos: int, env: str) -> int | None:
    """Where `\\end{env}` closes the environment opened just before `pos` (nesting counted)."""
    begin, end = f"\\begin{{{env}}}", f"\\end{{{env}}}"
    depth = 1
    while depth:
        nb, ne = text.find(begin, pos), text.find(end, pos)
        if ne < 0:
            return None
        if 0 <= nb < ne:
            depth, pos = depth + 1, nb + len(begin)
        else:
            depth, pos = depth - 1, ne + len(end)
    return pos


def _group(text: str, pos: int, open_: str, close: str) -> int | None:
    """End of a balanced `open_ ... close` group starting at `pos` (after whitespace)."""
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos >= len(text) or text[pos] != open_:
        return None
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "\\":
            continue
        if text[i] == open_ and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == close and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _split_top(text: str, sep: str) -> list[str]:
    """Split on `sep` outside braces and not escaped (\\&)."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            if sep == "\\\\" and depth == 0 and text.startswith("\\\\", i):
                parts.append(text[start:i])
                start = i = i + 2
                continue
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return parts


def parse(source: str) -> Table | None:
    """A table environment's source (from find_table) as cells; None if it can't be read."""
    m = _BEGIN.match(source)
    if not m:
        return None
    env, pos = m.group(1), m.end()
    needed = ENVS[env]
    while needed:  # optional [..] and the required {..} arguments, in any order
        opt = _group(source, pos, "[", "]")
        if opt is not None:
            pos = opt
            continue
        req = _group(source, pos, "{", "}")
        if req is None:
            return None
        pos, needed = req, needed - 1
    end = source.rfind(f"\\end{{{env}}}")
    if end < pos:
        return None
    end_line = source.rfind("\n", 0, end) + 1
    indent = source[end_line:end] if not source[end_line:end].strip() else ""
    first_row = re.search(r"\n([ \t]*)\S", source[pos:end])
    inner = first_row.group(1) if first_row else indent + "  "
    table = Table(source[:pos].strip(), env, [], indent=indent, inner=inner)
    chunks = _split_top(source[pos:end], "\\\\")
    for n, chunk in enumerate(chunks):
        row_end = ""
        if n > 0:  # what follows the previous row's \\: [2pt] or *
            em = re.match(r"\*?\s*(\[[^\]]*\])?", chunk)
            if em and em.group(1):
                table.ends[-1] = em.group(1)
            chunk = chunk[em.end() :] if em else chunk
        rules = []
        while rm := _RULE.match(chunk):
            rules.append(rm.group(1).strip())
            chunk = chunk[rm.end() :]
        if rules:
            table.rules[len(table.rows)] = rules
        if chunk.strip():
            table.rows.append([c.strip() for c in _split_top(chunk, "&")])
            table.ends.append(row_end)
    return table


def render(table: Table, last_row_breaks: bool = True) -> str:
    """The table as LaTeX: one row per line, columns aligned, rules on their own lines."""
    width = table.columns
    rows = [r + [""] * (width - span(r)) if not _MULTICOLUMN.search(" ".join(r)) else r
            for r in table.rows]  # fmt: skip
    widths = [0] * width
    for r in rows:
        if len(r) == width:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
    inner = table.inner
    lines = [table.head]  # the replaced text starts at \begin, after the indent
    for i, r in enumerate(rows):
        lines += [inner + rule for rule in table.rules.get(i, [])]
        cells = [c.ljust(widths[j]) for j, c in enumerate(r)] if len(r) == width else r
        last = i == len(rows) - 1
        end = " \\\\" + table.ends[i] if (not last or last_row_breaks) else ""
        lines.append(inner + " & ".join(cells).rstrip() + end)
    lines += [inner + rule for rule in table.rules.get(len(rows), [])]
    lines.append(table.indent + f"\\end{{{table.env}}}")
    return "\n".join(lines)


def ends_with_break(source: str) -> bool:
    """Whether the last row of the original ended with \\\\ (kept as it was)."""
    body = source[: source.rfind("\\end{")]
    body = _RULE.sub("", body).rstrip()
    return body.endswith("\\\\") or bool(re.search(r"\\\\\s*\[[^\]]*\]$", body))
