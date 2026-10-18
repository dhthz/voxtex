"""Greek documents under pdflatex + babel need Latin-script words switched to English
(e.g. \\en{RDF}), or they come out in Greek letters. Models forget; this wraps them.

Only plain text is touched. Left alone: command names, math, optional arguments,
arguments of commands that aren't plain text formatting (labels, cite keys, units,
file names, user macros...), arguments already wrapped, and verbatim/code.
"""

import re

from voxtex.project.latex import read_group

# Commands whose arguments are ordinary text, so Latin words inside them get wrapped.
TEXT_ARG_COMMANDS = {
    "textbf", "textit", "emph", "textsc", "textsl", "underline", "caption", "footnote",
    "part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph",
    "item", "mbox", "textup", "textmd", "textnormal", "title", "centering",
}  # fmt: skip
_MATH_ENVS = r"equation|align|gather|multline|eqnarray|flalign|math|displaymath"
_CODE_ENVS = r"verbatim|lstlisting|minted|tabbing|comment"
_PROTECTED_BLOCKS = re.compile(
    rf"\\begin\{{({_MATH_ENVS}|{_CODE_ENVS})\*?\}}.*?\\end\{{\1\*?\}}"
    r"|\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\verb(.).*?\2|(?<!\\)%[^\n]*",
    re.S,
)
_COMMAND = re.compile(r"\\([A-Za-z@]+)\*?|\\.")
_WORD = r"[A-Za-z][A-Za-z0-9]*(?:[-'.][A-Za-z0-9]+)*"
# Words separated by spaces/commas, with numbers in between, form one run:
# "In our experiments, Sesame answered in 85 milliseconds" → one \en{...}
_SEP = r"[ ,;:]+(?:[0-9][0-9.,]*%?[ ,;:]+)*"
_RUN = re.compile(rf"{_WORD}(?:{_SEP}{_WORD})*")


def _protected_mask(latex: str, wrappers: set[str]) -> list[bool]:
    mask = [False] * len(latex)

    def protect(a: int, b: int) -> None:
        for i in range(a, min(b, len(latex))):
            mask[i] = True

    for m in _PROTECTED_BLOCKS.finditer(latex):
        protect(m.start(), m.end())
    for m in _COMMAND.finditer(latex):
        if mask[m.start()]:
            continue
        protect(m.start(), m.end())
        name = m.group(1)
        if not name:
            continue
        pos = m.end()
        # Optional arguments are never prose ([t], [width=...], [table-format=1.2]).
        while pos < len(latex) and latex[pos] in " [":
            if latex[pos] == " ":
                pos += 1
                continue
            close = latex.find("]", pos)
            if close < 0:
                break
            protect(pos, close + 1)
            pos = close + 1
        if name in TEXT_ARG_COMMANDS:
            continue
        if name in ("begin", "end"):
            g = read_group(latex, pos)
            if g:
                protect(pos, g[1])
                after = g[1]
                opt = re.match(r"\s*\[[^\]]*\]", latex[after:])  # \begin{table}[!tb]
                if opt:
                    protect(after, after + opt.end())
                    after += opt.end()
                if name == "begin" and re.match(r"(tabular|array|longtable)", g[0]):
                    spec = read_group(latex, after)  # column spec: lS[table-format=1.2]
                    if spec:
                        protect(after, spec[1])
            continue
        # Any other command: its mandatory arguments are not ours to rewrite
        # (\cite{key}, \label{x}, \SI{3}{ms}, \en{RDF}, \metric{dev}, ...).
        while True:
            g = read_group(latex, pos)
            if g is None:
                break
            protect(pos, g[1])
            pos = g[1]
    for w in wrappers:  # already wrapped (covered above, but be explicit)
        for m in re.finditer(re.escape(w) + r"\s*\{", latex):
            g = read_group(latex, m.end() - 1)
            if g:
                protect(m.start(), g[1])
    return mask


def wrap_latin(latex: str, wrapper: str) -> str:
    """Wrap unprotected runs of Latin-script words in `wrapper` (e.g. \\en)."""
    if not wrapper:
        return latex
    mask = _protected_mask(latex, {wrapper})
    out, last = [], 0
    for m in _RUN.finditer(latex):
        if any(mask[m.start() : m.end()]):
            continue
        if m.start() > 0 and latex[m.start() - 1] == "\\":
            continue
        out.append(latex[last : m.start()])
        out.append(f"{wrapper}{{{m.group(0)}}}")
        last = m.end()
    out.append(latex[last:])
    return "".join(out)
