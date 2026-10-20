"""Word count of what the PDF will show: LaTeX commands, comments, labels, citations and
maths don't count; the text inside `\\section{...}`, `\\textbf{...}` or `\\en{...}` does."""

import re
from dataclasses import dataclass

from rich.table import Table
from rich.text import Text

# Commands whose arguments are never words on the page.
NO_TEXT = (
    "label|ref|cref|Cref|autoref|eqref|pageref|vref|nameref|"
    "cite|citep|citet|citealp|citeauthor|citeyear|parencite|textcite|autocite|footcite|nocite|"
    "includegraphics|input|include|subfile|usepackage|documentclass|bibliography|"
    "bibliographystyle|addbibresource|printbibliography|newcommand|renewcommand|"
    "providecommand|newenvironment|setlength|addtolength|hspace|vspace|"
    "begin|end|url|href|graphicspath|pagestyle|thispagestyle|setcounter|addtocounter|"
    "newpage|clearpage|cleardoublepage|tableofcontents|listoffigures|listoftables|"
    "maketitle|centering|hline|toprule|midrule|bottomrule|cline|item|noindent|"
    "small|footnotesize|large|Large|normalsize|bfseries|itshape|selectlanguage"
)
MATH_ENVS = r"equation|align|gather|multline|eqnarray|math|displaymath|flalign|alignat"

_COMMENT = re.compile(r"(?<!\\)%.*")
_MATH = re.compile(
    rf"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)"
    rf"|\\begin\{{({MATH_ENVS})\*?\}}.*?\\end\{{\1\*?\}}",
    re.S,
)
_GROUP = r"\{(?:[^{}]|\{[^{}]*\})*\}"  # one {...}, one level of nesting inside
_NO_TEXT = re.compile(rf"\\(?:{NO_TEXT})(?![A-Za-z])\*?(?:\s*(?:\[[^\]]*\]|{_GROUP}))*")
_COMMAND = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])*|\\.")
_WORD = re.compile(r"\w+(?:[-'’]\w+)*")


@dataclass
class Count:
    words: int
    formulas: int


def visible_text(latex: str) -> tuple[str, int]:
    """The text a reader sees, and how many formulas were left out."""
    m = re.search(r"\\begin\{document\}(.*?)(\\end\{document\}|$)", latex, re.S)
    text = m.group(1) if m else latex  # a main file: only the document body
    text = _COMMENT.sub("", text)
    formulas = len(_MATH.findall(text))
    text = _MATH.sub(" ", text)
    text = _NO_TEXT.sub(" ", text)
    text = _COMMAND.sub(" ", text)  # \textbf{x} -> {x}: the argument stays
    text = re.sub(r"[{}~&]", " ", text)
    return text, formulas


def count(latex: str) -> Count:
    text, formulas = visible_text(latex)
    return Count(len(_WORD.findall(text)), formulas)


def render(selection: Count | None, chapter: Count, at_open: int, pal) -> Table:
    """The words pane: the selection's count when there is one, the chapter's always."""
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(no_wrap=True, width=10)
    t.add_column(no_wrap=True, justify="right")

    def row(label: str, value: str, strong: bool) -> None:
        style = f"bold {pal.text}" if strong else pal.subtext
        t.add_row(Text(label, style=pal.overlay1), Text(value, style=style))

    if selection is not None:
        row("selection", f"{selection.words:,}", True)
    row("chapter", f"{chapter.words:,}", selection is None)
    added = chapter.words - at_open
    row("this session", f"{added:+,}" if added else "0", False)
    if chapter.formulas:
        row("formulas", f"{chapter.formulas:,}", False)
    return t
