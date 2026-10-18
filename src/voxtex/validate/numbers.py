"""No invented numbers: every number in the output must have been said, or already be in the
document text the model was shown (nearby text, recent blocks, the block being replaced).

Ignored: digits inside command names, cite keys, labels and refs, optional arguments,
column specs, lengths and graphics options, and math sub/superscripts like F_1 or x^2.
Spoken number words count, in English and Greek ("zero point six one", "τρία").
"""

import re
import unicodedata

from voxtex.project.latex import read_group, skip_ws
from voxtex.validate.base import CITE_CMDS, REF_CMDS, Issue, ValidationContext

# Commands whose (first) mandatory argument is structural, not prose.
_STRUCTURAL = (
    rf"label|{REF_CMDS}|{CITE_CMDS}|includegraphics|input|include|hspace|vspace|setlength|"
    r"addtolength|rule|resizebox|scalebox|fontsize|linespread|setcounter|addtocounter|"
    r"multicolumn|multirow|cline|cmidrule|arraystretch|renewcommand|url|href|cellcolor|"
    r"rowcolor|color|textcolor|definecolor|bibliographystyle|bibliography|addbibresource"
)
_SKIP_ARG = re.compile(rf"\\(?:{_STRUCTURAL})\*?(?![A-Za-z@])")
_BEGIN = re.compile(r"\\begin\s*\{(tabular\*?|tabularx|longtable|array|tabu|minipage|wrapfigure)\}")
_OPTIONAL = re.compile(r"\[[^\[\]]*\]")
_SUBSUP = re.compile(r"[_^](?:\{[^{}]*\}|\\?[A-Za-z0-9])")
_COMMAND = re.compile(r"\\[A-Za-z@]+")
# A number not glued to letters (F1, L2, 3D, BERT2 are names, not claims).
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)(?![\w])")

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "μηδεν": 0, "ενα": 1, "ενας": 1, "μια": 1, "δυο": 2, "τρια": 3,
    "τρεις": 3, "τεσσερα": 4, "τεσσερις": 4, "πεντε": 5, "εξι": 6, "επτα": 7, "εφτα": 7,
    "οκτω": 8, "οχτω": 8, "εννεα": 9, "εννια": 9, "δεκα": 10, "εντεκα": 11, "δωδεκα": 12,
    "εικοσι": 20, "τριαντα": 30, "σαραντα": 40, "πενηντα": 50, "εξηντα": 60, "εβδομηντα": 70,
    "ογδοντα": 80, "ενενηντα": 90, "εκατο": 100, "χιλια": 1000,
}  # fmt: skip
_DOUBLE_DIGIT_START = 20


def _norm_word(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


def spoken_numbers(text: str) -> set[float]:
    """Numbers in the transcript: digits plus simple number words and decimals."""
    found = {_to_float(m) for m in _NUMBER.findall(text)}
    # Number words never run across punctuation ("six three, three epochs").
    for segment in re.split(r"[,;:!?()]|\.(?!\d)", text):
        found |= _number_words([_norm_word(w) for w in re.findall(r"\w+", segment)])
    return found


def _number_words(words: list[str]) -> set[float]:
    found: set[float] = set()
    i = 0
    while i < len(words):
        if words[i] not in _WORDS:
            i += 1
            continue
        whole, i = _read_int(words, i)
        found.add(float(whole))
        if i < len(words) and words[i] in ("point", "κομμα"):
            digits, j = "", i + 1
            while j < len(words) and words[j] in _WORDS:
                v, j2 = _read_int(words, j)
                digits += str(v)
                j = j2
            if digits:
                found.add(float(f"{whole}.{digits}"))
            i = j
    return found


def _read_int(words: list[str], i: int) -> tuple[int, int]:
    """Read 'seventy one' / 'εξήντα ένα' / 'two hundred' starting at i."""
    total = _WORDS[words[i]]
    i += 1
    if total >= _DOUBLE_DIGIT_START and total < 100 and i < len(words):
        nxt = _WORDS.get(words[i])
        if nxt is not None and nxt < 10:
            total += nxt
            i += 1
    while i < len(words) and words[i] in ("hundred", "thousand", "εκατο", "χιλια"):
        total *= _WORDS[words[i]]
        i += 1
    return total, i


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def prose_numbers(latex: str) -> list[str]:
    """Numeric literals in the output after removing structural parts."""
    text = latex
    # Drop arguments of structural commands (with their optional args).
    out, pos = [], 0
    for m in _SKIP_ARG.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos : m.start()])
        end = m.end()
        while True:
            end = skip_ws(text, end)
            if end < len(text) and text[end] in "[(":  # [opt] or booktabs' (lr)
                close = text.find("]" if text[end] == "[" else ")", end)
                end = close + 1 if close > 0 else end + 1
                continue
            g = read_group(text, end)
            if g is None:
                break
            end = g[1]
            if m.group(0).rstrip("*") not in ("\\multicolumn", "\\multirow", "\\resizebox"):
                break  # one argument for most; these take two structural ones
        pos = end
    out.append(text[pos:])
    text = "".join(out)
    # Column specs: \begin{tabular}{lS[table-format=1.2]}, \begin{minipage}{0.5\textwidth}
    for m in list(_BEGIN.finditer(text))[::-1]:
        g = read_group(text, m.end())
        if g:
            text = text[: m.end()] + text[g[1] :]
    text = _OPTIONAL.sub(" ", text)
    text = _SUBSUP.sub(" ", text)
    text = _COMMAND.sub(" ", text)
    return _NUMBER.findall(text)


def check_numbers(latex: str, ctx: ValidationContext) -> list[Issue]:
    seen = prose_numbers(ctx.replaced) + prose_numbers(ctx.context)
    allowed = spoken_numbers(ctx.transcript) | {_to_float(n) for n in seen}
    issues = []
    for n in dict.fromkeys(prose_numbers(latex)):
        if _to_float(n) not in allowed:
            issues.append(Issue("number", f"number {n} was not said by the speaker"))
    return issues
