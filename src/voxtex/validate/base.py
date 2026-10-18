"""Shared types for validators."""

import re
from dataclasses import dataclass, field

from voxtex.project.index import ProjectIndex
from voxtex.project.latex import read_group, read_optional, skip_ws

CITE_CMDS = (
    "cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|parencite|Parencite|"
    "textcite|Textcite|autocite|Autocite|footcite|footcitetext|smartcite|supercite|nocite|"
    "fullcite|citetitle"
)
REF_CMDS = (
    "ref|cref|Cref|crefrange|Crefrange|autoref|Autoref|eqref|pageref|vref|Vref|nameref|labelcref"
)


@dataclass
class Issue:
    code: str  # citation | ref | label | number | syntax | compile
    message: str
    error: bool = True  # False = warning: shown, but doesn't trigger a retry


@dataclass
class ValidationContext:
    idx: ProjectIndex
    transcript: str  # what the speaker said for this decision
    replaced: str = ""  # previous text of the block being replaced, if any
    other_labels: set[str] = field(default_factory=set)  # labels in other blocks this session
    context: str = ""  # document text the model was shown (numbers there aren't invented)


@dataclass
class CommandArg:
    name: str  # command name without backslash
    arg: str  # content of the mandatory argument
    start: int
    end: int


def command_args(text: str, names: str) -> list[CommandArg]:
    """Every `\\name*[..][..]{arg}` occurrence; the mandatory argument is returned."""
    out = []
    for m in re.finditer(rf"\\({names})\*?(?![A-Za-z@])", text):
        pos = m.end()
        for _ in range(2):  # biblatex allows [prenote][postnote]
            o = read_optional(text, pos)
            if o is None:
                break
            pos = o[1]
        g = read_group(text, skip_ws(text, pos))
        if g is not None:
            out.append(CommandArg(m.group(1), g[0], m.start(), g[1]))
    return out


def split_keys(arg: str) -> list[str]:
    return [k.strip() for k in arg.split(",") if k.strip()]
