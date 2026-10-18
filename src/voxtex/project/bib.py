"""Parse .bib files into compact entries (key, surnames, title, year)."""

import re
import unicodedata
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser
from pydantic import BaseModel

from voxtex.project.latex import read_text

# Symbol accents (\'e, \'{e}) and letter accents, which need braces or a space (\v{s}, \v s)
# so commands like \url aren't mistaken for \u + rl.
_ACCENT = re.compile(
    r"\\(?:([`'^\"~=.])\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z]))"
    r"|([uvHcdbkr])(?:\s*\{\s*([A-Za-z])\s*\}|\s+([A-Za-z])))"
)
_COMBINING = {
    "`": "\u0300", "'": "\u0301", "^": "\u0302", "~": "\u0303", "=": "\u0304", "u": "\u0306",
    ".": "\u0307", '"': "\u0308", "r": "\u030a", "H": "\u030b", "v": "\u030c", "d": "\u0323",
    "c": "\u0327", "k": "\u0328", "b": "\u0331",
}  # fmt: skip
_SPECIAL = {r"\L": "Ł", r"\l": "ł", r"\o": "ø", r"\O": "Ø", r"\ss": "ß", r"\ae": "æ", r"\aa": "å",
            r"\i": "i", r"\j": "j"}  # fmt: skip


class BibEntry(BaseModel):
    key: str
    type: str = ""
    authors: list[str] = []  # surnames
    title: str = ""
    year: str = ""
    source: str = ""

    @property
    def short(self) -> str:
        """Compact form used in prompts: `key | first author | year | short title`."""
        first = self.authors[0] if self.authors else "?"
        title = self.title if len(self.title) <= 60 else self.title[:57].rstrip() + "..."
        return f"{self.key} | {first} | {self.year or '?'} | {title}"


def _accent(m: re.Match) -> str:
    accent = m.group(1) or m.group(4)
    letter = m.group(2) or m.group(3) or m.group(5) or m.group(6)
    return letter + _COMBINING[accent]


def clean_latex(s: str) -> str:
    """Turn `{\\L}ukasz` / `{LEGAL-BERT}` / `M{\\\"u}ller` into plain Unicode text."""
    for k, v in _SPECIAL.items():
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", v, s)
    s = _ACCENT.sub(_accent, s)
    s = re.sub(r"\\[A-Za-z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s).strip())


def surnames(author_field: str) -> list[str]:
    names = []
    for raw in re.split(r"\s+and\s+", author_field.strip()):
        raw = raw.strip()
        if not raw or raw.lower() == "others":
            continue
        if raw.startswith("{") and raw.endswith("}") and raw.count("{") == 1:
            names.append(clean_latex(raw))  # corporate author: {Google Research}
            continue
        if "," in raw:
            last = raw.split(",")[0]
        else:
            # "First von Last": take everything from the first lowercase particle, else last word.
            parts = raw.split()
            idx = next((i for i, w in enumerate(parts[:-1]) if w[:1].islower()), len(parts) - 1)
            last = " ".join(parts[idx:])
        names.append(clean_latex(last))
    return names


def parse_bib(path: Path) -> list[BibEntry]:
    parser = BibTexParser(common_strings=True, ignore_nonstandard_types=False)
    db = bibtexparser.loads(read_text(path), parser=parser)
    entries = []
    for e in db.entries:
        year = e.get("year") or e.get("date", "")[:4]
        entries.append(
            BibEntry(
                key=e["ID"],
                type=e.get("ENTRYTYPE", ""),
                authors=surnames(e.get("author") or e.get("editor", "")),
                title=clean_latex(e.get("title", "")),
                year=year.strip(),
                source=path.name,
            )
        )
    return entries


def parse_bibs(paths: list[Path]) -> list[BibEntry]:
    out: dict[str, BibEntry] = {}
    for p in paths:
        for e in parse_bib(p):
            out.setdefault(e.key, e)
    return list(out.values())
