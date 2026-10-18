"""Per-request retrieval: the few bib entries, labels and macros the speech is about.

Everything is matched on normalised tokens (lowercase, accents stripped, Greek
transliterated to Latin), so "Χαλκίδης 2020" finds chalkidis2020 and a Whisper
spelling like "Halkidis" still matches fuzzily.
"""

import re
import unicodedata
from difflib import SequenceMatcher

from voxtex.project.bib import BibEntry
from voxtex.project.labels import Label
from voxtex.project.preamble import Macro

MAX_BIB = 3
MAX_LABELS = 8
MAX_MACROS = 12
ALWAYS_MACROS = 5  # most-used macros are always offered
SURNAME_RATIO = 0.82

_GREEK_DIGRAPHS = {"ου": "ou", "μπ": "b", "ντ": "d", "γκ": "g", "γγ": "ng", "αι": "ai", "ει": "i"}
_GREEK = dict(
    zip(
        "αβγδεζηθικλμνξοπρσςτυφχψω",
        [
            "a",
            "v",
            "g",
            "d",
            "e",
            "z",
            "i",
            "th",
            "i",
            "k",
            "l",
            "m",
            "n",
            "x",
            "o",
            "p",
            "r",
            "s",
            "s",
            "t",
            "y",
            "f",
            "ch",
            "ps",
            "o",
        ],  # fmt: skip
        strict=True,
    )
)
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using", "paper", "study",
    "their", "about", "which", "what", "when", "where", "towards", "learning",
}  # fmt: skip

_KIND_WORDS = {
    "figure": ("figure", "fig", "plot", "graph", "image", "diagram",
               "σχημα", "εικονα", "διαγραμμα"),
    "table": ("table", "tabular", "πινακα", "πινακας"),
    "equation": ("equation", "formula", "εξισωση", "τυπος"),
    "section": ("section", "subsection", "ενοτητα", "παραγραφο"),
    "chapter": ("chapter", "κεφαλαιο"),
}  # fmt: skip


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def translit(s: str) -> str:
    """Lowercase, accent-free, Greek → Latin."""
    s = strip_accents(s.lower())
    for gr, lat in _GREEK_DIGRAPHS.items():
        s = s.replace(gr, lat)
    return "".join(_GREEK.get(c, c) for c in s)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", translit(text))


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    # Whisper often drops/changes the initial (Chalkidis → Halkidis), so compare loosely.
    return SequenceMatcher(None, a, b).ratio() >= SURNAME_RATIO


def kinds_mentioned(text: str) -> set[str]:
    norm = strip_accents(text.lower())
    words = set(re.findall(r"\w+", norm))
    return {
        kind
        for kind, keys in _KIND_WORDS.items()
        if any(w.startswith(k) for w in words for k in keys)
    }


def match_bib(text: str, entries: list[BibEntry], limit: int = MAX_BIB) -> list[BibEntry]:
    toks = [t for t in tokens(text) if len(t) >= 3 and t not in _STOP]
    if not toks:
        return []
    years = {t for t in toks if re.fullmatch(r"(19|20)\d\d", t)}
    words = {t for t in toks if not t.isdigit()}
    scored: list[tuple[float, int, BibEntry]] = []
    for order, e in enumerate(entries):
        score = 0.0
        named = False
        for i, surname in enumerate(e.authors[:3]):
            parts = tokens(surname)
            if parts and any(_similar(parts[-1], w) for w in words):
                score += 4 if i == 0 else 2
                named = True
                break
        key_words = set(tokens(e.key)) - {e.year}
        if any(_similar(k, w) for k in key_words if len(k) >= 4 for w in words):
            score += 1 if named else 3
            named = True
        if e.year and e.year in years:
            score += 2
        title_hits = len({t for t in tokens(e.title) if len(t) >= 4 and t not in _STOP} & words)
        score += min(title_hits, 3)
        if named or title_hits >= 2:
            scored.append((score, order, e))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [e for _, _, e in scored[:limit]]


def match_labels(text: str, labels: list[Label], limit: int = MAX_LABELS) -> list[Label]:
    words = {t for t in tokens(text) if len(t) >= 3 and t not in _STOP}
    kinds = kinds_mentioned(text)
    picked: dict[str, Label] = {}
    for lab in labels:
        parts = [t for t in tokens(lab.name) if len(t) >= 3]
        if any(_similar(p, w) for p in parts for w in words):
            picked.setdefault(lab.name, lab)
    for kind in kinds:
        # Recent labels of that kind ("the previous table").
        for lab in [lab for lab in labels if lab.kind == kind][-3:]:
            picked.setdefault(lab.name, lab)
    return list(picked.values())[:limit]


def match_macros(
    text: str, nearby: str, macros: list[Macro], always: list[str], limit: int = MAX_MACROS
) -> list[Macro]:
    squashed = "".join(tokens(text))
    in_nearby = set(re.findall(r"\\([A-Za-z]+)", nearby))
    by_name = {m.name: m for m in macros}
    picked: dict[str, Macro] = {}
    for name in always:
        if name in by_name:
            picked[name] = by_name[name]
    for m in sorted(macros, key=lambda m: -m.uses)[:ALWAYS_MACROS]:
        if m.uses:
            picked.setdefault(m.name, m)
    for m in macros:
        bare = m.name[1:].lower()
        relevant = (len(bare) >= 3 and bare in squashed) or m.name[1:] in in_nearby
        if relevant and (m.kind not in ("renewcommand", "let") or m.uses):
            picked.setdefault(m.name, m)
    return list(picked.values())[:limit]


def macro_signature(m: Macro) -> str:
    """`\\en{…}`, `\\metric{…}[…]`, `\\legalbert`."""
    required = m.arity - m.optional
    return m.name + "[…]" * m.optional + "{…}" * required
