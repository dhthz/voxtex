"""Derive writing conventions deterministically from the preamble and existing text.

Package defaults come first; if the existing chapters clearly use something
else (e.g. plain \\cite under biblatex), the observed usage wins, because the
goal is to match *this* thesis.
"""

import re
from collections import Counter

from pydantic import BaseModel

from voxtex.project.discover import SourceFile
from voxtex.project.labels import Label, LabelConventions, label_conventions
from voxtex.project.preamble import Preamble

MIN_OBSERVED = 3  # occurrences needed before observed usage overrides package defaults

_LANG_CODES = {
    "english": "en",
    "british": "en",
    "american": "en",
    "UKenglish": "en",
    "USenglish": "en",
    "australian": "en",
    "canadian": "en",
    "newzealand": "en",
    "greek": "el",
    "polutonikogreek": "el",
    "german": "de",
    "ngerman": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
}
_GREEK_CHAR = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")

_CITE_PAREN = ("parencite", "autocite", "citep", "cite", "footcite", "supercite")
_CITE_TEXT = ("textcite", "citet", "citeauthor")
_REF_CMDS = ("cref", "Cref", "autoref", "ref", "vref", "nameref")

# Nouns written before ~\ref{} when the thesis doesn't show its own.
_REF_NOUNS = {
    "en": {"figure": "Figure", "table": "Table", "chapter": "Chapter", "section": "Section"},
    "el": {"figure": "Σχήμα", "table": "Πίνακας", "chapter": "Κεφάλαιο", "section": "Ενότητα"},
}
_NOUN_BEFORE_REF = re.compile(r"(\w+)\s*~\s*\\ref\s*\{([^}]+)\}")
_PAREN_REF = re.compile(r"\(\s*\\ref\s*\{([^}]+)\}\s*\)")
# A one-argument macro that switches to a Latin-script language, e.g. \en{...}.
_LATIN_SWITCH = re.compile(
    r"\\foreignlanguage\s*\{(?:english|american|british|USenglish|UKenglish|latin)\}"
    r"|\\text(?:latin|english)\b|\\latintext\b|\\selectlanguage\s*\{english\}"
)


class Conventions(BaseModel):
    bib_package: str = "none"  # biblatex | natbib | bibtex | none
    bib_style: str = ""  # e.g. authoryear, numeric, apa; .bst name for bibtex
    cite_paren: str = "\\cite"
    cite_text: str = "\\cite"
    ref: str = "\\ref"
    ref_capital: str = ""  # sentence-initial form, e.g. \Cref
    ref_style: str = "plain"  # cleveref | autoref | plain ("Figure~\\ref{}")
    ref_nouns: dict[str, str] = {}  # plain style: kind -> noun before ~\ref, e.g. figure -> Σχήμα
    eq_ref: str = "(\\ref{label})"  # or \eqref{label}
    tables: str = "plain"  # booktabs | hline | plain
    units: str = "none"  # siunitx | none
    unit_cmd: str = ""  # \SI or \qty
    num_cmd: str = ""  # \num
    labels: LabelConventions = LabelConventions()
    languages: list[str] = []  # declared languages, main first
    language: str = "en"  # main document language code
    language_source: str = "default"  # babel | polyglossia | text | default
    latin_wrapper: str = ""  # non-Latin documents: macro for English terms, e.g. \en

    def rules(self) -> list[str]:
        """Compact one-line rules for the LLM prompt (addendum §5.3)."""
        r = []
        if self.cite_paren == self.cite_text:
            r.append(f"cite: {self.cite_paren}{{key}}")
        else:
            r.append(f"cite: {self.cite_paren} (parenthetical), {self.cite_text} (in text)")
        if self.ref_style == "cleveref":
            r.append(f"ref: {self.ref}{{label}} ({self.ref_capital} at sentence start), no ~\\ref")
        elif self.ref_style == "autoref":
            r.append("ref: \\autoref{label}")
        else:
            nouns = ", ".join(f"{n}~\\ref{{…}}" for n in self.ref_nouns.values())
            r.append(f"ref: {nouns}; equations {self.eq_ref}")
        if self.tables == "booktabs":
            r.append("tables: booktabs (\\toprule/\\midrule/\\bottomrule), no vertical rules")
        elif self.tables == "hline":
            r.append("tables: \\hline rules")
        if self.units == "siunitx":
            r.append(f"units: siunitx ({self.unit_cmd}{{value}}{{unit}}, {self.num_cmd}{{n}})")
        if self.labels.example_by_kind:
            pairs = ", ".join(f"{k} {ex}" for k, ex in sorted(self.labels.example_by_kind.items()))
            r.append(f"labels like: {pairs}")
        r.append(f"language: {self.language}")
        if self.latin_wrapper:
            r.append(f"English/Latin-script terms: wrap in {self.latin_wrapper}{{…}}")
        return r


def _count_cmds(body: list[SourceFile], names: tuple[str, ...]) -> Counter[str]:
    c: Counter[str] = Counter()
    pattern = re.compile(r"\\(" + "|".join(names) + r")\*?(?![A-Za-z@])")
    for f in body:
        c.update(pattern.findall(f.text))
    return c


def _declared_languages(pre: Preamble) -> tuple[list[str], str]:
    babel = pre.package("babel")
    if babel:
        main = babel.option("main")
        langs = [o for o in babel.options if "=" not in o] or [
            o for o in pre.class_options if o in _LANG_CODES
        ]
        if main:
            langs = [main] + [lang for lang in langs if lang != main]
        elif langs:
            langs = [langs[-1]] + langs[:-1]  # babel: last option is the main language
        if langs:
            return langs, "babel"
    return [], ""


def _polyglossia_languages(preamble_sources: list[SourceFile]) -> list[str]:
    main, others = None, []
    for f in preamble_sources:
        m = re.search(r"\\set(?:default|main)language\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", f.text)
        if m and main is None:
            main = m.group(1).strip()
        for o in re.finditer(r"\\set(?:other)?languages?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", f.text):
            others += [s.strip() for s in o.group(1).split(",")]
    if not main:
        return []
    return [main] + [o for o in others if o != main]


def _text_language(body: list[SourceFile]) -> str | None:
    greek = latin = 0
    for f in body:
        prose = re.sub(r"\\[A-Za-z@]+", " ", f.text)  # don't count command names as Latin text
        greek += len(_GREEK_CHAR.findall(prose))
        latin += len(_LATIN_CHAR.findall(prose))
    if greek + latin < 200:
        return None
    return "el" if greek > latin else "en"


def _bibtex_style(sources: list[SourceFile]) -> str | None:
    for f in sources:
        m = re.search(r"\\bibliographystyle\s*\{([^}]+)\}", f.text)
        if m:
            return m.group(1).strip().rsplit("/", 1)[-1]
    return None


def _ref_nouns(body: list[SourceFile], kinds: dict[str, str], language: str) -> dict[str, str]:
    observed: dict[str, Counter[str]] = {}
    for f in body:
        for m in _NOUN_BEFORE_REF.finditer(f.text):
            kind = kinds.get(m.group(2).strip())
            if kind and not m.group(1).isdigit():
                observed.setdefault(kind, Counter())[m.group(1)] += 1
    nouns = dict(_REF_NOUNS.get(language, _REF_NOUNS["en"]))
    nouns.update({k: c.most_common(1)[0][0] for k, c in observed.items() if k != "equation"})
    return nouns


def _latin_wrapper(pre: Preamble, body: list[SourceFile]) -> str:
    """The macro this thesis uses to wrap English terms, by usage; babel's \\textlatin otherwise."""
    candidates = [m for m in pre.macros if m.arity == 1 and _LATIN_SWITCH.search(m.body)]
    if candidates:
        uses = _count_cmds(body, tuple(re.escape(m.name[1:]) for m in candidates))
        best = max(candidates, key=lambda m: uses.get(m.name[1:], 0))
        return best.name
    return "\\textlatin" if pre.has("babel") else ""


def derive_conventions(
    pre: Preamble,
    preamble_sources: list[SourceFile],
    body: list[SourceFile],
    labels: list[Label],
) -> Conventions:
    c = Conventions(labels=label_conventions(labels))

    # Language first: some rules below depend on it. Polyglossia, then babel, then the text.
    langs = _polyglossia_languages(preamble_sources)
    source = "polyglossia" if langs else ""
    if not langs:
        langs, source = _declared_languages(pre)
    c.languages = langs
    if langs and langs[0] in _LANG_CODES:
        c.language, c.language_source = _LANG_CODES[langs[0]], source
    else:
        detected = _text_language(body)
        if detected:
            c.language, c.language_source = detected, "text"
    if c.language == "el" and source != "polyglossia":
        # pdflatex + babel-greek typesets Latin letters as Greek unless switched.
        c.latin_wrapper = _latin_wrapper(pre, body)

    # Citations
    biblatex = pre.package("biblatex")
    if biblatex:
        c.bib_package = "biblatex"
        c.bib_style = biblatex.option("style") or biblatex.option("citestyle") or "numeric"
        c.cite_paren, c.cite_text = "\\parencite", "\\textcite"
    elif pre.has("natbib"):
        c.bib_package = "natbib"
        c.cite_paren, c.cite_text = "\\citep", "\\citet"
    elif (bst := _bibtex_style(preamble_sources + body)) is not None:
        c.bib_package, c.bib_style = "bibtex", bst
    used = _count_cmds(body, _CITE_PAREN + _CITE_TEXT)
    paren_used = Counter({k: v for k, v in used.items() if k in _CITE_PAREN})
    text_used = Counter({k: v for k, v in used.items() if k in _CITE_TEXT and k != "citeauthor"})
    if paren_used and paren_used.most_common(1)[0][1] >= MIN_OBSERVED:
        c.cite_paren = "\\" + paren_used.most_common(1)[0][0]
    if text_used and text_used.most_common(1)[0][1] >= MIN_OBSERVED:
        c.cite_text = "\\" + text_used.most_common(1)[0][0]
    if c.bib_package == "none":
        c.cite_text = c.cite_paren

    # Cross-references
    if pre.has("cleveref"):
        c.ref, c.ref_capital, c.ref_style = "\\cref", "\\Cref", "cleveref"
    refs = _count_cmds(body, _REF_CMDS + ("eqref",))
    if refs.get("autoref", 0) >= MIN_OBSERVED and refs["autoref"] > refs.get("cref", 0):
        c.ref, c.ref_capital, c.ref_style = "\\autoref", "\\autoref", "autoref"
    elif c.ref_style == "cleveref" and refs.get("ref", 0) > 2 * (refs["cref"] + refs["Cref"]) + 2:
        c.ref, c.ref_capital, c.ref_style = "\\ref", "", "plain"  # cleveref loaded but unused
    kinds = {lab.name: lab.kind for lab in labels}
    if c.ref_style == "plain":
        c.ref_nouns = _ref_nouns(body, kinds, c.language)
    paren = sum(
        1 for f in body for m in _PAREN_REF.finditer(f.text) if kinds.get(m.group(1)) == "equation"
    )
    if refs.get("eqref", 0) or (pre.has("amsmath") and not paren):
        c.eq_ref = "\\eqref{label}"

    # Tables
    if pre.has("booktabs"):
        c.tables = "booktabs"
    else:
        hlines = sum(f.text.count("\\hline") for f in body)
        if hlines:
            c.tables = "hline"

    # Units
    if pre.has("siunitx"):
        c.units, c.num_cmd = "siunitx", "\\num"
        q = _count_cmds(body, ("SI", "qty"))
        c.unit_cmd = "\\qty" if q.get("qty", 0) > q.get("SI", 0) else "\\SI"
    return c
