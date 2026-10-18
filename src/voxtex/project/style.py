"""Pick representative samples of the author's writing to use as few-shot examples."""

import re

from pydantic import BaseModel

from voxtex.project.discover import SourceFile

PARA_MIN, PARA_MAX, PARA_IDEAL = 250, 1400, 650
MAX_PARAGRAPHS = 3

_BLOCK_ENV = re.compile(r"\\begin\s*\{(?!document\})(\w+\*?)\}.*?\\end\s*\{\1\}", re.S)
_HEADING = re.compile(r"^\s*\\(part|chapter|section|subsection|subsubsection|paragraph)\b")
_FLOAT = {
    kind: re.compile(rf"\\begin\{{{kind}\*?\}}.*?\\end\{{{kind}\*?\}}", re.S)
    for kind in ("table", "figure")
}


class Sample(BaseModel):
    kind: str  # paragraph | table | figure
    text: str
    file: str


class StyleSamples(BaseModel):
    paragraphs: list[Sample] = []
    table: Sample | None = None
    figure: Sample | None = None

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in [*self.paragraphs, self.table, self.figure]:
            if s:
                seen[s.file] = None
        return list(seen)


def _tidy(text: str) -> str:
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def _paragraph_score(p: str) -> float | None:
    """Higher is better; None if it isn't ordinary prose."""
    n = len(p)
    if not PARA_MIN <= n <= PARA_MAX or _HEADING.match(p) or "\\begin" in p or "\\item" in p:
        return None
    commands = len(re.findall(r"\\[A-Za-z]+", p))
    words = len(p.split())
    if words < 40 or commands / words > 0.15:
        return None
    score = 1.0 - abs(n - PARA_IDEAL) / PARA_MAX
    if re.search(r"\\(cite|parencite|textcite|citep|citet|autocite)", p):
        score += 0.3  # shows how citations are woven into sentences
    if re.search(r"\\(cref|Cref|ref|autoref)\{", p):
        score += 0.15
    return score


def _paragraphs(text: str) -> list[str]:
    text = _BLOCK_ENV.sub("\n\n", text)  # drop floats/equations so paragraphs are pure prose
    return [_tidy(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


def pick_samples(body: list[SourceFile], exclude: set[str] | None = None) -> StyleSamples:
    exclude = exclude or set()
    files = [f for f in body if f.rel not in exclude]
    scored: list[tuple[float, int, str, str]] = []  # (score, order, file, text)
    for order, f in enumerate(files):
        for p in _paragraphs(f.text):
            s = _paragraph_score(p)
            if s is not None:
                scored.append((s, order, f.rel, p))
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Best paragraphs, preferring different files for variety.
    chosen: list[tuple[float, int, str, str]] = []
    for pick in scored:
        if len(chosen) < MAX_PARAGRAPHS and pick[2] not in {c[2] for c in chosen}:
            chosen.append(pick)
    for pick in scored:
        if len(chosen) < MAX_PARAGRAPHS and pick not in chosen:
            chosen.append(pick)

    out = StyleSamples(
        paragraphs=[Sample(kind="paragraph", text=t, file=fn) for _, _, fn, t in chosen]
    )
    for kind, pattern in _FLOAT.items():
        for f in files:
            m = pattern.search(f.text)
            if m and (kind != "table" or "tabular" in m.group(0)):
                setattr(out, kind, Sample(kind=kind, text=_tidy(m.group(0)), file=f.rel))
                break
    return out
