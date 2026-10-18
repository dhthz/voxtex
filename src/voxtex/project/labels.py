"""Collect every \\label and work out the project's label-prefix convention."""

import re
from collections import Counter, defaultdict

from pydantic import BaseModel

from voxtex.project.discover import SourceFile
from voxtex.project.latex import line_of

_LABEL = re.compile(r"\\label\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_ENV = re.compile(r"\\(begin|end)\s*\{([^}]+)\}")
_SECTION = re.compile(r"\\(part|chapter|section|subsection|subsubsection|paragraph)\*?\s*[\[{]")

_ENV_KIND = {
    "figure": "figure",
    "figure*": "figure",
    "subfigure": "figure",
    "wrapfigure": "figure",
    "table": "table",
    "table*": "table",
    "subtable": "table",
    "longtable": "table",
    "equation": "equation",
    "align": "equation",
    "gather": "equation",
    "multline": "equation",
    "eqnarray": "equation",
    "flalign": "equation",
    "lstlisting": "listing",
    "listing": "listing",
    "algorithm": "algorithm",
    "theorem": "theorem",
    "lemma": "theorem",
    "definition": "theorem",
    "proposition": "theorem",
    "corollary": "theorem",
}
_SECTION_KIND = {
    "part": "part",
    "chapter": "chapter",
    "section": "section",
    "subsection": "section",
    "subsubsection": "section",
    "paragraph": "section",
}
# A prefix is a short run of letters followed by a separator (fig:x, fig-x) or running
# straight into a number / capital (figure4.1, tableAppA.01).
_PREFIX = re.compile(r"^([A-Za-z]{1,12}?)(?:([:\-_.])|(?=[0-9A-Z]))")


class Label(BaseModel):
    name: str
    kind: str = "other"  # figure | table | equation | chapter | section | ...
    file: str = ""
    line: int = 0


class LabelConventions(BaseModel):
    separator: str = ":"  # "" when the prefix runs straight into the name (figure4.1)
    prefix_counts: dict[str, int] = {}  # "fig" -> 12
    prefix_by_kind: dict[str, str] = {}  # "figure" -> "fig"
    example_by_kind: dict[str, str] = {}  # "figure" -> "fig:pipeline"

    @property
    def summary(self) -> str:
        top = sorted(self.prefix_counts.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{p}{self.separator or '…'}" for p, _ in top)


def split_prefix(name: str) -> tuple[str, str] | None:
    """(prefix, separator) of a label name, or None if it has no recognisable prefix."""
    m = _PREFIX.match(name)
    if not m or m.end() >= len(name):
        return None
    return m.group(1), m.group(2) or ""


def _kind_at(text: str, pos: int) -> str:
    """Kind of the thing a label at `pos` refers to: innermost environment or last heading."""
    stack: list[tuple[str, int]] = []
    for m in _ENV.finditer(text, 0, pos):
        if m.group(1) == "begin":
            stack.append((m.group(2).strip(), m.start()))
        elif stack and stack[-1][0] == m.group(2).strip():
            stack.pop()
    env_kind, env_start = "other", -1
    for name, start in reversed(stack):
        if name in _ENV_KIND:
            env_kind, env_start = _ENV_KIND[name], start
            break
    last_sec = None
    for m in _SECTION.finditer(text, max(0, pos - 400), pos):
        last_sec = m
    if last_sec and last_sec.start() > env_start:
        return _SECTION_KIND[last_sec.group(1)]
    return env_kind


def collect_labels(sources: list[SourceFile]) -> list[Label]:
    labels = []
    for f in sources:
        for m in _LABEL.finditer(f.text):
            labels.append(
                Label(
                    name=m.group(1).strip(),
                    kind=_kind_at(f.text, m.start()),
                    file=f.rel,
                    line=line_of(f.text, m.start()),
                )
            )
    return labels


def label_conventions(labels: list[Label]) -> LabelConventions:
    parsed = [(lab, split_prefix(lab.name)) for lab in labels]
    parsed = [(lab, ps) for lab, ps in parsed if ps]
    if not parsed:
        return LabelConventions()
    seps = Counter(sep for _, (_, sep) in parsed)
    counts: Counter[str] = Counter(prefix for _, (prefix, _) in parsed)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for lab, (prefix, _) in parsed:
        if lab.kind != "other":
            by_kind[lab.kind][prefix] += 1
    prefix_by_kind = {k: c.most_common(1)[0][0] for k, c in by_kind.items()}
    examples = {}
    for lab, (prefix, _) in parsed:
        if prefix_by_kind.get(lab.kind) == prefix:
            examples.setdefault(lab.kind, lab.name)
    return LabelConventions(
        separator=seps.most_common(1)[0][0],
        prefix_counts=dict(counts.most_common()),
        prefix_by_kind=prefix_by_kind,
        example_by_kind=examples,
    )
