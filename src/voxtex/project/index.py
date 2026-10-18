"""Build the ProjectIndex and cache it in `.voxtex/index.json`.

The cache stores (mtime, size) for every tracked file and is rebuilt as a
whole when any of them changes. Building is cheap (tens of milliseconds for a
thesis), so per-file incremental rebuilds aren't worth the complexity yet.
"""

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ValidationError

from voxtex.project.bib import BibEntry, parse_bibs
from voxtex.project.conventions import Conventions, derive_conventions
from voxtex.project.discover import _rel, discover, find_main
from voxtex.project.labels import Label, collect_labels
from voxtex.project.preamble import Preamble, count_macro_uses, parse_macros, parse_preamble
from voxtex.project.style import StyleSamples, pick_samples

INDEX_VERSION = 4
STATE_DIR = ".voxtex"


class FileStamp(BaseModel):
    mtime_ns: int
    size: int


class ProjectIndex(BaseModel):
    version: int = INDEX_VERSION
    root: str
    main: str  # relative to root
    preamble_files: list[str] = []
    body_files: list[str] = []
    bib_files: list[str] = []
    missing: list[str] = []
    preamble: Preamble
    conventions: Conventions
    bib: list[BibEntry] = []
    labels: list[Label] = []
    style: StyleSamples
    known_commands: list[str] = []  # every \command and environment name used in the project
    latin_terms: list[str] = []  # most used terms wrapped in \en{...} etc. (Whisper hint)
    stamps: dict[str, FileStamp] = {}

    @property
    def root_path(self) -> Path:
        return Path(self.root)


def _stamp(p: Path) -> FileStamp:
    st = p.stat()
    return FileStamp(mtime_ns=st.st_mtime_ns, size=st.st_size)


def build_index(path: Path) -> ProjectIndex:
    pf = discover(path)
    root = pf.root
    preamble = parse_preamble(pf.preamble)
    # Chapters sometimes define their own macros too.
    macros, envs = parse_macros(pf.preamble + pf.body)
    preamble.macros, preamble.environments = macros, envs
    chapters = [f for f in pf.body if f.path != pf.main] or pf.body
    count_macro_uses(macros, chapters)
    labels = collect_labels(pf.body)
    conventions = derive_conventions(preamble, pf.preamble, pf.body, labels)
    style = pick_samples(chapters)
    return ProjectIndex(
        root=str(root),
        main=_rel(pf.main, root),
        preamble_files=[_rel(f.path, root) for f in pf.preamble],
        body_files=[_rel(f.path, root) for f in pf.body],
        bib_files=[_rel(b, root) for b in pf.bib_files],
        missing=pf.missing,
        preamble=preamble,
        conventions=conventions,
        bib=parse_bibs(pf.bib_files),
        labels=labels,
        style=style,
        known_commands=sorted(_commands_used(pf.preamble + pf.body)),
        latin_terms=_wrapped_terms(chapters, conventions.latin_wrapper),
        stamps={_rel(p, root): _stamp(p) for p in pf.tracked_paths},
    )


def _commands_used(sources) -> set[str]:
    names: set[str] = set()
    for f in sources:
        names.update(re.findall(r"\\([A-Za-z@]+)", f.text))
        names.update("env:" + e for e in re.findall(r"\\begin\s*\{([^}]+)\}", f.text))
    return names


def _wrapped_terms(sources, wrapper: str, limit: int = 60) -> list[str]:
    """Most frequent arguments of the Latin-term wrapper, e.g. RDF, SeRQL, Sesame."""
    if not wrapper:
        return []
    counts: Counter[str] = Counter()
    for f in sources:
        for m in re.finditer(re.escape(wrapper) + r"\s*\{([^{}\\]{2,40})\}", f.text):
            term = " ".join(m.group(1).split()).strip(" ().,")
            if _is_spoken_term(term):
                counts[term] += 1
    out: dict[str, str] = {}
    for t, _ in counts.most_common():
        out.setdefault(t.lower(), t)  # one spelling per term
    return list(out.values())[:limit]


def _is_spoken_term(term: str) -> bool:
    """A term someone would say aloud: not a URL, path, code identifier or long phrase."""
    if not term or len(term) > 25 or len(term.split()) > 3 or any(c in term for c in "/:#=_<>"):
        return False
    if any(c.isdigit() for c in term):
        return False
    # camelCase identifiers (classesToWrite) are code; acronyms (RDF, SeRQL) are fine
    return not re.search(r"^[a-z]+[A-Z]", term)


def _cache_path(root: Path) -> Path:
    return root / STATE_DIR / "index.json"


def _load_cached(root: Path) -> ProjectIndex | None:
    cp = _cache_path(root)
    try:
        idx = ProjectIndex.model_validate_json(cp.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None
    if idx.version != INDEX_VERSION or Path(idx.root) != root:
        return None
    for rel, stamp in idx.stamps.items():
        p = root / rel
        try:
            if _stamp(p) != stamp:
                return None
        except OSError:
            return None
    return idx


def _save(idx: ProjectIndex) -> None:
    """Atomic write; silently skipped if the project folder is read-only."""
    cp = _cache_path(idx.root_path)
    try:
        cp.parent.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cp.parent, prefix=".index-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(idx.model_dump(mode="json"), fh, ensure_ascii=False, indent=1)
        os.replace(tmp, cp)
    except OSError:
        pass


def load_index(path: Path, rebuild: bool = False) -> tuple[ProjectIndex, bool]:
    """Return (index, from_cache)."""
    root = find_main(path).parent
    if not rebuild:
        cached = _load_cached(root)
        if cached is not None:
            return cached, True
    idx = build_index(path)
    _save(idx)
    return idx, False
