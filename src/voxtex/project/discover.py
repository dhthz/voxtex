"""Find the main .tex file and every file the project pulls in."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from voxtex.project.latex import (
    find_commands,
    read_group,
    read_optional,
    read_text,
    split_csv,
    strip_comments,
)

SKIP_DIRS = {".git", ".voxtex", "node_modules", "_minted", "build", "out"}
MAIN_NAMES = {"main", "thesis", "master", "dissertation", "report", "paper"}
NON_MAIN_CLASSES = {"subfiles", "standalone"}
MAX_DEPTH = 3

_DOCCLASS = re.compile(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_BEGIN_DOC = re.compile(r"\\begin\s*\{document\}")
_BARE_INPUT = re.compile(r"\\input\s+([^\s{\\]+)")


class DiscoveryError(Exception):
    pass


@dataclass
class SourceFile:
    path: Path
    text: str  # comment-stripped
    rel: str = ""  # path relative to the project root, for display

    def __post_init__(self) -> None:
        self.rel = self.rel or self.path.name


@dataclass
class ProjectFiles:
    root: Path
    main: Path
    class_name: str
    preamble: list[SourceFile] = field(default_factory=list)  # main preamble, inputs, local cls/sty
    body: list[SourceFile] = field(default_factory=list)  # main body + included chapters
    bib_files: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def tracked_paths(self) -> list[Path]:
        """Every file whose change should invalidate the index."""
        seen: dict[Path, None] = {}
        for f in [*self.preamble, *self.body]:
            seen[f.path] = None
        for b in self.bib_files:
            seen[b] = None
        return list(seen)


def _iter_tex(root: Path, depth: int = 0):
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for p in entries:
        if p.is_dir():
            if depth < MAX_DEPTH and p.name not in SKIP_DIRS and not p.name.startswith("."):
                yield from _iter_tex(p, depth + 1)
        elif p.suffix == ".tex":
            yield p


def find_main(path: Path) -> Path:
    """Locate the main file: the one with \\documentclass and \\begin{document}."""
    path = path.resolve()
    if path.is_file():
        return path
    candidates: list[tuple[tuple, Path]] = []
    for p in _iter_tex(path):
        text = strip_comments(read_text(p))
        m = _DOCCLASS.search(text)
        if not m or not _BEGIN_DOC.search(text) or m.group(1).strip() in NON_MAIN_CLASSES:
            continue
        n_includes = len(find_commands(text, "input|include|subfile|import|subimport"))
        # Prefer: root level, conventional name, more includes, shorter path.
        score = (p.parent == path, p.stem.lower() in MAIN_NAMES, n_includes, -len(p.parts))
        candidates.append((score, p))
    if not candidates:
        raise DiscoveryError(f"No main .tex file (with \\documentclass) found under {path}")
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _resolve(base: Path, name: str, exts: tuple[str, ...] = (".tex",)) -> Path | None:
    name = name.strip().strip('"')
    if not name:
        return None
    p = base / name
    if p.suffix in exts and p.is_file():
        return p
    for ext in exts:
        q = p.with_name(p.name + ext)
        if q.is_file():
            return q
    return p if p.is_file() else None


@dataclass
class _Ref:
    pos: int
    target: Path | None
    raw: str


def _include_refs(text: str, root: Path, here: Path) -> list[_Ref]:
    """All \\input-like references in `text`, in order of appearance."""
    refs: list[_Ref] = []
    for m in find_commands(text, "input|include|subfile|import|subimport|includeonly"):
        cmd = m.group(1)
        if cmd == "includeonly":
            continue
        if cmd in ("import", "subimport"):
            d = read_group(text, m.end())
            f = read_group(text, d[1]) if d else None
            if not (d and f):
                continue
            base = (here if cmd == "subimport" else root) / d[0].strip()
            refs.append(_Ref(m.start(), _resolve(base, f[0]), f"{d[0]}{f[0]}"))
            continue
        g = read_group(text, m.end())
        if g is None:
            if cmd == "input":
                bare = _BARE_INPUT.match(text, m.start())
                if bare:
                    refs.append(_Ref(m.start(), _resolve(root, bare.group(1)), bare.group(1)))
            continue
        base = here if cmd == "subfile" else root
        refs.append(_Ref(m.start(), _resolve(base, g[0]), g[0]))
    return refs


def _subfile_body(text: str) -> str:
    """A subfile is a full document; only its body counts as content."""
    m = _DOCCLASS.search(text)
    if m and m.group(1).strip() == "subfiles":
        b = _BEGIN_DOC.search(text)
        if b:
            return text[b.end() :].split("\\end{document}")[0]
    return text


def _local_style_files(text: str, root: Path) -> list[Path]:
    """Local .cls/.sty files loaded by \\documentclass / \\usepackage / \\RequirePackage."""
    found: list[Path] = []
    for m in find_commands(text, "documentclass|LoadClass|usepackage|RequirePackage"):
        pos = m.end()
        opt = read_optional(text, pos)
        if opt:
            pos = opt[1]
        g = read_group(text, pos)
        if not g:
            continue
        ext = ".cls" if m.group(1) in ("documentclass", "LoadClass") else ".sty"
        for name in split_csv(g[0]):
            p = _resolve(root, name, (ext,))
            if p:
                found.append(p)
    return found


def discover(path: Path) -> ProjectFiles:
    main = find_main(path)
    root = main.parent
    main_text = strip_comments(read_text(main))
    m = _DOCCLASS.search(main_text)
    class_name = m.group(1).strip() if m else ""

    b = _BEGIN_DOC.search(main_text)
    split_at = b.start() if b else len(main_text)
    pf = ProjectFiles(root=root, main=main, class_name=class_name)
    visited: set[Path] = {main}

    def follow(text: str, here: Path, into: list[SourceFile]) -> None:
        for ref in _include_refs(text, root, here):
            if "#" in ref.raw:  # inside a macro definition, not a real include
                continue
            if ref.target is None:
                pf.missing.append(ref.raw)
                continue
            if ref.target in visited:
                continue
            visited.add(ref.target)
            sub = _subfile_body(strip_comments(read_text(ref.target)))
            into.append(SourceFile(ref.target, sub, _rel(ref.target, root)))
            follow(sub, ref.target.parent, into)

    preamble_text = main_text[:split_at]
    pf.preamble.append(SourceFile(main, preamble_text, main.name))
    follow(preamble_text, root, pf.preamble)

    # Pad with newlines so line numbers in the body still match the file.
    body_text = "\n" * preamble_text.count("\n") + main_text[split_at:]
    pf.body.append(SourceFile(main, body_text, main.name))
    follow(body_text, root, pf.body)

    # Local class / style files (thesis templates often define macros there).
    queue = [f.text for f in pf.preamble]
    while queue:
        for p in _local_style_files(queue.pop(), root):
            if p not in visited:
                visited.add(p)
                text = strip_comments(read_text(p))
                pf.preamble.append(SourceFile(p, text, _rel(p, root)))
                queue.append(text)

    pf.bib_files = _find_bibs(pf, root)
    return pf


def _find_bibs(pf: ProjectFiles, root: Path) -> list[Path]:
    bibs: list[Path] = []
    for f in [*pf.preamble, *pf.body]:
        for m in find_commands(f.text, "addbibresource|bibliography|addglobalbib"):
            pos = m.end()
            opt = read_optional(f.text, pos)
            if opt:
                pos = opt[1]
            g = read_group(f.text, pos)
            if not g:
                continue
            for name in split_csv(g[0]):
                p = _resolve(root, name, (".bib",))
                if p is None:
                    pf.missing.append(name if name.endswith(".bib") else f"{name}.bib")
                elif p not in bibs:
                    bibs.append(p)
    return bibs
