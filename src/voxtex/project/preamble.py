"""Extract packages (with options) and user-defined macros from preamble sources."""

import re

from pydantic import BaseModel

from voxtex.project.discover import SourceFile
from voxtex.project.latex import (
    find_commands,
    read_group,
    read_name_arg,
    read_optional,
    skip_ws,
    split_csv,
)


class Package(BaseModel):
    name: str
    options: list[str] = []
    source: str = ""

    def option(self, key: str) -> str | None:
        """Value of `key=value` option, or None."""
        for o in self.options:
            k, _, v = o.partition("=")
            if k.strip() == key:
                return v.strip()
        return None


class Macro(BaseModel):
    name: str  # including the backslash, e.g. "\\legalbert"
    arity: int = 0
    optional: int = 0  # how many of `arity` are optional
    kind: str = "newcommand"
    math: bool = False
    source: str = ""
    body: str = ""  # definition, truncated
    uses: int = 0  # occurrences in the chapters (filled in by the index)


class Environment(BaseModel):
    name: str
    arity: int = 0
    source: str = ""


class Preamble(BaseModel):
    documentclass: str = ""
    class_options: list[str] = []
    packages: list[Package] = []
    macros: list[Macro] = []
    environments: list[Environment] = []

    def package(self, name: str) -> Package | None:
        return next((p for p in self.packages if p.name == name), None)

    def has(self, name: str) -> bool:
        return self.package(name) is not None


_NEWCMD = "newcommand|renewcommand|providecommand|DeclareRobustCommand|newrobustcmd|renewrobustcmd"
_XPARSE = "NewDocumentCommand|RenewDocumentCommand|ProvideDocumentCommand|DeclareDocumentCommand"
_NEWENV = "newenvironment|renewenvironment|NewDocumentEnvironment|RenewDocumentEnvironment"
_DEF = re.compile(r"\\(?:g|e|x)?def\s*(\\[A-Za-z@]+)((?:#\d)*)\s*\{")
_LET = re.compile(r"\\let\s*(\\[A-Za-z@]+)\s*=?\s*\\[A-Za-z@]+")

# xparse argument specifiers -> how many brace groups / tokens follow the letter.
_SPEC_GROUPS = {"O": 1, "G": 1, "e": 1, "E": 2}
_SPEC_TOKENS = {"d": 2, "D": 2, "r": 2, "R": 2, "t": 1}
_SPEC_OPTIONAL = set("oOdDsteEgG")
BODY_MAX = 200


def xparse_arity(spec: str) -> tuple[int, int]:
    """Count (arguments, optional arguments) in an xparse argument spec like `s m O{x}`."""
    n = opt = 0
    i = 0
    while i < len(spec):
        c = spec[i]
        i += 1
        if c.isspace() or c in "+!>=":
            if c in ">=":  # processor: skip its brace group
                g = read_group(spec, i)
                if g:
                    i = g[1]
            continue
        if not c.isalpha():
            continue
        n += 1
        if c in _SPEC_OPTIONAL:
            opt += 1
        for _ in range(_SPEC_TOKENS.get(c, 0)):
            i = skip_ws(spec, i) + 1
        if c in ("D", "R"):
            g = read_group(spec, i)
            if g:
                i = g[1]
        for _ in range(_SPEC_GROUPS.get(c, 0)):
            g = read_group(spec, i)
            if g:
                i = g[1]
    return n, opt


def _is_math(body: str) -> bool:
    """Math-mode macro: uses math constructs and doesn't switch into math itself."""
    if "$" in body or "\\(" in body or "\\text" in body:
        return False
    return bool(re.search(r"\\(math[a-z]+|operatorname|boldsymbol)|\^|_", body))


def parse_packages(sources: list[SourceFile]) -> tuple[str, list[str], list[Package]]:
    docclass, class_opts = "", []
    packages: list[Package] = []
    for f in sources:
        for m in find_commands(f.text, "documentclass|usepackage|RequirePackage"):
            pos = m.end()
            opts: list[str] = []
            o = read_optional(f.text, pos)
            if o:
                opts, pos = split_csv(o[0]), o[1]
            g = read_group(f.text, pos)
            if not g:
                continue
            if m.group(1) == "documentclass":
                if not docclass:
                    docclass, class_opts = g[0].strip(), opts
                continue
            for name in split_csv(g[0]):
                existing = next((p for p in packages if p.name == name), None)
                if existing:
                    existing.options += [o for o in opts if o not in existing.options]
                else:
                    packages.append(Package(name=name, options=opts, source=f.rel))
    return docclass, class_opts, packages


def parse_macros(sources: list[SourceFile]) -> tuple[list[Macro], list[Environment]]:
    macros: dict[str, Macro] = {}
    envs: dict[str, Environment] = {}

    def add(m: Macro) -> None:
        if m.name.startswith("\\") and "@" not in m.name:  # skip internal @-macros
            macros[m.name] = m

    for f in sources:
        t, src = f.text, f.rel

        for m in find_commands(t, _NEWCMD):
            na = read_name_arg(t, m.end())
            if not na:
                continue
            name, pos = na
            arity = optional = 0
            o = read_optional(t, pos)
            if o and o[0].strip().isdigit():
                arity, pos = int(o[0]), o[1]
                d = read_optional(t, pos)
                if d:
                    optional, pos = 1, d[1]
            body = read_group(t, pos)
            add(
                Macro(
                    name=name,
                    arity=arity,
                    optional=optional,
                    kind=m.group(1),
                    math=_is_math(body[0]) if body else False,
                    source=src,
                    body=body[0][:BODY_MAX] if body else "",
                )
            )

        for m in find_commands(t, "DeclareMathOperator"):
            na = read_name_arg(t, m.end())
            if na:
                add(Macro(name=na[0], kind="DeclareMathOperator", math=True, source=src))

        for m in find_commands(t, _XPARSE):
            na = read_name_arg(t, m.end())
            spec = read_group(t, na[1]) if na else None
            if not (na and spec):
                continue
            n, opt = xparse_arity(spec[0])
            body = read_group(t, spec[1])
            add(
                Macro(
                    name=na[0],
                    arity=n,
                    optional=opt,
                    kind=m.group(1),
                    math=_is_math(body[0]) if body else False,
                    source=src,
                    body=body[0][:BODY_MAX] if body else "",
                )
            )

        for m in _DEF.finditer(t):
            body = read_group(t, m.end() - 1)
            add(
                Macro(
                    name=m.group(1),
                    arity=len(m.group(2)) // 2,
                    kind="def",
                    source=src,
                    body=body[0][:BODY_MAX] if body else "",
                )
            )

        for m in _LET.finditer(t):
            add(Macro(name=m.group(1), kind="let", source=src))

        for m in find_commands(t, _NEWENV):
            g = read_group(t, m.end())
            if not g:
                continue
            name, pos, arity = g[0].strip(), g[1], 0
            if m.group(1).endswith("DocumentEnvironment"):
                spec = read_group(t, pos)
                if spec:
                    arity = xparse_arity(spec[0])[0]
            else:
                o = read_optional(t, pos)
                if o and o[0].strip().isdigit():
                    arity = int(o[0])
            envs[name] = Environment(name=name, arity=arity, source=src)

        # \newtheorem{name}{Title} also defines an environment.
        for m in find_commands(t, "newtheorem"):
            g = read_group(t, m.end())
            if g:
                envs[g[0].strip()] = Environment(name=g[0].strip(), source=src)

    return list(macros.values()), list(envs.values())


def parse_preamble(sources: list[SourceFile]) -> Preamble:
    docclass, class_opts, packages = parse_packages(sources)
    macros, envs = parse_macros(sources)
    return Preamble(
        documentclass=docclass,
        class_options=class_opts,
        packages=packages,
        macros=macros,
        environments=envs,
    )


def count_macro_uses(macros: list[Macro], text: list[SourceFile]) -> None:
    """Set `uses` on each macro: how often it appears in `text` (outside its definition)."""
    if not macros:
        return
    names = "|".join(re.escape(m.name[1:]) for m in macros if m.name[1:].isalpha())
    if not names:
        return
    pattern = re.compile(rf"\\({names})(?![A-Za-z@])")
    counts: dict[str, int] = {}
    for f in text:
        for name in pattern.findall(f.text):
            counts[name] = counts.get(name, 0) + 1
    for m in macros:
        m.uses = counts.get(m.name[1:], 0)
