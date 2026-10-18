"""Human-readable summary of a ProjectIndex (used by `voxtex index` and at startup)."""

from collections import Counter

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from voxtex.project.index import ProjectIndex

KEY_PACKAGES = (
    "biblatex", "natbib", "cleveref", "booktabs", "siunitx", "amsmath",
    "babel", "polyglossia", "hyperref", "graphicx", "subcaption", "listings", "minted",
)  # fmt: skip


def summary_lines(idx: ProjectIndex) -> list[str]:
    pre, conv = idx.preamble, idx.conventions
    pkgs = []
    for name in KEY_PACKAGES:
        p = pre.package(name)
        if p is None:
            continue
        if name == "biblatex":
            pkgs.append(f"biblatex ({conv.bib_style})")
        else:
            pkgs.append(name)
    if conv.bib_package == "bibtex":
        pkgs.insert(0, f"bibtex ({conv.bib_style})")
    used = sum(1 for m in pre.macros if m.uses)
    cls = f"{pre.documentclass}" + (f"[{','.join(pre.class_options)}]" if pre.class_options else "")
    lines = [
        f"{idx.main} — {cls}; "
        + ", ".join(pkgs + [f"{len(pre.macros)} macros ({used} used in chapters)"]),
        f"{len(idx.bib)} bib entries · {len(idx.labels)} labels"
        + (f" ({conv.labels.summary})" if conv.labels.prefix_counts else ""),
    ]
    if idx.style.sources:
        lines.append("Style learned from " + ", ".join(idx.style.sources))
    return lines


def print_index(console: Console, idx: ProjectIndex, cached: bool, verbose: bool = False) -> None:
    for line in summary_lines(idx):
        console.print(f"[green]✓[/green] {escape(line)}")
    for miss in idx.missing:
        console.print(f"[yellow]![/yellow] referenced but not found: {escape(miss)}")
    if not idx.bib_files:
        console.print("[yellow]![/yellow] no .bib file found (citations can't be validated)")
    if not idx.style.paragraphs:
        console.print(
            "[yellow]![/yellow] no prose paragraphs found to learn the writing style from"
        )

    console.print()
    console.print("[bold]Rules[/bold]")
    for r in idx.conventions.rules():
        console.print(f"  {escape(r)}")
    langs = idx.conventions.languages
    console.print(
        f"  [dim](language from {idx.conventions.language_source}"
        + (f": {', '.join(langs)}" if langs else "")
        + ")[/dim]"
    )

    if verbose:
        _print_details(console, idx)

    console.print()
    console.print(
        f"[dim]{len(idx.body_files)} content files, {len(idx.preamble_files)} preamble files"
        f" · {'cached' if cached else 'rebuilt'} index in {idx.root}/.voxtex/[/dim]"
    )


def _print_details(console: Console, idx: ProjectIndex) -> None:
    console.print()
    t = Table("package", "options", "from", title="Packages", title_justify="left", box=None)
    for p in idx.preamble.packages:
        t.add_row(p.name, escape(", ".join(p.options)), p.source)
    console.print(t)

    console.print()
    t = Table(
        "macro", "args", "uses", "kind", "from", title="Macros", title_justify="left", box=None
    )
    for m in sorted(idx.preamble.macros, key=lambda m: -m.uses):
        args = str(m.arity) + (f" ({m.optional} opt)" if m.optional else "")
        kind = m.kind + (" · math" if m.math else "")
        t.add_row(escape(m.name), args, str(m.uses or ""), kind, m.source)
    console.print(t)
    if idx.preamble.environments:
        envs = ", ".join(e.name for e in idx.preamble.environments)
        console.print(f"[bold]Environments:[/bold] {escape(envs)}")

    console.print()
    kinds = Counter(lab.kind for lab in idx.labels)
    console.print("[bold]Labels by kind:[/bold] " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
    for lab in idx.labels:
        console.print(f"  {escape(lab.name):30} {lab.kind:9} {lab.file}:{lab.line}")

    console.print()
    console.print(f"[bold]Bib[/bold] ({', '.join(idx.bib_files)})")
    for e in idx.bib[:10]:
        console.print(f"  {escape(e.short)}")
    if len(idx.bib) > 10:
        console.print(f"  … and {len(idx.bib) - 10} more")

    for s in [*idx.style.paragraphs, idx.style.table, idx.style.figure]:
        if s:
            console.print()
            console.print(f"[bold]Sample {s.kind}[/bold] [dim]({s.file})[/dim]")
            console.print(escape(s.text), style="dim", highlight=False)
