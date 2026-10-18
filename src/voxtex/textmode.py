"""`voxtex text`: the engine driven by typed lines instead of the microphone."""

import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax

from voxtex.app import build_chain, choose_target, location_in, offer_gitignore, open_project
from voxtex.engine.decide import Delete, Ignore, Replace, UncertainInsert, Wait
from voxtex.engine.engine import Engine, Step
from voxtex.io.marker import fallback_location
from voxtex.io.writer import Writer
from voxtex.providers import ChainExhausted
from voxtex.session import Session


def print_step(console: Console, step: Step, show_prompt: bool) -> None:
    if show_prompt:
        for m in step.messages:
            console.rule(f"[dim]{m['role']}[/dim]")
            console.print(escape(m["content"]), style="dim", highlight=False)
        console.rule()
    for n in step.notices:
        console.print(f"[cyan]{escape(n)}[/cyan]")
    if step.retried:
        console.print(
            "[magenta]⟲ fixed after validation:[/magenta] " + escape("; ".join(step.first_errors))
        )
    d, b = step.decision, step.block
    match d:
        case Wait():
            console.print("[dim]… waiting for more (empty line to force a decision)[/dim]")
        case Ignore():
            console.print("[dim]· ignored (not content)[/dim]")
        case Delete():
            console.print(f"[red]✗ deleted {b.id}[/red]" if b else "[red]✗ deleted[/red]")
        case _:
            verb = "↻ replaced" if isinstance(d, Replace) else "+ inserted"
            color = "yellow" if isinstance(d, UncertainInsert) else "green"
            console.print(f"[{color}]{verb} {b.id if b else ''}[/{color}]")
            console.print(Syntax(b.latex if b else "", "latex", word_wrap=True, theme="ansi_dark"))
            if b and b.note:
                console.print(f"[yellow]% voxtex-check: {escape(b.note)}[/yellow]")
    r = step.report
    if r:
        for w in r.warnings:
            console.print(f"[yellow]! {escape(w.message)}[/yellow]")
    checks = ""
    if r:
        checks = f" · checked {r.seconds:.1f}s" + (" (compiled)" if r.compiled else "")
    console.print(
        f"[dim]{step.provider} · {step.input_tokens} in / {step.output_tokens} out tokens"
        f" · {step.latency_s:.1f}s{checks}[/dim]"
    )


def run_text_mode(
    console: Console,
    path: Path,
    private: bool,
    provider: str | None,
    show_prompt: bool,
    file_opt: Path | None = None,
    dry_run: bool = False,
) -> int:
    interactive = sys.stdin.isatty()
    opened = open_project(console, path)
    if opened is None:
        return 1
    idx, config = opened
    chain = build_chain(console, idx, config, private, provider)
    if chain is None:
        return 1
    root = idx.root_path

    session = None
    if dry_run:
        target = file_opt and (root / file_opt)
        location = location_in(root, target) if target else fallback_location(root, idx.body_files)
    else:
        target = choose_target(console, idx, file_opt, interactive)
        if target is None:
            return 1
        offer_gitignore(console, root, interactive)
        writer = Writer(root)
        writer.ensure_marker(target)
        location = location_in(root, target)
    engine = Engine(idx, chain, config, location)
    if not dry_run:
        session = Session(engine, writer, target)

    lang = idx.conventions.language if config.language == "auto" else config.language
    where = f"{location.rel} at %> HERE" if location and location.found else "(no target)"
    console.print(f"⌨  Typing → {where}   (style: {config.style}, lang: {lang})")
    note = "dry run: nothing is written" if dry_run else "voxtex undo reverts writes"
    console.print(f"[dim]   Empty line = pause · {note}[/dim]")

    def step(line: str, force: bool = False) -> None:
        try:
            s = engine.feed(line, force=force) if line else engine.flush()
        except ChainExhausted as e:
            console.print(f"[red]✗ {escape(str(e))}[/red] (speech kept; try again)")
            return
        if not s:
            return
        print_step(console, s, show_prompt)
        if session:
            try:
                for n in session.write(s):
                    console.print(f"[cyan]{escape(n)}[/cyan]")
            except OSError as e:
                console.print(f"[red]✗ could not write: {escape(str(e))}[/red]")

    try:
        while True:
            try:
                line = console.input("[bold]› [/bold]") if interactive else input()
            except EOFError:
                break
            step(line.strip())
        step("")  # flush whatever is pending at the end
    except KeyboardInterrupt:
        console.print()

    t = engine.chain.totals
    if t.calls:
        console.print(
            f"[dim]Session: {t.calls} calls · {t.input_tokens} in / {t.output_tokens} out tokens"
            f" (avg {t.input_tokens // t.calls} in/call)[/dim]"
        )
    return 0
