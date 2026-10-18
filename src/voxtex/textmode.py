"""`voxtex text`: the engine driven by typed lines instead of the microphone."""

import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax

from voxtex.config import load_config
from voxtex.engine.decide import Delete, Ignore, Replace, UncertainInsert, Wait
from voxtex.engine.engine import Engine, Step
from voxtex.io.marker import fallback_location, find_marker
from voxtex.project.discover import DiscoveryError
from voxtex.project.index import STATE_DIR, load_index
from voxtex.project.report import summary_lines
from voxtex.providers import Chain, ChainExhausted, QuotaStore


def build_engine(
    console: Console, path: Path, private: bool, provider: str | None
) -> Engine | None:
    try:
        idx, _ = load_index(path)
    except DiscoveryError as e:
        console.print(f"[red]✗[/red] {e}")
        return None
    root = idx.root_path
    config = load_config(root)
    providers = config.chain_providers(private_only=private)
    if provider:
        providers = [p for p in config.providers.values() if p.name == provider]
        if private:
            providers = [p for p in providers if p.private]
        if not providers:
            console.print(f"[red]✗[/red] no provider named {provider!r} (or it isn't private)")
            return None
    chain = Chain(
        providers,
        default_timeout=config.timeout,
        quota=QuotaStore(root / STATE_DIR / "quota.json"),
        log_path=root / STATE_DIR / "timings.log",
    )
    if not chain.usable():
        names = ", ".join(p.name for p in providers) or "none"
        console.print(f"[red]✗[/red] no usable AI provider (chain: {names}).")
        console.print(
            "  Set an API key, e.g. [bold]export GROQ_API_KEY=...[/bold], or start Ollama."
        )
        return None

    location = find_marker(root, idx.body_files) or fallback_location(root, idx.body_files)
    for line in summary_lines(idx):
        console.print(f"[green]✓[/green] {escape(line)}")
    chain_names = " → ".join(p.name for p in chain.usable())
    console.print(f"[green]✓[/green] AI: {chain_names}" + (" (private)" if private else ""))
    if location and location.found:
        where = f"{location.rel} at %> HERE (line {location.line})"
    elif location:
        where = f"end of {location.rel} [yellow](no %> HERE marker found)[/yellow]"
    else:
        where = "[yellow]no target file[/yellow]"
    lang = idx.conventions.language if config.language == "auto" else config.language
    console.print(f"⌨  Typing → {where}   (style: {config.style}, lang: {lang})")
    console.print("[dim]   Nothing is written to files yet (M2). Empty line = pause.[/dim]")
    return Engine(idx, chain, config, location)


def print_step(console: Console, step: Step, show_prompt: bool) -> None:
    if show_prompt:
        for m in step.messages:
            console.rule(f"[dim]{m['role']}[/dim]")
            console.print(escape(m["content"]), style="dim", highlight=False)
        console.rule()
    for n in step.notices:
        console.print(f"[cyan]{escape(n)}[/cyan]")
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
    console.print(
        f"[dim]{step.provider} · {step.input_tokens} in / {step.output_tokens} out tokens"
        f" · {step.latency_s:.1f}s[/dim]"
    )


def run_text_mode(
    console: Console, path: Path, private: bool, provider: str | None, show_prompt: bool
) -> int:
    engine = build_engine(console, path, private, provider)
    if engine is None:
        return 1
    interactive = sys.stdin.isatty()

    def step(line: str, force: bool = False) -> None:
        try:
            s = engine.feed(line, force=force) if line else engine.flush()
        except ChainExhausted as e:
            console.print(f"[red]✗ {escape(str(e))}[/red] (speech kept; try again)")
            return
        if s:
            print_step(console, s, show_prompt)

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
