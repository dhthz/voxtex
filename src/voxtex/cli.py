"""Command-line interface. Every command is a stub until its milestone lands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from voxtex import __version__

app = typer.Typer(
    help="Talk into your LaTeX project. voxtex learns your setup, listens, and writes LaTeX.",
    no_args_is_help=False,
    add_completion=False,
)
console = Console()

ProjectPath = Annotated[
    Path, typer.Argument(help="LaTeX project folder (default: current directory).")
]


def _not_yet(milestone: str) -> None:
    console.print(f"[yellow]Not implemented yet[/yellow] (arrives in {milestone}).")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"voxtex {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """With no command, `voxtex` starts listening in the current directory."""
    if ctx.invoked_subcommand is None:
        listen(Path("."), False, None, None, False, None, None, False)


@app.command()
def listen(
    path: ProjectPath = Path("."),
    private: Annotated[
        bool, typer.Option("--private", help="Keep everything local: local AI and speech-to-text.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Use only this AI provider from the config.")
    ] = None,
    file: Annotated[
        Path | None, typer.Option("--file", help="File to write into (if there's no marker).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be written; change nothing.")
    ] = False,
    stt: Annotated[
        str | None, typer.Option("--stt", help="Speech-to-text: auto, local, groq or openai.")
    ] = None,
    mic: Annotated[
        str | None, typer.Option("--mic", help="Microphone name or index (see --list-mics).")
    ] = None,
    list_mics: Annotated[
        bool, typer.Option("--list-mics", help="List microphones and exit.")
    ] = False,
) -> None:
    """Index the project, listen to the mic, and write LaTeX at the %> HERE marker."""
    if list_mics:
        import subprocess

        from voxtex.audio.capture import list_inputs, prefer_parec

        if prefer_parec():
            out = subprocess.run(
                ["pactl", "list", "sources", "short"], capture_output=True, text=True
            ).stdout
            console.print("PulseAudio sources (WSL; use the name with --mic):")
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) > 1 and not parts[1].endswith(".monitor"):
                    console.print(f"     {parts[1]}")
            return
        try:
            for i, name in list_inputs():
                console.print(f"{i:3}  {name}")
        except OSError as e:
            console.print(f"[red]✗[/red] audio system unavailable: {e}")
            raise typer.Exit(1) from e
        return
    from voxtex.listenmode import run_listen

    raise typer.Exit(run_listen(console, path, private, provider, file, dry_run, stt, mic))


@app.command()
def text(
    path: ProjectPath = Path("."),
    private: Annotated[
        bool, typer.Option("--private", help="Only use private (local) providers.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Use only this provider from the config.")
    ] = None,
    show_prompt: Annotated[
        bool, typer.Option("--show-prompt", help="Print the messages sent to the model.")
    ] = False,
    file: Annotated[
        Path | None, typer.Option("--file", help="File to write into (if there's no marker).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be written; change nothing.")
    ] = False,
) -> None:
    """Same engine as `listen`, but reads typed lines from stdin.

    Each line is an utterance; an empty line means "I paused" and forces a decision.
    """
    from voxtex.textmode import run_text_mode

    raise typer.Exit(run_text_mode(console, path, private, provider, show_prompt, file, dry_run))


@app.command()
def watch(
    path: ProjectPath = Path("."),
    private: Annotated[
        bool, typer.Option("--private", help="Only use private (local) providers.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Use only this provider from the config.")
    ] = None,
    poll: Annotated[
        bool | None,
        typer.Option(
            "--poll/--no-poll",
            help="Poll for changes instead of file events (default: auto, on for /mnt/ drives).",
        ),
    ] = None,
) -> None:
    """Convert %> lines typed by any dictation app into LaTeX on save."""
    from voxtex.watchmode import run_watch

    raise typer.Exit(run_watch(console, path, private, provider, poll))


@app.command()
def init() -> None:
    """First-run wizard: backend, API key, microphone."""
    _not_yet("M6")


@app.command()
def doctor() -> None:
    """Check mic, backend, models, TeX and project detection."""
    _not_yet("M6")


@app.command()
def undo(
    n: Annotated[int, typer.Argument(min=1, help="Number of writes to revert.")] = 1,
    path: Annotated[Path, typer.Option("--path", help="Project folder.")] = Path("."),
) -> None:
    """Revert the last N writes (newest first), even from earlier sessions."""
    from voxtex.io.writer import Writer
    from voxtex.project.discover import DiscoveryError, find_main

    try:
        root = find_main(path).parent
    except DiscoveryError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e
    writer = Writer(root)
    if not writer.history.pending():
        console.print("Nothing to undo.")
        return
    done, problem = writer.undo(n)
    for e in done:
        what = {"insert": "removed", "replace": "restored", "delete": "restored"}.get(
            e.kind, "reverted"
        )
        label = f" {e.block}" if e.block else ""
        console.print(f"[green]↶[/green] {what}{label} ({e.kind}) in {e.file}")
    if problem:
        console.print(f"[red]✗[/red] stopped: {problem}")
        raise typer.Exit(1)


@app.command()
def index(
    path: ProjectPath = Path("."),
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Ignore the cache.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Dump the full index as JSON.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also list macros, labels, samples.")
    ] = False,
) -> None:
    """Print what voxtex learned about the project."""
    from voxtex.project.discover import DiscoveryError
    from voxtex.project.index import load_index
    from voxtex.project.report import print_index

    try:
        idx, cached = load_index(path, rebuild=rebuild)
    except DiscoveryError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1) from e
    if as_json:
        print(idx.model_dump_json(indent=2))
        return
    print_index(console, idx, cached=cached, verbose=verbose)


def main() -> None:
    app()
