"""Command-line interface."""

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
    """With no command, `voxtex` opens the editor on your last chapter."""
    if ctx.invoked_subcommand is None:
        from voxtex.config import load_config

        _first_run()
        if load_config(Path.cwd()).stt_mode == "external":  # dictating with another app
            watch(Path("."), False, None, None, False)
        edit(Path("."), None, False, None, None, None, False)


def _first_run() -> None:
    """On the very first use in a terminal, offer `voxtex init` once."""
    import sys

    from rich.prompt import Confirm

    from voxtex.initwizard import first_run_needed, mic_test, run_init, skip_first_run

    if not (sys.stdin.isatty() and sys.stdout.isatty()) or not first_run_needed():
        return
    console.print("[bold]Welcome to voxtex.[/bold] Setup takes about a minute.")
    if Confirm.ask("Set it up now?", default=True, console=console):
        run_init(console, mic_test=mic_test)
        console.print()
    else:
        skip_first_run()
        console.print("[dim]Skipped. Run voxtex init any time.[/dim]")


@app.command()
def edit(
    path: Annotated[
        Path, typer.Argument(help="A chapter file, or the project folder (default: here).")
    ] = Path("."),
    file: Annotated[
        Path | None, typer.Option("--file", help="Chapter to open (default: the last one).")
    ] = None,
    private: Annotated[
        bool, typer.Option("--private", help="Keep everything local: local AI and speech-to-text.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Use only this AI provider from the config.")
    ] = None,
    stt: Annotated[
        str | None, typer.Option("--stt", help="Speech-to-text: auto, local, groq or openai.")
    ] = None,
    mic: Annotated[
        str | None, typer.Option("--mic", help="Microphone name or index (see listen --list-mics).")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show token totals at the end.")
    ] = False,
) -> None:
    """Open a chapter in the voxtex editor: ctrl+space, talk, ctrl+space.

    What you said goes at the cursor; with text selected, it says what to change.
    This is what `voxtex` with no command does.
    """
    _first_run()
    from voxtex.editor.app import run_editor

    raise typer.Exit(run_editor(console, path, file, private, provider, stt, mic, verbose))


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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also show tokens, timings and retries.")
    ] = False,
) -> None:
    """Without the editor: write at a %> HERE marker in a file open in your own editor."""
    if not list_mics:
        _first_run()
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

    raise typer.Exit(run_listen(console, path, private, provider, file, dry_run, stt, mic, verbose))


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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also show tokens, timings and retries.")
    ] = False,
) -> None:
    """Convert %> lines typed by any dictation app into LaTeX on save."""
    from voxtex.watchmode import run_watch

    _first_run()

    raise typer.Exit(run_watch(console, path, private, provider, poll, verbose))


@app.command()
def init() -> None:
    """Set up speech, free AI keys (tested on the spot), local AI, privacy and microphone."""
    from voxtex.initwizard import mic_test, run_init

    raise typer.Exit(run_init(console, mic_test=mic_test))


@app.command()
def doctor(
    path: ProjectPath = Path("."),
    quick: Annotated[
        bool, typer.Option("--quick", help="Skip the microphone recording and the speed test.")
    ] = False,
) -> None:
    """Check the project, AI providers and models, speech-to-text speed, microphone and TeX."""
    from voxtex.doctor import run_doctor

    raise typer.Exit(run_doctor(console, path, quick))


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
        console.print(f"[green]⏺[/green] [bold]Undo[/bold]({e.file})")
        console.print(f"[grey50]  ⎿  {what}{label} ({e.kind})[/grey50]")
    if problem:
        console.print(f"[red]⏺ Stopped: {problem}[/red]")
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
