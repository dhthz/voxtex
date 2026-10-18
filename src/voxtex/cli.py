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
        listen(Path("."))


@app.command()
def listen(path: ProjectPath = Path(".")) -> None:
    """Index the project, listen to the mic, and write LaTeX at the %> HERE marker."""
    _not_yet("M5")


@app.command()
def text(path: ProjectPath = Path(".")) -> None:
    """Same engine as `listen`, but reads typed lines from stdin."""
    _not_yet("M2")


@app.command()
def init() -> None:
    """First-run wizard: backend, API key, microphone."""
    _not_yet("M6")


@app.command()
def doctor() -> None:
    """Check mic, backend, models, TeX and project detection."""
    _not_yet("M6")


@app.command()
def undo(n: Annotated[int, typer.Argument(min=1, help="Number of writes to revert.")] = 1) -> None:
    """Revert the last N writes."""
    _not_yet("M4")


@app.command()
def index(path: ProjectPath = Path(".")) -> None:
    """Print what voxtex learned about the project."""
    _not_yet("M1")


def main() -> None:
    app()
