"""Startup shared by `listen`, `text` and `watch`: project, AI chain, target file."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt

from voxtex.config import Config, load_config
from voxtex.engine.engine import Engine
from voxtex.io.marker import MarkerLocation, fallback_location, find_marker
from voxtex.io.writer import Writer, read_file
from voxtex.project.discover import DiscoveryError
from voxtex.project.index import STATE_DIR, ProjectIndex, load_index
from voxtex.providers import Chain, QuotaStore
from voxtex.session import Session


def open_project(console: Console, path: Path) -> tuple[ProjectIndex, Config] | None:
    try:
        idx, _ = load_index(path)
    except DiscoveryError as e:
        console.print(f"[red]✗[/red] {e}")
        return None
    return idx, load_config(idx.root_path)


def build_chain(
    console: Console, idx: ProjectIndex, config: Config, private: bool, provider: str | None
) -> Chain | None:
    root = idx.root_path
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
        console.print("  Put GROQ_API_KEY=... in a .env file, or start Ollama.")
        return None
    return chain


def location_in(root: Path, path: Path) -> MarkerLocation:
    """Where text goes in `path`: at its marker, or (until one is added) its end."""
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    found = find_marker(root, [rel])
    if found:
        return found
    return MarkerLocation(file=path, rel=rel, line=0, before=read_file(path))


def choose_target(
    console: Console, idx: ProjectIndex, file_opt: Path | None, interactive: bool
) -> Path | None:
    """The file to write into: --file, else the %> HERE marker, else ask once."""
    root = idx.root_path
    if file_opt is not None:
        p = file_opt if file_opt.is_absolute() else root / file_opt
        if not p.is_file():
            console.print(f"[red]✗[/red] {file_opt} doesn't exist")
            return None
        return p
    found = find_marker(root, idx.body_files)
    if found:
        return found.file
    chapters = [f for f in idx.body_files if f != idx.main] or idx.body_files
    if not interactive:
        console.print(
            "[red]✗[/red] no %> HERE marker found. Add a line `%> HERE` where text should go, "
            "or pass --file chapters/x.tex"
        )
        return None
    console.print("[yellow]No %> HERE marker found.[/yellow] Which file should I write into?")
    for i, rel in enumerate(chapters, 1):
        console.print(f"  {i}. {rel}")
    choice = IntPrompt.ask("File", choices=[str(i) for i in range(1, len(chapters) + 1)])
    return root / chapters[choice - 1]


def _state(root: Path) -> dict:
    try:
        return json.loads((root / STATE_DIR / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(root: Path, state: dict) -> None:
    p = root / STATE_DIR / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1), encoding="utf-8")


def offer_gitignore(console: Console, root: Path, interactive: bool) -> None:
    """If the thesis is a git repo and .voxtex/ isn't ignored, offer (once) to ignore it."""
    if not interactive or _state(root).get("gitignore_asked"):
        return
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() == "true"  # fmt: skip
        if not inside:
            return
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", f"{STATE_DIR}/"],
            capture_output=True, timeout=5,
        ).returncode == 0  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return
    state = _state(root)
    state["gitignore_asked"] = True
    _save_state(root, state)
    if ignored:
        return
    if Confirm.ask(f"Add {STATE_DIR}/ (backups, history, cache) to .gitignore?", default=True):
        gi = root / ".gitignore"
        existing = read_file(gi) if gi.exists() else ""
        sep = "" if not existing or existing.endswith("\n") else "\n"
        gi.write_text(existing + f"{sep}{STATE_DIR}/\n", encoding="utf-8")
        console.print(f"[green]✓[/green] added {STATE_DIR}/ to .gitignore")


@dataclass
class Workspace:
    """Everything a dictation session needs except the audio."""

    idx: ProjectIndex
    config: Config
    chain: Chain
    engine: Engine
    location: MarkerLocation | None
    session: Session | None  # None in a dry run

    @property
    def root(self) -> Path:
        return self.idx.root_path


def open_workspace(
    console: Console,
    path: Path,
    private: bool,
    provider: str | None,
    file_opt: Path | None,
    dry_run: bool,
    interactive: bool,
) -> Workspace | None:
    """Project, AI chain, target file, engine and (unless dry run) the writing session."""
    opened = open_project(console, path)
    if opened is None:
        return None
    idx, config = opened
    chain = build_chain(console, idx, config, private, provider)
    if chain is None:
        return None
    root = idx.root_path
    if dry_run:
        target = (root / file_opt) if file_opt else None
        location = location_in(root, target) if target else fallback_location(root, idx.body_files)
        return Workspace(idx, config, chain, Engine(idx, chain, config, location), location, None)
    target = choose_target(console, idx, file_opt, interactive)
    if target is None:
        return None
    offer_gitignore(console, root, interactive)
    writer = Writer(root)
    writer.ensure_marker(target)
    location = location_in(root, target)
    engine = Engine(idx, chain, config, location)
    return Workspace(idx, config, chain, engine, location, Session(engine, writer, target))
