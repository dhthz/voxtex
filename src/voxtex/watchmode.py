"""`voxtex watch`: convert `%>` lines dictated by any app into LaTeX when the file is saved.

No microphone, no speech-to-text: an external tool (Voxtype, macOS dictation, Wispr
Flow...) types onto `%>` lines, which are LaTeX comments, so the document still
compiles mid-dictation. On save, each block of `%>` lines goes through the normal
engine and is replaced in place by validated LaTeX. Every change is undoable.
"""

import queue
import re
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from voxtex.app import build_chain, open_project
from voxtex.engine.buffer import Block
from voxtex.engine.decide import Ignore, Insert, UncertainInsert, Wait
from voxtex.engine.engine import Engine
from voxtex.io.dictation import Dictation, find_dictations
from voxtex.io.history import sha
from voxtex.io.marker import MarkerLocation
from voxtex.io.writer import TextSpan, Writer, read_file
from voxtex.providers import ChainExhausted
from voxtex.session import Session
from voxtex.ui import UI, project_row, session_summary

DEBOUNCE_S = 0.3
POLL_INTERVAL_S = 0.5


class DictationProcessor:
    """Processes the %> blocks of one file at a time (no file watching; easy to test)."""

    def __init__(
        self, engine: Engine, writer: Writer, console: Console | None = None, verbose: bool = False
    ):
        self.engine = engine
        self.ui = UI(console, verbose) if console else None
        self.writes = 0
        self.busy = False
        self.engine.blocks_in_location = True  # earlier blocks are already in the file text
        self.writer = writer
        self.session = Session(engine, writer, target=None)
        self.console = console
        self.skip: set[tuple[Path, str]] = set()  # blocks that failed; retried after an edit

    def process(self, path: Path) -> int:
        """Convert every %> block in `path`, top to bottom. Returns how many were handled."""
        handled = 0
        while True:
            content = read_file(path)
            todo = [d for d in find_dictations(content) if (path, d.transcript) not in self.skip]
            if not todo:
                return handled
            if not self.handle(path, content, todo[0]):
                self.skip.add((path, todo[0].transcript))
            handled += 1

    def handle(self, path: Path, content: str, d: Dictation) -> bool:
        rel = self.writer.rel(path)
        ui = self.ui
        if ui:
            ui.heard(d.transcript, f"{rel}:{d.line}")
        self.engine.location = MarkerLocation(path, rel, d.line, content[: d.start])
        self.busy = True
        try:
            step = self.engine.feed(d.transcript, force=True)
        except ChainExhausted as e:
            if ui:
                ui.error(f"{e} (left the %> lines as they are)")
            self.engine.buffer.clear_pending()
            return False
        finally:
            self.busy = False
        if step is None:
            return False
        raw = d.raw(content)
        trailing = re.search(r"\r?\n$", raw)
        eol = trailing.group(0) if trailing else ""

        def in_place(text: str, block: Block) -> TextSpan:
            e = self.writer.edit(path, d.start, raw, text + eol, kind="watch", block=block.id)
            return TextSpan(path, d.start, e.added[: len(e.added) - len(eol)] if eol else e.added)

        dec = step.decision
        if isinstance(dec, Wait):
            self.engine.buffer.clear_pending()
            if ui:
                ui.result("Left as is: it sounds unfinished. Add to it and save again.")
            return False
        if ui:
            ui.step(step, self.session, in_place, rel)
        else:
            self.session.write(step, in_place)
        if step.block is not None:
            self.writes += 1
        if not isinstance(dec, Insert | UncertainInsert):
            # A correction of an earlier block, a delete, or thinking aloud: also remove
            # the %> lines themselves (undo brings them back).
            current = read_file(path)
            if current[d.start : d.end] == raw:
                kind = "ignore" if isinstance(dec, Ignore) else "watch"
                self.writer.edit(path, d.start, raw, "", kind=kind)
        return True


def is_wsl_windows_drive(path: Path) -> bool:
    """Files under /mnt/c, /mnt/e... are edited from Windows; inotify doesn't see that."""
    return bool(re.match(r"^/mnt/[a-z]/", str(path.resolve())))


def run_watch(
    console: Console,
    path: Path,
    private: bool,
    provider: str | None,
    poll: bool | None,
    verbose: bool = False,
) -> int:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    from watchdog.observers.polling import PollingObserver

    opened = open_project(console, path)
    if opened is None:
        return 1
    idx, config = opened
    chain = build_chain(console, idx, config, private, provider)
    if chain is None:
        return 1
    root = idx.root_path
    writer = Writer(root)
    engine = Engine(idx, chain, config, None)
    proc = DictationProcessor(engine, writer, console, verbose)
    ui = proc.ui

    use_poll = is_wsl_windows_drive(root) if poll is None else poll
    events: queue.Queue[Path] = queue.Queue()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory or event.event_type not in ("modified", "created", "moved"):
                return
            p = Path(getattr(event, "dest_path", "") or event.src_path)
            if p.suffix == ".tex" and ".voxtex" not in p.parts and not p.name.startswith("."):
                events.put(p)

    observer = PollingObserver(timeout=POLL_INTERVAL_S) if use_poll else Observer()
    observer.schedule(Handler(), str(root), recursive=True)
    observer.start()
    mode = "polling (Windows drive)" if use_poll else "file events"
    ui.banner(
        [
            ("thesis", project_row(idx)),
            ("watching", f"{root} · {mode}"),
            ("AI", " → ".join(p.name for p in chain.usable()) + (" (private)" if private else "")),
        ],
        hint="Dictate onto lines starting with %>, then save. Ctrl+C to stop.",
    )

    def status():
        if proc.busy:
            return ui.status("Thinking", chain.active or "", hints="ctrl+c to stop")
        return ui.status("Watching for %> lines", hints="ctrl+c to stop", active=False)

    live = Live(get_renderable=status, console=console, refresh_per_second=10, transient=True)
    live.start()

    # Convert anything already waiting in the project.
    for rel in idx.body_files:
        p = root / rel
        if p.is_file() and find_dictations(read_file(p)):
            proc.process(p)

    try:
        while True:
            first = events.get()
            pending = {first}
            deadline = time.monotonic() + DEBOUNCE_S
            while (left := deadline - time.monotonic()) > 0:  # debounce bursts of saves
                try:
                    pending.add(events.get(timeout=left))
                    deadline = time.monotonic() + DEBOUNCE_S
                except queue.Empty:
                    break
            for p in sorted(pending):
                if not p.is_file():
                    continue
                content = read_file(p)
                if writer.last_written.get(p.resolve()) == sha(content):
                    continue  # our own write
                proc.skip = {k for k in proc.skip if k[0] != p}  # the author edited: retry
                if find_dictations(content):
                    proc.process(p)
    except KeyboardInterrupt:
        console.print()
    finally:
        live.stop()
        observer.stop()
        observer.join(timeout=2)
    ui.done(session_summary(chain, proc.writes))
    return 0
