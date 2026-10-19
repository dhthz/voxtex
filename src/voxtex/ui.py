"""How voxtex looks in the terminal, in Claude Code's visual language.

- a rounded welcome box with the project, target, AI chain and speech backend
- what was said, as `> …`
- each write as `⏺ Update(file)` + `⎿ summary` + a diff with real line numbers
- while listening: a spinner line and a rounded box with the words heard so far

Details (tokens, timings, validation retries) only show with --verbose.
"""

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from voxtex import __version__
from voxtex.engine.decide import Delete, Ignore, Replace, UncertainInsert, Wait
from voxtex.engine.engine import Step
from voxtex.io.writer import read_file
from voxtex.session import Placer, Session, render

ACCENT = "#d77757"
DIM = "grey50"
FAINT = "grey37"
ADD_BG = "on #1d3b26"
DEL_BG = "on #4a2026"
ADD_WORD = "on #2f6b3f"  # the changed words inside a replaced line
DEL_WORD = "on #7a2f38"
SPINNER = "·✢✳✶✻✽✻✶✳✢"
CONTEXT_LINES = 2
MAX_DIFF_LINES = 14


@dataclass
class DiffLine:
    number: int | None
    sign: str  # " " | "+" | "-"
    text: str
    marks: list[tuple[int, int]] = field(default_factory=list)  # changed character ranges


@dataclass
class WriteView:
    """What one step did to a file, ready to draw."""

    verb: str  # Write | Update | Delete
    target: str
    summary: str
    lines: list[DiffLine] = field(default_factory=list)
    status: str = "ok"  # ok | check (written with a note) | error


def _lines(text: str) -> list[str]:
    return text.split("\n") if text else []


def _line_of(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def word_changes(old: str, new: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Character ranges that differ, word by word: (in old, in new)."""
    tok = re.compile(r"\s+|\w+|[^\w\s]")
    a = [(m.start(), m.end()) for m in tok.finditer(old)]
    b = [(m.start(), m.end()) for m in tok.finditer(new)]
    sm = SequenceMatcher(None, [old[i:j] for i, j in a], [new[i:j] for i, j in b], autojunk=False)
    marks_old, marks_new = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if i2 > i1:
            marks_old.append((a[i1][0], a[i2 - 1][1]))
        if j2 > j1:
            marks_new.append((b[j1][0], b[j2 - 1][1]))
    return marks_old, marks_new


def _paired(removed: list[str], added: list[str], start: int) -> list[DiffLine]:
    """Removed lines, then added ones; line pairs get word-level marks."""
    old_rows = [DiffLine(None, "-", t) for t in removed]
    new_rows = [DiffLine(start + i, "+", t) for i, t in enumerate(added)]
    if len(removed) == len(added):
        for o, n in zip(old_rows, new_rows, strict=True):
            if o.text != n.text:
                o.marks, n.marks = word_changes(o.text, n.text)
    return old_rows + new_rows


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def view_of(step: Step, before: str, span, rel: str) -> WriteView | None:
    """Build the view for a written step. `before`: the block's text before this step
    (replace/delete); `span`: where it is now (None after a delete or in a dry run)."""
    d, b = step.decision, step.block
    if b is None or isinstance(d, Wait | Ignore):
        return None
    new = render(b) if not isinstance(d, Delete) else ""
    status = "check" if isinstance(d, UncertainInsert) or b.note else "ok"
    added, removed = _lines(new), _lines(before)
    verb = {Replace: "Update", Delete: "Delete"}.get(type(d), "Write")
    if span is None:  # dry run, or the block is gone
        rows = [DiffLine(None, "-", t) for t in removed] + [DiffLine(None, "+", t) for t in added]
        what = "would add" if not removed else "would change"
        return WriteView(verb, rel or "(dry run)", f"{b.id}: {what} {_plural(len(added), 'line')}",
                         rows, status)  # fmt: skip
    content = read_file(span.file)
    start = _line_of(content, span.offset)
    all_lines = content.split("\n")
    rows = []
    for n in range(max(1, start - CONTEXT_LINES), start):
        rows.append(DiffLine(n, " ", all_lines[n - 1]))
    if isinstance(d, Delete):
        rows += [DiffLine(start + i, "-", t) for i, t in enumerate(removed)]
        summary = f"Removed {b.id} ({_plural(len(removed), 'line')})"
        after = start
    else:
        rows += _paired(removed, added, start)
        after = start + len(added)
        if removed:
            summary = f"Updated {b.id}: {_plural(len(added), 'line')} at line {start}"
        else:
            summary = f"Added {_plural(len(added), 'line')} at line {start} ({b.id})"
    for n in range(after, min(after + CONTEXT_LINES, len(all_lines) + 1)):
        rows.append(DiffLine(n, " ", all_lines[n - 1]))
    if b.note and not isinstance(d, Delete):
        summary += " · needs a check"
    return WriteView(verb, rel, summary, rows, status)


class UI:
    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose

    # --- static pieces ---------------------------------------------------------------------

    def banner(self, rows: Iterable[tuple[str, str]], hint: str = "") -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style=DIM, no_wrap=True)
        grid.add_column()
        for label, value in rows:
            grid.add_row(label, Text(value))
        title = Text.assemble(("✻ ", ACCENT), ("voxtex", "bold"), (f" {__version__}", DIM))
        body: list[RenderableType] = [title, Text(""), grid]
        if hint:
            body += [Text(""), Text(hint, style=DIM)]
        self.console.print(
            Panel(Group(*body), box=box.ROUNDED, border_style=ACCENT, padding=(0, 1), expand=False)
        )

    def heard(self, text: str, where: str = "") -> None:
        self.console.print()
        line = Text.assemble(("> ", DIM), (text, "grey85"))
        if where:
            line.append(f"  {where}", style=FAINT)
        self.console.print(line)

    def result(self, text: str, style: str = DIM) -> None:
        self.console.print(Text.assemble(("  ⎿  ", DIM), (text, style)))

    def more(self, text: str, style: str = DIM) -> None:
        """A further line under a ⎿ result."""
        self.console.print(Text.assemble("     ", (text, style)))

    def error(self, text: str) -> None:
        self.console.print(Text.assemble(("⏺ ", "red"), (text, "red")))

    def notice(self, text: str) -> None:
        self.console.print(Text.assemble(("  ⎿  ", DIM), (text, "cyan")))

    def done(self, text: str) -> None:
        self.console.print()
        self.console.print(Text(text, style=DIM))

    # --- a write ---------------------------------------------------------------------------

    def write(self, v: WriteView) -> None:
        color = {"ok": "green", "check": "yellow", "error": "red"}[v.status]
        self.console.print(Text.assemble(("⏺ ", color), (v.verb, "bold"), f"({v.target})"))
        self.result(v.summary, "yellow" if v.status == "check" else DIM)
        self.console.print(self._diff(v.lines))

    def _diff(self, lines: list[DiffLine]) -> Table:
        changed = [i for i, ln in enumerate(lines) if ln.sign != " "]
        if len(changed) > MAX_DIFF_LINES:  # long blocks: show the start and how much more
            cut = changed[MAX_DIFF_LINES - 1]
            hidden = len(changed) - MAX_DIFF_LINES + 1
            lines = lines[:cut] + [DiffLine(None, "…", f"{hidden} more lines")]
        width = max([len(str(ln.number)) for ln in lines if ln.number] or [2])
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(width=3 + width, justify="right", no_wrap=True)
        t.add_column(width=1, no_wrap=True)
        t.add_column(ratio=1, overflow="fold")
        for ln in lines:
            number = Text(str(ln.number) if ln.number else "", style=FAINT)
            if ln.sign == "…":
                t.add_row(number, Text(""), Text(ln.text, style=DIM))
                continue
            if ln.sign == "+":
                code = _highlight(ln.text)
            else:  # context and removed lines stay quiet
                code = Text(ln.text, style=DIM if ln.sign == " " else "grey70")
            bg = {"+": ADD_BG, "-": DEL_BG}.get(ln.sign)
            for i, j in ln.marks:
                code.stylize(ADD_WORD if ln.sign == "+" else DEL_WORD, i, j)
            t.add_row(number, Text(ln.sign, style="green" if ln.sign == "+" else "red"), code,
                      style=bg)  # fmt: skip
        return t

    # --- a whole step ----------------------------------------------------------------------

    def step(
        self,
        step: Step,
        session: Session | None = None,
        place: Placer | None = None,
        rel: str = "",
    ) -> None:
        """Apply `step` through `session` (unless it's a dry run) and draw it."""
        d, b = step.decision, step.block
        for n in step.notices:
            self.notice(n)
        if isinstance(d, Wait):
            self.result("Waiting for the rest of the thought…")
            return
        if isinstance(d, Ignore) or b is None:
            self.result("Not content, nothing written")
            self._meta(step)
            return
        before = ""
        span = None
        if session is not None:
            old = session.spans.get(b.id)
            before = old.text if old else ""
            try:
                notes = session.write(step, place)
            except OSError as e:
                self.error(f"Could not write: {e}")
                return
            span = session.spans.get(b.id)
            if isinstance(d, Delete):
                span = old
            if span is not None:
                rel = session.writer.rel(span.file)
        else:
            notes = []
            if step.previous is not None:
                before = render(step.previous)
        v = view_of(step, before, span, rel)
        if v is not None:
            self.write(v)
        for n in notes:
            self.more(n, "cyan")
        if step.report:
            for w in step.report.warnings:
                self.more(f"! {w.message}", "yellow")
        self._meta(step)

    def _meta(self, step: Step) -> None:
        if not self.verbose:
            return
        r = step.report
        checks = f" · checked {r.seconds:.1f}s" + (" (compiled)" if r.compiled else "") if r else ""
        self.more(
            f"{step.provider} · {step.input_tokens} in / {step.output_tokens} out"
            f" · {step.latency_s:.1f}s{checks}",
            FAINT,
        )
        if step.retried:
            self.more("fixed after checks: " + "; ".join(step.first_errors), FAINT)

    # --- live status (listen, watch) -------------------------------------------------------

    def status(self, verb: str, detail: str = "", speech: str | None = None,
               hints: str = "", active: bool = True) -> RenderableType:  # fmt: skip
        frame = SPINNER[int(time.monotonic() * 8) % len(SPINNER)] if active else "✻"
        line = Text.assemble((f"{frame} ", ACCENT if active else DIM),
                             (f"{verb}…", ACCENT if active else DIM))  # fmt: skip
        if detail:
            line += Text(f" ({detail})", style=DIM)
        parts: list[RenderableType] = [Text(""), line]
        if speech is not None:
            room = max(20, (self.console.width - 8) * 2)
            shown = speech[-room:] if speech else ""
            body = (
                Text.assemble(("> ", DIM), ("…" if len(speech) > room else "") + shown)
                if shown
                else Text("> speak…", style=FAINT)
            )
            parts.append(Panel(body, box=box.ROUNDED, border_style=FAINT, padding=(0, 1)))
        if hints:
            parts.append(Text(f"  {hints}", style=FAINT))
        return Group(*parts)


def _highlight(line: str) -> Text:
    text = Syntax("", "latex", theme="ansi_dark").highlight(line)
    text.rstrip()
    return text


def project_row(idx) -> str:
    pre = idx.preamble
    return (
        f"{idx.main} · {pre.documentclass or 'article'} · {_plural(len(idx.bib), 'reference')}"
        f" · {_plural(len(idx.labels), 'label')}"
    )


def target_row(rel: str, line: int, found: bool) -> str:
    return f"{rel}:{line} (%> HERE)" if found and line else f"{rel} (end of file)"


def session_summary(
    chain, writes: int, extra: str = "", verbose: bool = False, undo_hint: bool = True
) -> str:
    t = chain.totals
    parts = [f"{writes} write{'' if writes == 1 else 's'}"]
    if t.calls:
        parts.append(f"{t.calls} AI call{'' if t.calls == 1 else 's'}")
        parts.append(f"{t.input_tokens:,} tokens in")
        if verbose:
            parts.append(f"{t.output_tokens:,} out · avg {t.input_tokens // t.calls:,} in/call")
    if extra:
        parts.append(extra)
    tail = " · voxtex undo reverts" if writes and undo_hint else ""
    return "Session: " + " · ".join(parts) + tail


def mic_bar(level: float) -> str:
    return "▁▂▃▄▅▆▇█"[min(7, int(level * 8))]


def rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
