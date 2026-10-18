"""Turns engine steps into file edits and remembers where each block lives."""

from collections.abc import Callable
from pathlib import Path

from voxtex.engine.buffer import Block
from voxtex.engine.decide import Delete, Insert, Replace, UncertainInsert
from voxtex.engine.engine import Engine, Step
from voxtex.io.writer import TextSpan, Writer

CHECK_PREFIX = "% voxtex-check: "

# How a new block gets into a file: text mode inserts at the marker, watch mode replaces
# the dictated %> lines.
Placer = Callable[[str, Block], TextSpan]


def render(block: Block) -> str:
    """The text written to the file: the LaTeX plus a check comment for uncertain output."""
    text = block.latex.rstrip()
    if block.note:
        text += "\n" + CHECK_PREFIX + " ".join(block.note.split())
    return text


class Session:
    def __init__(self, engine: Engine, writer: Writer, target: Path | None):
        self.engine = engine
        self.writer = writer
        self.target = target
        self.spans: dict[str, TextSpan] = {}

    def at_marker(self, text: str, block: Block) -> TextSpan:
        if self.target is None:
            raise RuntimeError("no target file to write to")
        return self.writer.insert_at_marker(self.target, text, block.id)

    def write(self, step: Step, place: Placer | None = None) -> list[str]:
        """Apply `step` to the files. Returns notices for the user."""
        place = place or self.at_marker
        d, b = step.decision, step.block
        if b is None:
            return []
        if isinstance(d, Insert | UncertainInsert):
            self.spans[b.id] = place(render(b), b)
            return []
        span = self.spans.get(b.id)
        if isinstance(d, Replace):
            new = self.writer.replace_span(span, render(b), "replace", b.id) if span else None
            if new is None:
                # The author edited or removed that text since: don't touch it, add the new one.
                self.spans[b.id] = place(render(b), b)
                return [f"{b.id} was changed in the file since; wrote the new version below it"]
            self.spans[b.id] = new
            return []
        if isinstance(d, Delete):
            if span is None or self.writer.replace_span(span, "", "delete", b.id) is None:
                return [f"{b.id} was changed in the file since; left it in place"]
            self.spans.pop(b.id, None)
        return []
