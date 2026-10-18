"""Orchestrates one decision: buffer → retrieve → prompt → chain → decision → buffer update.

File writing arrives in M4 and validation in M3; for now the engine keeps the
written blocks in memory and reports what it would write.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from voxtex.config import Config
from voxtex.engine import retrieve
from voxtex.engine.buffer import Block, Buffer
from voxtex.engine.decide import (
    Decision,
    Delete,
    Ignore,
    Insert,
    Replace,
    UncertainInsert,
    Wait,
    parse_decision,
)
from voxtex.engine.prompts import PromptContext, build_messages
from voxtex.io.marker import MarkerLocation, nearby_text
from voxtex.project.index import ProjectIndex
from voxtex.providers import Chain, ChainResult

MAX_PENDING_CHARS = 1200  # force a decision once this much speech is waiting
MAX_PENDING_SECONDS = 20.0
NEARBY_CHARS = 800


@dataclass
class Step:
    decision: Decision
    block: Block | None = None  # the block inserted / replaced / deleted
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    notices: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # what was sent, for --show-prompt


class Engine:
    def __init__(
        self,
        idx: ProjectIndex,
        chain: Chain,
        config: Config,
        location: MarkerLocation | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.idx = idx
        self.chain = chain
        self.config = config
        self.location = location
        self.clock = clock
        self.buffer = Buffer()

    # --- context ---------------------------------------------------------------------------------

    def _nearby(self) -> str:
        before = self.location.before if self.location else ""
        written = "\n\n".join(b.latex for b in self.buffer.blocks)
        return nearby_text(before + ("\n\n" + written if written else ""), NEARBY_CHARS)

    def context(self, force: bool) -> PromptContext:
        speech = self.buffer.pending_text
        recent = self.buffer.recent(3)
        # Retrieval also looks at recent blocks, so "no wait, cite the other one" still works.
        query = speech + " " + " ".join(b.latex for b in recent[-1:])
        nearby = self._nearby()
        always = [self.idx.conventions.latin_wrapper] if self.idx.conventions.latin_wrapper else []
        return PromptContext(
            pending=speech,
            nearby=nearby,
            recent=recent,
            bib=retrieve.match_bib(query, self.idx.bib),
            labels=retrieve.match_labels(speech, self.idx.labels),
            macros=retrieve.match_macros(speech, nearby, self.idx.preamble.macros, always),
            kinds=retrieve.kinds_mentioned(speech),
            force=force,
        )

    # --- main entry ------------------------------------------------------------------------------

    def feed(self, text: str, force: bool = False) -> Step | None:
        """Add an utterance and decide. Returns None if there is nothing to decide on."""
        self.buffer.add_utterance(text, self.clock())
        if not self.buffer.pending:
            return None
        since = self.buffer.pending_since or self.clock()
        force = (
            force
            or len(self.buffer.pending_text) > MAX_PENDING_CHARS
            or self.clock() - since > MAX_PENDING_SECONDS
        )
        messages = build_messages(
            self.idx, self.context(force), self.config.style, self.config.language
        )
        result: ChainResult[Decision] = self.chain.complete(messages, parse_decision)
        step = self.apply(result.value)
        step.provider = result.provider
        step.input_tokens = result.input_tokens
        step.output_tokens = result.output_tokens
        step.latency_s = result.latency_s
        step.notices = result.notices + step.notices
        step.messages = messages
        return step

    def flush(self) -> Step | None:
        """The speaker paused: decide on whatever is pending."""
        return self.feed("", force=True) if self.buffer.pending else None

    def apply(self, d: Decision) -> Step:
        buf = self.buffer
        match d:
            case Wait():
                return Step(d)
            case Ignore():
                buf.clear_pending()
                return Step(d)
            case Insert(latex=latex):
                buf.clear_pending()
                return Step(d, buf.new_block(latex))
            case UncertainInsert(latex=latex, note=note):
                buf.clear_pending()
                return Step(d, buf.new_block(latex, note))
            case Replace(block_id=bid, latex=latex):
                buf.clear_pending()
                if buf.get(bid) is None:
                    note = f"meant to replace {bid}, which doesn't exist"
                    fallback = UncertainInsert(action="uncertain_insert", latex=latex, note=note)
                    return Step(fallback, buf.new_block(latex, note))
                return Step(d, buf.replace(bid, latex))
            case Delete(block_id=bid):
                buf.clear_pending()
                removed = buf.delete(bid)
                if removed is None:
                    return Step(Ignore(action="ignore"), notices=[f"no block {bid} to delete"])
                return Step(d, removed)
        raise AssertionError(f"unhandled decision {d!r}")
