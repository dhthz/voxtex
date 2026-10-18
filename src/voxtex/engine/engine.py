"""Orchestrates one decision:
buffer → retrieve → prompt → chain → decision → validate (→ one retry) → buffer update.

File writing arrives in M4; for now the engine keeps the written blocks in memory and
reports what it would write.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

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
from voxtex.project.discover import SourceFile
from voxtex.project.index import ProjectIndex
from voxtex.project.labels import Label, collect_labels
from voxtex.providers import Chain, ChainExhausted, ChainResult
from voxtex.validate import Report, ValidationContext, Validator, labels_in
from voxtex.validate.latin import wrap_latin

MAX_PENDING_CHARS = 1200  # force a decision once this much speech is waiting
MAX_PENDING_SECONDS = 20.0
NEARBY_CHARS = 800


@dataclass
class Step:
    decision: Decision
    block: Block | None = None  # the block inserted / replaced / deleted
    previous: Block | None = None  # replace: the block as it was before
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    notices: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # what was sent, for --show-prompt
    report: Report | None = None  # validation of the final output
    retried: bool = False  # validation failed once and the model was asked to fix it
    first_errors: list[str] = field(default_factory=list)  # what the retry was asked to fix

    def add(self, r: ChainResult) -> None:
        self.provider = r.provider
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.latency_s += r.latency_s
        self.notices += r.notices


def has_latex(d: Decision) -> bool:
    return isinstance(d, Insert | UncertainInsert | Replace)


def retry_feedback(report: Report) -> str:
    problems = "\n".join(f"- {i.message}" for i in report.errors)
    return (
        f"Your LaTeX has problems:\n{problems}\n"
        "Fix them and reply with the corrected JSON object only, keeping the same action and "
        "block_id. Don't invent replacements: leave out a citation, reference or number that "
        "isn't available."
    )


class Engine:
    def __init__(
        self,
        idx: ProjectIndex,
        chain: Chain,
        config: Config,
        location: MarkerLocation | None = None,
        clock: Callable[[], float] = time.monotonic,
        validator: Validator | None = None,
    ):
        self.idx = idx
        self.validator = validator or Validator(idx, config.compile_check)
        self.chain = chain
        self.config = config
        self.location = location
        self.clock = clock
        self.buffer = Buffer()
        # Watch mode reads context from the live file, which already holds earlier blocks.
        self.blocks_in_location = False

    # --- context ---------------------------------------------------------------------------------

    def _nearby(self, recent: list[Block]) -> str:
        """The file before the cursor plus older blocks (recent ones are listed separately)."""
        before = self.location.before if self.location else ""
        if self.blocks_in_location:
            return nearby_text(before, NEARBY_CHARS)
        older = [b for b in self.buffer.blocks if b not in recent]
        written = "\n\n".join(b.latex for b in older)
        return nearby_text(before + ("\n\n" + written if written else ""), NEARBY_CHARS)

    def session_labels(self) -> list[Label]:
        """Labels created by blocks this session (newest last, so "the table" finds them)."""
        files = [SourceFile(Path(f"<{b.id}>"), b.latex, b.id) for b in self.buffer.blocks]
        return collect_labels(files)

    def context(self, force: bool) -> PromptContext:
        speech = self.buffer.pending_text
        recent = self.buffer.recent(3)
        # Retrieval also looks at recent blocks, so "no wait, cite the other one" still works.
        query = speech + " " + " ".join(b.latex for b in recent[-1:])
        nearby = self._nearby(recent)
        always = [self.idx.conventions.latin_wrapper] if self.idx.conventions.latin_wrapper else []
        return PromptContext(
            pending=speech,
            nearby=nearby,
            recent=recent,
            bib=retrieve.match_bib(query, self.idx.bib),
            labels=retrieve.match_labels(speech, self.idx.labels + self.session_labels()),
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
        speech = self.buffer.pending_text
        step = Step(Wait(action="wait"), messages=messages)
        result: ChainResult[Decision] = self.chain.complete(messages, parse_decision)
        step.add(result)
        decision = self.polish(result.value)
        report = self.validate(decision, speech) if has_latex(decision) else None

        if report and not report.ok:
            step.retried = True
            step.first_errors = [i.message for i in report.errors]
            retry = messages + [
                {"role": "assistant", "content": decision.model_dump_json()},
                {"role": "user", "content": retry_feedback(report)},
            ]
            try:
                second = self.chain.complete(retry, parse_decision)
                step.add(second)
                fixed = self.polish(second.value)
                if (
                    isinstance(fixed, Replace)
                    and self.buffer.get(fixed.block_id) is None
                    and isinstance(decision, Insert | UncertainInsert)
                ):
                    # The model "replaced" its own rejected insert, which was never a block.
                    fixed = Insert(action="insert", latex=fixed.latex)
                if has_latex(fixed):
                    decision = fixed
                    report = self.validate(decision, speech)
            except ChainExhausted as e:
                step.notices.append(f"retry failed: {e}")

        note = "; ".join(i.message for i in report.errors) if report else ""
        applied = self.apply(decision, note)
        applied.provider, applied.input_tokens = step.provider, step.input_tokens
        applied.output_tokens, applied.latency_s = step.output_tokens, step.latency_s
        applied.notices = step.notices + applied.notices
        applied.messages, applied.report = messages, report
        applied.retried, applied.first_errors = step.retried, step.first_errors
        return applied

    def polish(self, d: Decision) -> Decision:
        """Deterministic fixes before validation: wrap English terms in Greek documents."""
        wrapper = self.idx.conventions.latin_wrapper
        if wrapper and has_latex(d):
            return d.model_copy(update={"latex": wrap_latin(d.latex, wrapper)})
        return d

    def validate(self, d: Decision, speech: str) -> Report:
        replaced_block = self.buffer.get(d.block_id) if isinstance(d, Replace) else None
        other = {
            name
            for b in self.buffer.blocks
            if b is not replaced_block
            for name in labels_in(b.latex)
        }
        ctx = ValidationContext(
            idx=self.idx,
            transcript=speech,
            replaced=replaced_block.latex if replaced_block else "",
            other_labels=other,
            context=self._nearby([]),  # file text before the cursor plus every block so far
        )
        return self.validator.validate(d.latex, ctx)

    def flush(self) -> Step | None:
        """The speaker paused: decide on whatever is pending."""
        return self.feed("", force=True) if self.buffer.pending else None

    def apply(self, d: Decision, problems: str = "") -> Step:
        """Update the buffer. `problems` (validation errors left after the retry) turn an
        insert into an uncertain one and are kept as the block's check note."""
        buf = self.buffer
        if problems and isinstance(d, Insert):
            d = UncertainInsert(action="uncertain_insert", latex=d.latex, note=problems)
        elif problems and isinstance(d, UncertainInsert):
            d = d.model_copy(update={"note": "; ".join(x for x in (d.note, problems) if x)})
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
                    note = "; ".join(
                        x for x in (f"meant to replace {bid}, which doesn't exist", problems) if x
                    )
                    fallback = UncertainInsert(action="uncertain_insert", latex=latex, note=note)
                    return Step(fallback, buf.new_block(latex, note))
                before = replace(buf.get(bid))
                return Step(d, buf.replace(bid, latex, problems), previous=before)
            case Delete(block_id=bid):
                buf.clear_pending()
                removed = buf.delete(bid)
                if removed is None:
                    return Step(Ignore(action="ignore"), notices=[f"no block {bid} to delete"])
                return Step(d, removed, previous=removed)
        raise AssertionError(f"unhandled decision {d!r}")
