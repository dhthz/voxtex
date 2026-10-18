"""Prompt construction.

The system prompt depends only on the project index and config, so it is
byte-identical across calls (providers with prompt caching can reuse it).
Everything that changes per utterance goes into the user message.
"""

from dataclasses import dataclass, field

from voxtex.engine.buffer import Block
from voxtex.engine.retrieve import macro_signature
from voxtex.project.bib import BibEntry
from voxtex.project.index import ProjectIndex
from voxtex.project.labels import Label
from voxtex.project.preamble import Macro

BLOCK_MAX_CHARS = 700
LANGUAGE_NAMES = {"en": "English", "el": "Greek"}

STYLE_TEXT = {
    "academic": "formal academic prose that matches the register of the sample paragraph",
    "clean": "clear, concise prose that keeps the speaker's wording where it is already good",
}

SYSTEM = """\
You turn a thesis author's free speech into LaTeX for their document.
Reply with exactly one JSON object and nothing else:
{{"action":"wait"}}  the thought is clearly unfinished
{{"action":"ignore"}}  thinking aloud, or talk not meant for the document
{{"action":"insert","latex":"..."}}  new text at the cursor
{{"action":"replace","block_id":"b3","latex":"..."}}  rewrite a recent block (corrections, "make that more formal")
{{"action":"delete","block_id":"b3"}}  remove a recent block ("scratch that", "undo")
{{"action":"uncertain_insert","latex":"...","note":"..."}}  best guess; note says what to check

Rules:
- Only dictated content goes in the document. Greetings, microphone tests, questions or remarks addressed to you ("is this working?", "give me the result"), and speech that adds no content are "ignore". Never fill in content the speaker didn't say.
- Write {style} in {language}. The speaker may use Greek or English; never transcribe verbatim, drop fillers and false starts.
- SPEECH comes from speech recognition and may contain misheard words; correct them from context and TERMS (e.g. a misheard technical term or a word that makes no sense in the sentence).
- Rewrite only what was said: improve wording and register, but add no claims, interpretations, conclusions or qualifiers (e.g. "average", "significant") the speaker didn't make.
- Never invent facts, numbers, citations or labels. Use only numbers the speaker said. Cite only keys listed under SOURCES; reference only labels listed under LABELS or created in your own output.
- If the source or label is ambiguous, pick the best one and use uncertain_insert with a short note.
- Keep items (table rows, list entries, compared models) in the order the speaker said them, so "the first/second one" means the same to both of you.
- Corrections refer to the most recent block unless the speaker says otherwise; replace returns the whole rewritten block.
- "latex" is body content only: no preamble, no \\begin{{document}}, no markdown fences, no new packages or \\newcommand.
- Use the author's macros where they fit.

Conventions of this document:
{rules}
{sample}"""


@dataclass
class PromptContext:
    pending: str
    nearby: str
    recent: list[Block]
    bib: list[BibEntry]
    labels: list[Label]
    macros: list[Macro]
    kinds: set[str]  # figure/table/... mentioned in the speech
    force: bool = False  # the speaker paused: a decision is required
    terms: list[str] = field(default_factory=list)  # thesis spellings of terms heard


def document_language(idx: ProjectIndex, configured: str = "auto") -> str:
    code = idx.conventions.language if configured == "auto" else configured
    return LANGUAGE_NAMES.get(code, code)


def system_prompt(idx: ProjectIndex, style: str = "academic", language: str = "auto") -> str:
    rules = "\n".join(f"- {r}" for r in idx.conventions.rules())
    sample = ""
    if idx.style.paragraphs:
        sample = "\nSample paragraph from the thesis (match its tone, not its content):\n"
        sample += idx.style.paragraphs[0].text
    return SYSTEM.format(
        style=STYLE_TEXT.get(style, STYLE_TEXT["academic"]),
        language=document_language(idx, language),
        rules=rules,
        sample=sample,
    )


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def user_message(idx: ProjectIndex, ctx: PromptContext) -> str:
    parts = []
    if ctx.nearby:
        parts.append("TEXT BEFORE CURSOR:\n" + ctx.nearby)
    if ctx.recent:
        parts.append(
            "RECENT BLOCKS (yours, editable):\n"
            + "\n".join(f"[{b.id}] {_clip(b.latex, BLOCK_MAX_CHARS)}" for b in ctx.recent)
        )
    if ctx.bib:
        parts.append(
            "SOURCES (key | first author | year | title):\n" + "\n".join(e.short for e in ctx.bib)
        )
    if ctx.labels:
        parts.append("LABELS: " + ", ".join(f"{lab.name} ({lab.kind})" for lab in ctx.labels))
    if ctx.macros:
        parts.append("MACROS: " + ", ".join(macro_signature(m) for m in ctx.macros))
    if ctx.terms:
        parts.append("TERMS AS SPELLED IN THE THESIS: " + ", ".join(ctx.terms))
    for kind in ("table", "figure"):
        sample = getattr(idx.style, kind)
        if kind in ctx.kinds and sample and sample.text not in ctx.nearby:
            parts.append(f"EXAMPLE {kind.upper()} FROM THE THESIS:\n{sample.text}")
    parts.append("SPEECH:\n" + ctx.pending)
    if ctx.force:
        parts.append('The speaker has paused. Do not answer "wait".')
    return "\n\n".join(parts)


def build_messages(
    idx: ProjectIndex, ctx: PromptContext, style: str = "academic", language: str = "auto"
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(idx, style, language)},
        {"role": "user", "content": user_message(idx, ctx)},
    ]


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count (≈4 chars/token for English, fewer for Greek) for budgeting and tests."""
    total = 0
    for m in messages:
        text = m["content"]
        non_ascii = sum(1 for c in text if ord(c) > 127)
        total += (len(text) - non_ascii) / 4 + non_ascii / 2
    return int(total)
