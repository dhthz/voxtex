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
- When in doubt, write it: a sentence about the thesis topic is content even if it is a fragment or continues the previous block (e.g. starts with "but"/"αλλά"); losing what the speaker said is worse than a sentence they delete.
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


# The editor works like Wispr Flow: you record, and what you said is always written.
DICTATE_SYSTEM = """\
You format a thesis author's dictation as LaTeX for their document. The words are theirs.
Reply with exactly one JSON object and nothing else:
{{"action":"insert","latex":"..."}}
{{"action":"uncertain_insert","latex":"...","note":"..."}}  if something should be checked (a source, a number or a word you're unsure you heard right); note says what
{{"action":"unclear","note":"..."}}  only if the dictation is speech-recognition noise, not words: random letters or syllables, or words that make no sense in Greek or English here (e.g. "M-A-U-A-O-E", "Belit."). Never for a real word or phrase, however short

Rules:
- Keep the speaker's words, their order and their language(s). If the transcript is already right, return it as it is, only formatted as LaTeX. Never rephrase, polish, translate, shorten or change the register.
- Fix only: punctuation and capitalisation; fillers and stutters ("um", "εεε", repeated words); self-corrections inside the dictation ("0.71, no, 0.73" → 0.73); words speech recognition clearly misheard (use TERMS and the text before the cursor).
- Everything said is dictation, even a question or a remark: write it, never answer it, comment on it or leave it out.
- Spoken punctuation becomes the mark, not the word: "κόμμα"/"comma" → ",", "τελεία"/"full stop"/"period" → ".", "άνω τελεία" → "·", "ερωτηματικό"/"question mark" → ";" in Greek or "?" in English, "άνοιξε/κλείσε παρένθεση" → ( ).
- Spoken formatting becomes LaTeX: a source named from SOURCES is cited with the document's citation command; a figure or table from LABELS is referenced with its reference command; spoken maths becomes math; "new paragraph"/"νέα παράγραφος" starts one; a dictated table or list becomes the environment, following the thesis's examples.
- Escape LaTeX special characters (%, &, _, #, $) in plain text.
- Never add words, facts, numbers or citations the speaker didn't say. Cite only keys under SOURCES; reference only labels under LABELS.
- "latex" is body content only: no preamble, no markdown fences, no new packages or \\newcommand. Use the author's macros where they fit.

Conventions of this document:
{rules}"""

COMMAND_SYSTEM = """\
You edit a thesis author's LaTeX on a spoken instruction. They selected some text and said what to do with it.
Reply with exactly one JSON object and nothing else:
{{"action":"replace","block_id":"sel","latex":"..."}}  the selected text, changed as asked

Rules:
- Change only what the instruction asks for; leave the rest of the selection exactly as it is.
- If they just said new text rather than an instruction (a number, a word, a sentence), it replaces the selection, formatted as LaTeX.
- Rewrites ("more formal", "shorter") use {style} in {language}.
- Never invent facts, numbers, citations or labels. Cite only keys under SOURCES; reference only labels under LABELS.
- "latex" is body content only: no preamble, no markdown fences, no new packages or \\newcommand. Use the author's macros where they fit.

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
    selected: Block | None = None  # text the speaker selected in the editor


def document_language(idx: ProjectIndex, configured: str = "auto") -> str:
    code = idx.conventions.language if configured == "auto" else configured
    return LANGUAGE_NAMES.get(code, code)


def system_prompt(
    idx: ProjectIndex, style: str = "academic", language: str = "auto", mode: str = "converse"
) -> str:
    """mode: converse (listen/watch: the model decides what the speech is),
    dictate (always written) or command (an instruction for the selection)."""
    rules = "\n".join(f"- {r}" for r in idx.conventions.rules())
    if mode == "dictate":
        return DICTATE_SYSTEM.format(rules=rules)
    sample = ""
    if idx.style.paragraphs:
        sample = "\nSample paragraph from the thesis (match its tone, not its content):\n"
        sample += idx.style.paragraphs[0].text
    template = COMMAND_SYSTEM if mode == "command" else SYSTEM
    return template.format(
        style=STYLE_TEXT.get(style, STYLE_TEXT["academic"]),
        language=document_language(idx, language),
        rules=rules,
        sample=sample,
    )


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


SPEECH_HEADINGS = {"converse": "SPEECH", "dictate": "DICTATION", "command": "INSTRUCTION"}


def user_message(idx: ProjectIndex, ctx: PromptContext, mode: str = "converse") -> str:
    parts = []
    if ctx.nearby:
        parts.append("TEXT BEFORE CURSOR:\n" + ctx.nearby)
    if ctx.selected:
        parts.append(
            f'SELECTED TEXT [{ctx.selected.id}] (the speaker selected this; "this"/"it" means'
            f" it; to change it, replace block {ctx.selected.id}):\n"
            + _clip(ctx.selected.latex, BLOCK_MAX_CHARS * 2)
        )
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
    parts.append(f"{SPEECH_HEADINGS[mode]}:\n" + ctx.pending)
    if ctx.force:
        parts.append('The speaker has paused. Do not answer "wait".')
    return "\n\n".join(parts)


def build_messages(
    idx: ProjectIndex,
    ctx: PromptContext,
    style: str = "academic",
    language: str = "auto",
    mode: str = "converse",
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(idx, style, language, mode)},
        {"role": "user", "content": user_message(idx, ctx, mode)},
    ]


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count (≈4 chars/token for English, fewer for Greek) for budgeting and tests."""
    total = 0
    for m in messages:
        text = m["content"]
        non_ascii = sum(1 for c in text if ord(c) > 127)
        total += (len(text) - non_ascii) / 4 + non_ascii / 2
    return int(total)
