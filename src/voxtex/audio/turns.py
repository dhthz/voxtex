"""When to call the model (addendum §5.1–5.2): skip filler, decide on pauses, not per word."""

import re
import unicodedata

_FILLER_WORDS = {
    # English
    "um", "umm", "uh", "uhh", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mmm", "eh", "oh",
    # Greek (accents stripped)
    "εμ", "εμμ", "εεε", "εε", "ε", "χμ", "χμμ", "μμ", "αα", "ααα", "οκ",
}  # fmt: skip
_FILLER_PHRASES = {
    "let me think", "let me see", "wait", "hold on", "one second", "one moment",
    "ας δουμε", "για να δω", "μια στιγμη", "περιμενε", "ας σκεφτω", "λοιπον",
    # Whisper's usual inventions on near-silence
    "thank you", "thanks for watching", "thank you for watching", "you", "bye",
    "ευχαριστω", "ευχαριστω πολυ", "υποτιτλοι authorwave", "υποτιτλισμος authorwave",
}  # fmt: skip
# Whisper's stock phrases from subtitled training videos: real dictation never contains them.
_HALLUCINATION_MARKS = ("authorwave", "υποτιτλ", "subtitles by", "izlediginiz icin", "abone ol")
_ONE_PHRASE_LOOPING = re.compile(r"^(.{1,40}?)(?: \1)+$")
_END_OF_SENTENCE = re.compile(r"[.!?;·…»\"']\s*$")
MIN_WORDS_TO_DECIDE_EARLY = 6


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", text))


def is_filler(text: str, prompt: str = "") -> bool:
    """True for transcripts that are only hesitation, or Whisper hallucinations
    (including Whisper echoing its own prompt hint back on noise)."""
    norm = _norm(text)
    if not norm:
        return True
    if prompt and set(norm.split()) <= set(_norm(prompt).split()):
        return True
    if norm in _FILLER_PHRASES:
        return True
    return all(w in _FILLER_WORDS for w in norm.split())


def is_garbled(text: str) -> bool:
    """True for Whisper output that is noise rather than anything said: letters from scripts
    nobody spoke (speech is only ever Greek or English here), stock subtitle phrases, or one
    phrase looping ("All right. All right. All right.")."""
    foreign = sum(1 for c in text if c.isalpha() and not _greek_or_latin(c))
    if foreign >= 2:
        return True
    norm = _norm(text)
    if any(mark in norm for mark in _HALLUCINATION_MARKS):
        return True
    return bool(_ONE_PHRASE_LOOPING.match(norm))


_UNDO = re.compile(
    r"^(?:undo|undo (?:that|it|the last (?:change|edit|one))|αναιρεση|αναιρεσε(?: το| αυτο)?)"
    r"(?: (twice|two times|three times|δυο φορες|τρεις φορες|[23] φορες))?$"
)
_TIMES = {"twice": 2, "two times": 2, "δυο φορες": 2, "2 φορες": 2,
          "three times": 3, "τρεις φορες": 3, "3 φορες": 3}  # fmt: skip


def spoken_undo(text: str) -> int:
    """How many edits to undo if the whole recording was "undo" ("undo twice", "αναίρεση"),
    else 0. Only the whole recording counts: "undo" inside a sentence is dictation."""
    m = _UNDO.match(_norm(text))
    return (_TIMES.get(m.group(1), 1) if m.group(1) else 1) if m else 0


def _greek_or_latin(c: str) -> bool:
    return unicodedata.name(c, "").startswith(("LATIN", "GREEK"))


class TurnTimer:
    """Decides when the pending speech goes to the model.

    - A finished-looking sentence (ends with punctuation, 6+ words) is decided after
      `early_pause` of silence. Not sooner: Whisper puts a full stop after every chunk,
      so a breath in the middle of a sentence looks like its end.
    - Anything else waits for `decision_pause` of silence, then is forced.
    - Speech pending longer than `max_pending_s` is forced at the next pause, never
      while you are still talking.
    - After the model answered "wait", it isn't asked again until the long pause.
    """

    def __init__(
        self, decision_pause: float = 2.5, max_pending_s: float = 20.0, early_pause: float = 1.2
    ):
        self.decision_pause = decision_pause
        self.max_pending_s = max_pending_s
        self.early_pause = min(early_pause, decision_pause)
        self.waited_on = ""  # pending text the model already said "wait" to

    def check(
        self,
        pending: str,
        pending_since: float | None,
        last_voice: float,
        speaking: bool,
        now: float,
    ) -> tuple[bool, bool]:
        """(decide now?, force a decision?)"""
        if not pending or pending_since is None or speaking:
            return False, False
        silence = now - last_voice
        if silence >= self.decision_pause or now - pending_since >= self.max_pending_s:
            return True, True
        early = (
            silence >= self.early_pause
            and pending != self.waited_on
            and _END_OF_SENTENCE.search(pending) is not None
            and len(pending.split()) >= MIN_WORDS_TO_DECIDE_EARLY
        )
        return early, False

    def model_waited(self, pending: str) -> None:
        self.waited_on = pending
