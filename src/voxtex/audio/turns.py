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


class TurnTimer:
    """Decides when the pending speech goes to the model.

    - A finished-looking sentence (ends with punctuation, 6+ words) is decided as soon
      as its utterance ends: that's what keeps latency near the 2 s target.
    - Anything else waits for `decision_pause` of silence, then is forced.
    - Speech pending longer than `max_pending_s` is forced too.
    - After the model answered "wait", it isn't asked again until the long pause.
    """

    def __init__(self, decision_pause: float = 2.5, max_pending_s: float = 20.0):
        self.decision_pause = decision_pause
        self.max_pending_s = max_pending_s
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
        if not pending or pending_since is None:
            return False, False
        if now - pending_since >= self.max_pending_s:
            return True, True
        if speaking:
            return False, False
        if now - last_voice >= self.decision_pause:
            return True, True
        early = (
            pending != self.waited_on
            and _END_OF_SENTENCE.search(pending) is not None
            and len(pending.split()) >= MIN_WORDS_TO_DECIDE_EARLY
        )
        return early, False

    def model_waited(self, pending: str) -> None:
        self.waited_on = pending
