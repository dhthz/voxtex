"""`voxtex listen`: microphone → speech-to-text → engine → validated LaTeX in the file.

Threads: the microphone thread cuts utterances, a speech-to-text thread transcribes
them in order, and the main thread decides when to call the model and writes.
Speech keeps being captured and transcribed while the model is thinking.
"""

import json
import queue
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.live import Live

from voxtex.app import open_workspace
from voxtex.audio.capture import MicError, Segmenter, Utterance, open_microphone, webrtc_vad
from voxtex.audio.stt import SttChoice, Transcriber, Transcript, build_transcriber, choose_stt
from voxtex.audio.turns import TurnTimer, is_filler
from voxtex.engine.decide import Wait
from voxtex.project.index import STATE_DIR, ProjectIndex, _is_spoken_term
from voxtex.providers import ChainExhausted
from voxtex.ui import UI, mic_bar, project_row, session_summary, target_row

HINT_SURNAMES = 25
HINT_TERMS = 30
_PROMPT_LEAD = {"el": "Διπλωματική εργασία.", "en": "Thesis notes."}


def whisper_prompt(idx: ProjectIndex) -> str:
    """A hint for Whisper: a phrase in the document's language, the thesis's own technical
    terms (what it wraps in \\en{...}), and the most cited surnames."""
    surnames = Counter(e.authors[0] for e in idx.bib if e.authors)
    names = [n for n, _ in surnames.most_common() if _is_spoken_term(n)][:HINT_SURNAMES]
    terms = [t for t in idx.latin_terms[:HINT_TERMS] if t not in names]
    lead = _PROMPT_LEAD.get(idx.conventions.language, "")
    return f"{lead} {', '.join(terms + names)}".strip()


def stt_languages(idx: ProjectIndex) -> tuple[str, ...]:
    """Languages speech can be in: the document's, its other declared ones, and English."""
    from voxtex.project.conventions import _LANG_CODES

    codes = [idx.conventions.language]
    codes += [_LANG_CODES[name] for name in idx.conventions.languages if name in _LANG_CODES]
    return tuple(dict.fromkeys([*codes, "en"]))


@dataclass
class Heard:
    text: str
    utterance: Utterance
    transcript: Transcript | None


class SpeechToText(threading.Thread):
    """Transcribes utterances in order. Falls back to local CPU Whisper if the cloud fails."""

    def __init__(
        self,
        transcriber: Transcriber,
        prompt: str,
        fallback: Transcriber | None = None,
        root: Path | None = None,
    ):
        super().__init__(daemon=True)
        self.root = root  # if set, every transcript is logged to .voxtex/timings.log
        self.transcriber = transcriber
        self.fallback = fallback
        self.prompt = prompt
        self.inbox: queue.Queue[Utterance | None] = queue.Queue()
        self.outbox: queue.Queue[Heard] = queue.Queue()
        self.busy = False
        self.notice = ""

    def stop(self) -> None:
        self.inbox.put(None)
        self.join(timeout=10)

    def run(self) -> None:
        while True:
            u = self.inbox.get()
            if u is None:
                return
            self.busy = True
            try:
                t = self.transcriber.transcribe(u.audio, self.prompt)
            except Exception as e:
                if self.fallback is None:
                    self.outbox.put(Heard("", u, None))
                    self.notice = f"speech-to-text failed: {e}"
                    self.busy = False
                    continue
                self.notice = f"↪ speech-to-text switched to {self.fallback.name} ({e})"
                self.transcriber, self.fallback = self.fallback, None
                t = self.transcriber.transcribe(u.audio, self.prompt)
            if self.root is not None:
                _log(self.root, {
                    "ts": round(time.time(), 3), "kind": "stt", "backend": t.backend,
                    "language": t.language, "audio_s": round(t.audio_seconds, 2),
                    "latency_ms": round(t.seconds * 1000), "text": t.text,
                })  # fmt: skip
            self.outbox.put(Heard(t.text, u, t))
            self.busy = False


def _log(root: Path, entry: dict) -> None:
    p = root / STATE_DIR / "timings.log"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run_listen(
    console: Console,
    path: Path,
    private: bool,
    provider: str | None,
    file_opt: Path | None,
    dry_run: bool,
    stt_mode: str | None,
    mic: str | None,
    verbose: bool = False,
) -> int:
    interactive = sys.stdin.isatty()
    ui = UI(console, verbose)
    ws = open_workspace(console, path, private, provider, file_opt, dry_run, interactive)
    if ws is None:
        return 1
    idx, config, chain, engine, location, session = (
        ws.idx,
        ws.config,
        ws.chain,
        ws.engine,
        ws.location,
        ws.session,
    )
    root = ws.root

    choice: SttChoice = choose_stt(config, private, stt_mode)
    transcriber = build_transcriber(choice, config, stt_languages(idx))
    fallback = None
    if choice.kind != "local":
        fallback = build_transcriber(SttChoice("local", "small", "cpu"), config)

    wanted = mic if mic is not None else config.mic
    try:
        segmenter = Segmenter(webrtc_vad(), pause_s=config.utterance_pause)
        microphone = open_microphone(segmenter, wanted)
    except MicError as e:
        ui.error(f"{e}. Try voxtex listen --list-mics")
        return 1

    if choice.kind == "local":
        with console.status(
            f"[grey50]Loading {choice.model} (downloads once on first use)…[/grey50]",
            spinner="dots", spinner_style="#d77757",
        ):  # fmt: skip
            transcriber.load()
    stt = SpeechToText(transcriber, whisper_prompt(idx), fallback, root)
    stt.start()
    try:
        microphone.start()
    except MicError as e:
        ui.error(str(e))
        return 1

    lang = idx.conventions.language if config.language == "auto" else config.language
    where = target_row(location.rel, location.line, location.found) if location else "(no target)"
    if dry_run:
        where += "  · dry run, nothing is written"
    ui.banner(
        [
            ("thesis", project_row(idx)),
            ("writing", where),
            ("AI", " → ".join(p.name for p in chain.usable()) + (" (private)" if private else "")),
            ("speech", f"{choice.label} · {lang} · {config.style}"),
        ],
        hint="Speak freely; pause and it writes. Ctrl+C to stop.",
    )

    timer = TurnTimer(config.decision_pause, config.max_pending_seconds)
    latencies: list[float] = []
    writes = 0
    speech_end = 0.0  # end of the last utterance that's part of the pending speech
    state = {"what": "listening"}

    verbs = {"listening": "Listening", "hearing": "Hearing", "transcribing": "Transcribing",
             "thinking": "Thinking", "writing": "Writing"}  # fmt: skip

    def status():
        what = state["what"]
        if what == "listening" and microphone.speaking:
            what = "hearing"
        elif what == "listening" and (stt.busy or not microphone.utterances.empty()):
            what = "transcribing"
        prov = chain.active or (chain.usable()[0].name if chain.usable() else "no AI")
        left = chain.quota.remaining_tokens(prov)
        quota = f" · ~{left // 1000}K tok left" if left and left >= 1000 else ""
        return ui.status(
            verbs[what],
            f"{prov}{quota}",
            speech=engine.buffer.pending_text,
            hints=f"{mic_bar(microphone.level)} mic · ctrl+c to stop",
            active=what != "listening",
        )

    def decide(force: bool) -> None:
        nonlocal speech_end, writes
        said = engine.buffer.pending_text
        state["what"] = "thinking"
        try:
            step = engine.feed("", force=force)
        except ChainExhausted as e:
            ui.error(f"{e} (speech kept)")
            state["what"] = "listening"
            return
        state["what"] = "listening"
        if step is None:
            return
        if isinstance(step.decision, Wait):
            timer.model_waited(engine.buffer.pending_text)
            return
        ui.heard(said)
        state["what"] = "writing"
        ui.step(step, session, rel=location.rel if location else "")
        state["what"] = "listening"
        written = time.monotonic()
        if step.block is not None:
            writes += 1
            latency = written - speech_end
            latencies.append(latency)
            if verbose:
                ui.more(f"{latency:.1f}s from end of speech to file", "grey37")
            _log(root, {
                "ts": round(time.time(), 3), "kind": "utterance",
                "end_to_write_ms": round(latency * 1000), "llm_ms": round(step.latency_s * 1000),
                "validate_ms": round(step.report.seconds * 1000) if step.report else 0,
                "retried": step.retried, "provider": step.provider,
                "input_tokens": step.input_tokens, "output_tokens": step.output_tokens,
            })  # fmt: skip

    try:
        # get_renderable: the spinner keeps moving while the main thread waits on the model.
        with Live(get_renderable=status, console=console, refresh_per_second=10, transient=True):
            while True:
                while not microphone.utterances.empty():
                    stt.inbox.put(microphone.utterances.get())
                try:
                    heard = stt.outbox.get(timeout=0.1)
                except queue.Empty:
                    heard = None
                if stt.notice:
                    ui.notice(stt.notice)
                    stt.notice = ""
                if heard is not None and heard.transcript is not None:
                    t = heard.transcript
                    _log(root, {
                        "ts": round(time.time(), 3), "kind": "stt", "backend": t.backend,
                        "stt_ms": round(t.seconds * 1000),
                        "audio_ms": round(t.audio_seconds * 1000),
                    })  # fmt: skip
                    if heard.text and not is_filler(heard.text, stt.prompt):
                        engine.buffer.add_utterance(heard.text, heard.utterance.started)
                        speech_end = heard.utterance.ended
                now = time.monotonic()
                still_talking = microphone.speaking or stt.busy or not stt.inbox.empty()
                go, force = timer.check(
                    engine.buffer.pending_text,
                    engine.buffer.pending_since,
                    max(segmenter.last_voice, speech_end),
                    still_talking,
                    now,
                )
                if go:
                    decide(force)
    except KeyboardInterrupt:
        pass
    finally:
        microphone.stop()
        stt.stop()
    if engine.buffer.pending:
        decide(force=True)
    # Free the GPU model from the main thread; otherwise CUDA can abort at exit
    # ("terminate called without an active exception").
    for t in (stt.transcriber, stt.fallback):
        if hasattr(t, "unload"):
            t.unload()

    extra = ""
    if latencies:
        p90 = sorted(latencies)[int(0.9 * (len(latencies) - 1))]
        extra = f"speech→file median {statistics.median(latencies):.1f}s, p90 {p90:.1f}s"
    ui.done(session_summary(chain, writes, extra, verbose))
    return 0
