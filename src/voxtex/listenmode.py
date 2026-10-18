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
from rich.markup import escape
from rich.text import Text

from voxtex.app import build_chain, choose_target, location_in, offer_gitignore, open_project
from voxtex.audio.capture import (
    MicError,
    Microphone,
    Segmenter,
    Utterance,
    pick_device,
    prefer_parec,
    webrtc_vad,
)
from voxtex.audio.stt import SttChoice, Transcriber, Transcript, build_transcriber, choose_stt
from voxtex.audio.turns import TurnTimer, is_filler
from voxtex.engine.decide import Wait
from voxtex.engine.engine import Engine
from voxtex.io.writer import Writer
from voxtex.project.index import STATE_DIR, ProjectIndex, _is_spoken_term
from voxtex.providers import ChainExhausted
from voxtex.session import Session
from voxtex.textmode import print_step

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

    def __init__(self, transcriber: Transcriber, prompt: str, fallback: Transcriber | None = None):
        super().__init__(daemon=True)
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
) -> int:
    interactive = sys.stdin.isatty()
    opened = open_project(console, path)
    if opened is None:
        return 1
    idx, config = opened
    chain = build_chain(console, idx, config, private, provider)
    if chain is None:
        return 1
    root = idx.root_path

    choice: SttChoice = choose_stt(config, private, stt_mode)
    console.print(f"[green]✓[/green] Speech: {choice.label} [dim]({choice.reason})[/dim]")
    transcriber = build_transcriber(choice, config, stt_languages(idx))
    fallback = None
    if choice.kind != "local":
        fallback = build_transcriber(SttChoice("local", "small", "cpu"), config)

    session = None
    if dry_run:
        target = (root / file_opt) if file_opt else None
        location = location_in(root, target) if target else None
    else:
        target = choose_target(console, idx, file_opt, interactive)
        if target is None:
            return 1
        offer_gitignore(console, root, interactive)
        writer = Writer(root)
        writer.ensure_marker(target)
        location = location_in(root, target)
    engine = Engine(idx, chain, config, location)
    if not dry_run:
        session = Session(engine, writer, target)

    wanted = mic if mic is not None else config.mic
    try:
        segmenter = Segmenter(webrtc_vad(), pause_s=config.utterance_pause)
        if prefer_parec() and not wanted.isdigit():
            microphone = Microphone(segmenter, backend="parec", source=wanted)
        else:
            microphone = Microphone(segmenter, pick_device(wanted))
    except MicError as e:
        console.print(f"[red]✗[/red] {e}. Try [bold]voxtex listen --list-mics[/bold].")
        return 1

    if choice.kind == "local":
        with console.status(f"Loading {choice.model} (downloads once on first use)…"):
            transcriber.load()
    stt = SpeechToText(transcriber, whisper_prompt(idx), fallback)
    stt.start()
    try:
        microphone.start()
    except MicError as e:
        console.print(f"[red]✗[/red] {e}")
        return 1

    lang = idx.conventions.language if config.language == "auto" else config.language
    where = f"{location.rel} at %> HERE" if location else "(dry run)"
    console.print(f"🎙  Listening → {where}   (style: {config.style}, lang: {lang})")
    console.print("[dim]   Speak freely; pause to let it write. Ctrl+C to stop.[/dim]")

    timer = TurnTimer(config.decision_pause, config.max_pending_seconds)
    latencies: list[float] = []
    speech_end = 0.0  # end of the last utterance that's part of the pending speech
    state = {"what": "listening"}

    def status() -> Text:
        what = state["what"]
        if what == "listening" and microphone.speaking:
            what = "hearing"
        elif what == "listening" and (stt.busy or not microphone.utterances.empty()):
            what = "transcribing"
        bar = "▁▂▃▄▅▆▇█"[min(7, int(microphone.level * 8))]
        prov = chain.active or (chain.usable()[0].name if chain.usable() else "no AI")
        left = chain.quota.remaining_tokens(prov)
        quota = f" · ~{left // 1000}K tok left" if left and left >= 1000 else ""
        pending = engine.buffer.pending_text
        tail = f" · “{pending[-60:]}”" if pending else ""
        return Text(f"🎙 {bar} {what} · {prov}{quota}{tail}", style="dim")

    def decide(force: bool) -> None:
        nonlocal speech_end
        state["what"] = "thinking"
        try:
            step = engine.feed("", force=force)
        except ChainExhausted as e:
            console.print(f"[red]✗ {escape(str(e))}[/red] (speech kept)")
            state["what"] = "listening"
            return
        state["what"] = "listening"
        if step is None:
            return
        if isinstance(step.decision, Wait):
            timer.model_waited(engine.buffer.pending_text)
            return
        print_step(console, step, show_prompt=False)
        written = time.monotonic()
        if session:
            state["what"] = "writing"
            try:
                for n in session.write(step):
                    console.print(f"[cyan]{escape(n)}[/cyan]")
            except OSError as e:
                console.print(f"[red]✗ could not write: {escape(str(e))}[/red]")
            written = time.monotonic()
            state["what"] = "listening"
        if step.block is not None:
            latency = written - speech_end
            latencies.append(latency)
            console.print(f"[dim]  ⏱ {latency:.1f}s from end of speech to file[/dim]")
            _log(root, {
                "ts": round(time.time(), 3), "kind": "utterance",
                "end_to_write_ms": round(latency * 1000), "llm_ms": round(step.latency_s * 1000),
                "validate_ms": round(step.report.seconds * 1000) if step.report else 0,
                "retried": step.retried, "provider": step.provider,
                "input_tokens": step.input_tokens, "output_tokens": step.output_tokens,
            })  # fmt: skip

    try:
        with Live(status(), console=console, refresh_per_second=8, transient=True) as live:
            while True:
                while not microphone.utterances.empty():
                    stt.inbox.put(microphone.utterances.get())
                try:
                    heard = stt.outbox.get(timeout=0.1)
                except queue.Empty:
                    heard = None
                if stt.notice:
                    console.print(f"[cyan]{escape(stt.notice)}[/cyan]")
                    stt.notice = ""
                if heard is not None and heard.transcript is not None:
                    t = heard.transcript
                    _log(root, {
                        "ts": round(time.time(), 3), "kind": "stt", "backend": t.backend,
                        "stt_ms": round(t.seconds * 1000),
                        "audio_ms": round(t.audio_seconds * 1000),
                    })  # fmt: skip
                    if heard.text and not is_filler(heard.text, stt.prompt):
                        console.print(f"[dim]  “{escape(heard.text)}”[/dim]")
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
                live.update(status())
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

    t = chain.totals
    if t.calls:
        console.print(
            f"[dim]Session: {t.calls} calls · {t.input_tokens} in / {t.output_tokens} out tokens"
            f" (avg {t.input_tokens // t.calls} in/call)[/dim]"
        )
    if latencies:
        p90 = sorted(latencies)[int(0.9 * (len(latencies) - 1))]
        console.print(
            f"[dim]Latency end of speech → file: median {statistics.median(latencies):.1f}s,"
            f" p90 {p90:.1f}s over {len(latencies)} writes[/dim]"
        )
    return 0
