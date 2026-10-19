"""`voxtex doctor`: check the project, AI providers, speech-to-text, microphone and TeX.

Every check says what it found and, when something is wrong, the one command or
setting that fixes it. Nothing here writes to the thesis.
"""

import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.markup import escape

from voxtex import __version__
from voxtex.config import Config, ProviderConfig, load_config

SPEED_CLIP_S = 5.0
SLOW_RTF = 0.3  # transcription taking >30% of the audio's length feels laggy in `listen`
QUIET_PEAK = 0.01  # RMS: below this the mic is effectively silent

KEY_URLS = {
    "GROQ_API_KEY": "https://console.groq.com/keys",
    "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    "OPENROUTER_API_KEY": "https://openrouter.ai/settings/keys",
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
}
OLLAMA_INSTALL = "curl -fsSL https://ollama.com/install.sh | sh"
WHISPER_SIZES_MB = {"small": 480, "medium": 1500, "large-v3-turbo": 1600, "large-v3": 3100}
SMALLER_MODEL = {"large-v3": "large-v3-turbo", "large-v3-turbo": "medium", "medium": "small"}

ICONS = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]",
         "info": "[dim]·[/dim]"}  # fmt: skip


@dataclass
class Check:
    status: str  # ok | warn | fail | info
    label: str
    detail: str = ""
    fix: str = ""


def print_check(console: Console, c: Check) -> None:
    detail = f"  [dim]{escape(c.detail)}[/dim]" if c.detail else ""
    console.print(f"  {ICONS[c.status]} {escape(c.label)}{detail}")
    if c.fix:
        console.print(f"      [cyan]→ {escape(c.fix)}[/cyan]")


# --- AI providers --------------------------------------------------------------------------------


def list_models(p: ProviderConfig, key: str, timeout: float = 6.0) -> list[str]:
    import openai

    client = openai.OpenAI(
        base_url=p.base_url, api_key=key or "not-needed", timeout=timeout, max_retries=0
    )
    return [m.id for m in client.models.list().data]


def _same_model(wanted: str, offered: str) -> bool:
    """Gemini lists `models/<id>`; Ollama adds `:latest` to untagged names."""

    def norm(m: str) -> str:
        m = m.removeprefix("models/")
        return m if ":" in m or "/" in m else f"{m}:latest"

    return norm(wanted) == norm(offered) or wanted == offered


def is_local(p: ProviderConfig) -> bool:
    return any(h in p.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))


def check_provider(
    p: ProviderConfig, lister: Callable[[ProviderConfig, str], list[str]] = list_models
) -> Check:
    label = f"{p.name}: {p.model}"
    key = p.resolve_key()
    if key is None:
        url = KEY_URLS.get(p.api_key_env, "")
        return Check(
            "info", p.name, f"skipped, no {p.api_key_env}",
            f"free key at {url}, then voxtex init" if url else "",
        )  # fmt: skip
    import openai

    start = time.monotonic()
    try:
        models = lister(p, key)
    except openai.AuthenticationError:
        return Check(
            "fail", label, f"{p.api_key_env} was rejected", "check the key, or run voxtex init"
        )
    except openai.APIConnectionError:
        if is_local(p):
            if shutil.which("ollama"):
                return Check("warn", label, "Ollama isn't running", "ollama serve")
            return Check("warn", label, "Ollama isn't installed", OLLAMA_INSTALL)
        return Check("warn", label, f"cannot reach {p.base_url}")
    except Exception as e:  # odd endpoints: report, never crash
        return Check("warn", label, f"could not list models ({type(e).__name__}: {e})")
    ms = round((time.monotonic() - start) * 1000)
    if any(_same_model(p.model, m) for m in models):
        return Check("ok", label, f"key works, model available ({ms} ms)")
    if is_local(p):
        pulled = ", ".join(models) or "none"
        return Check(
            "warn", label, f"not pulled (you have: {pulled})",
            f"ollama pull {p.model}, or voxtex init to use one you have",
        )  # fmt: skip
    close = get_close_matches(p.model, [m.removeprefix("models/") for m in models], n=3)
    hint = f"; similar: {', '.join(close)}" if close else ""
    return Check(
        "fail", label, f"this provider no longer offers the model{hint}",
        f'set model = "..." under [providers.{p.name}] in ~/.config/voxtex/config.toml',
    )  # fmt: skip


def check_ai(config: Config, lister=list_models) -> list[Check]:
    checks, working = [], []
    for p in config.chain_providers():
        c = check_provider(p, lister)
        checks.append(c)
        if c.status == "ok":
            working.append(p.name)
    if working:
        private = [p.name for p in config.chain_providers(private_only=True) if p.name in working]
        tail = "" if private else "; nothing private for --private"
        checks.append(Check("ok", "chain: " + " → ".join(working), tail.lstrip("; ")))
    else:
        checks.append(Check(
            "fail", "no AI provider works",
            "voxtex needs a free key or a running Ollama", "voxtex init",
        ))  # fmt: skip
    return checks


# --- speech, microphone, TeX ---------------------------------------------------------------------


def whisper_cached(model: str) -> bool:
    try:
        from faster_whisper.utils import download_model

        download_model(model, local_files_only=True)
        return True
    except Exception:
        return False


def record_clip(config: Config, console: Console, seconds: float = SPEED_CLIP_S):
    """Record from the configured mic. Returns (Check, audio or None)."""
    from voxtex.audio.capture import MicError, Segmenter, open_microphone, webrtc_vad

    try:
        mic = open_microphone(Segmenter(webrtc_vad()), config.mic)
        mic.recording = []
        mic.start()
    except MicError as e:
        return Check(
            "fail", "microphone", str(e), "voxtex listen --list-mics, then --mic NAME"
        ), None
    with console.status(
        f"Recording {seconds:.0f} s: say a sentence, in any language you dictate in…"
    ):
        time.sleep(seconds)
    mic.stop()
    frames = mic.recording or []
    if not frames:
        return Check("fail", "microphone", "no audio arrived", "voxtex listen --list-mics"), None
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    where = f"{mic.backend}" + (f", {config.mic}" if config.mic else ", default input")
    if mic.peak < QUIET_PEAK:
        return Check(
            "warn", "microphone", f"{where}: silent (peak {mic.peak:.3f})",
            "wrong input? voxtex listen --list-mics, then --mic NAME",
        ), audio  # fmt: skip
    return Check("ok", "microphone", f"{where}: peak {mic.peak:.2f}"), audio


def check_speech(
    config: Config, console: Console, audio: np.ndarray | None, quick: bool, languages=()
) -> list[Check]:
    from voxtex.audio.stt import build_transcriber, choose_stt, cuda_available

    checks = []
    gpu = cuda_available()
    checks.append(Check("ok" if gpu else "info", "NVIDIA GPU", "found" if gpu else "not found"))
    choice = choose_stt(config)
    checks.append(Check("ok", f"speech-to-text: {choice.label}", choice.reason))
    if choice.kind != "local":
        p = config.providers.get("groq-gptoss") if choice.kind == "groq" else None
        if p is not None and p.resolve_key() is None:
            checks.append(
                Check("fail", "cloud speech-to-text", f"no {p.api_key_env}", "voxtex init")
            )
        return checks
    if not whisper_cached(choice.model):
        size = WHISPER_SIZES_MB.get(choice.model)
        checks.append(Check(
            "info", f"Whisper {choice.model}",
            "not downloaded yet" + (f" (~{size} MB, once, on the first listen)" if size else ""),
        ))  # fmt: skip
        return checks
    if quick:
        return checks
    t = build_transcriber(choice, config, languages)
    with console.status(f"Loading {choice.model}…"):
        t.load()
    clip = audio if audio is not None else np.zeros(int(SPEED_CLIP_S * 16000), np.float32)
    result = t.transcribe(clip)
    t.unload()
    rtf = result.seconds / max(result.audio_seconds, 0.1)
    detail = f"{result.seconds:.2f} s for {result.audio_seconds:.1f} s of audio (RTF {rtf:.2f})"
    if audio is None:
        detail += "; a silent clip, so speech will take longer"
    if rtf <= SLOW_RTF:
        checks.append(Check("ok", "transcription speed", detail))
    else:
        smaller = SMALLER_MODEL.get(choice.model)
        fix = (
            f'set local_model = "{smaller}" under [stt]' if smaller
            else "use Groq's Whisper (free key): mode = \"groq\" under [stt]"
        )  # fmt: skip
        checks.append(Check("warn", "transcription speed", detail + ", slow for listening", fix))
    if audio is not None:
        heard = result.text or "(nothing)"
        checks.append(Check("info", "heard", f"“{heard}”"))
    return checks


def check_tex(idx) -> Check:
    from voxtex.validate.compile import pick_engine

    if idx is not None:
        engine = pick_engine(idx)
    else:
        engine = next((e for e in ("pdflatex", "xelatex") if shutil.which(e)), None)
    if engine:
        return Check("ok", f"TeX: {engine}", "unusual LaTeX is test-compiled before writing")
    return Check(
        "info", "TeX: not found", "writes are still checked, just not compiled",
        "sudo apt install texlive-latex-extra texlive-lang-greek  (optional)",
    )  # fmt: skip


def check_dictation() -> list[Check]:
    if shutil.which("voxtype"):
        return [Check(
            "info", "voxtype found", "voxtex watch turns its %> lines into LaTeX",
            "pick a multilingual model for Greek: voxtype setup model",
        )]  # fmt: skip
    return []


# --- the command ---------------------------------------------------------------------------------


def run_doctor(console: Console, path: Path, quick: bool, lister=list_models) -> int:
    from voxtex.project.discover import DiscoveryError
    from voxtex.project.index import load_index
    from voxtex.project.report import summary_lines

    sections: list[tuple[str, list[Check]]] = []

    def section(title: str, checks: list[Check]) -> None:
        console.print(f"[bold]{title}[/bold]")
        for c in checks:
            print_check(console, c)
        sections.append((title, checks))

    py = ".".join(map(str, sys.version_info[:3]))
    console.print(f"voxtex {__version__} · Python {py}")

    idx = None
    try:
        idx, _ = load_index(path)
        project = [Check("ok", line) for line in summary_lines(idx)]
    except DiscoveryError as e:
        project = [Check("info", "no LaTeX project here", str(e))]
    section("Project", project)

    config = load_config(idx.root_path if idx else None)
    sources = ", ".join(config.sources) or "built-in defaults only"
    settings = [Check("ok" if config.sources else "info", "config", sources)]
    if not config.sources:
        settings[0].fix = "voxtex init"
    if config.privacy == "strict":
        settings.append(Check("info", "privacy: strict", "only private (local) providers are used"))
    section("Settings", settings)

    with console.status("Checking AI providers…"):
        ai = check_ai(config, lister)
    section("AI", ai)

    audio = None
    speech: list[Check] = []
    if not quick and sys.stdin.isatty():
        mic, audio = record_clip(config, console)
        speech.append(mic)
    elif not quick:
        speech.append(Check("info", "microphone", "skipped (not a terminal)"))
    from voxtex.listenmode import stt_languages

    langs = stt_languages(idx) if idx else ("el", "en")
    speech += check_speech(config, console, audio, quick, langs)
    section("Speech", speech)

    section("Other", [check_tex(idx), *check_dictation()])

    failed = [c for _, cs in sections for c in cs if c.status == "fail"]
    warned = [c for _, cs in sections for c in cs if c.status == "warn"]
    if failed:
        console.print(f"[red]{len(failed)} problem(s)[/red] to fix before voxtex can work fully.")
        return 1
    if warned:
        console.print(f"[yellow]{len(warned)} warning(s)[/yellow]; voxtex works.")
    else:
        console.print("[green]All good.[/green]")
    return 0
