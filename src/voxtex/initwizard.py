"""`voxtex init`: what this machine has, privacy, a free AI key (tested right away), speech,
the local fallback and the microphone.

Settings go to ~/.config/voxtex/config.toml and keys to ~/.config/voxtex/.env (mode 600),
so they apply to every thesis. Every step can be skipped; running it again keeps what's
already there unless you change it.
"""

import os
import re
import shutil
import tomllib
from pathlib import Path

import tomli_w
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from voxtex.config import ENV_FILE, GLOBAL_CONFIG, load_config
from voxtex.doctor import KEY_URLS, OLLAMA_INSTALL, Check, check_provider, list_models

CONFIG_FILE = "config.toml"
HEADER = (
    "# voxtex settings, written by `voxtex init`. Edit freely; see the README for all options.\n"
)
SMALL_MODEL_B = 7  # local models below ~7B parameters make many mistakes

# Cloud keys offered, in chain order: (env var, name, what it's good for)
CLOUD_KEYS = [
    ("GROQ_API_KEY", "Groq", "free, fastest; also cloud speech-to-text"),
    ("GEMINI_API_KEY", "Google Gemini", "free tier, a good second in the chain"),
    ("OPENROUTER_API_KEY", "OpenRouter", "free models, a last cloud fallback"),
]


class Prompts:
    """The questions, behind one small interface so tests can script the answers."""

    def __init__(self, console: Console):
        self.console = console

    def choice(self, question: str, options: list[tuple[str, str]], default: str) -> str:
        for i, (_, text) in enumerate(options, 1):
            self.console.print(f"  {i}. {text}")
        keys = [k for k, _ in options]
        n = Prompt.ask(
            question,
            choices=[str(i) for i in range(1, len(options) + 1)],
            default=str(keys.index(default) + 1),
            console=self.console,
        )
        return keys[int(n) - 1]

    def secret(self, question: str) -> str:
        return Prompt.ask(question, password=True, default="", show_default=False).strip()

    def confirm(self, question: str, default: bool = True) -> bool:
        return Confirm.ask(question, default=default, console=self.console)


# --- files ---------------------------------------------------------------------------------------


def read_config(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def write_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + tomli_w.dumps(data), encoding="utf-8")


def save_env_value(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file, replacing an existing line; the file is private (600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=")
    line = f"{key}={value}"
    for i, old in enumerate(lines):
        if pattern.match(old):
            lines[i] = line
            break
    else:
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _set(data: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    for p in parents:
        data = data.setdefault(p, {})
    data[last] = value


def model_size_b(name: str) -> float | None:
    """Parameter count from an Ollama name like `qwen2.5-coder:14b` or `llama3.2:3b`."""
    m = re.search(r"[:\-_](\d+(?:\.\d+)?)b\b", name.lower())
    return float(m.group(1)) if m else None


# --- the wizard ----------------------------------------------------------------------------------


def run_init(
    console: Console,
    ask: Prompts | None = None,
    lister=list_models,
    global_dir: Path | None = None,
    mic_test=None,
) -> int:
    """Look at the machine first, then ask only what it can't decide: privacy, an AI key,
    and (without a GPU) where speech runs. Enter always takes the suggestion."""
    ask = ask or Prompts(console)
    gdir = global_dir or GLOBAL_CONFIG
    cfg_path, env_path = gdir / CONFIG_FILE, gdir / ENV_FILE
    data = read_config(cfg_path)
    config = load_config(None, global_dir=gdir)
    from voxtex.audio.stt import cuda_available

    gpu = cuda_available()
    ollama = shutil.which("ollama") is not None
    cloud = [
        (env, name, why, p)
        for env, name, why in CLOUD_KEYS
        for p in [next((q for q in config.providers.values()
                        if q.api_key_env == env and not q.private), None)]
        if p is not None
    ]  # fmt: skip
    keys_set = [env for env, _, _, p in cloud if p.resolve_key()]

    console.print("[bold]voxtex setup[/bold] [dim]· Enter takes the suggestion[/dim]\n")
    _found(console, "GPU", gpu, "NVIDIA: speech runs here, fast and private",
           "none: speech is slower here, or runs on Groq")  # fmt: skip
    _found(console, "Ollama", ollama, "installed: a local AI for offline and --private",
           "not installed (optional)")  # fmt: skip
    _found(console, "AI keys", bool(keys_set), ", ".join(_key_name(e) for e in keys_set),
           "none yet")  # fmt: skip

    # 1. Privacy decides whether the cloud questions are asked at all.
    _step(console, 1, "Privacy")
    strict = ask.confirm(
        "  Keep everything on this machine? (local AI and speech only; for unpublished work)",
        data.get("privacy") == "strict",
    )
    data["privacy"] = "strict" if strict else "normal"

    # 2. AI: one key is enough; more are fallbacks.
    have_keys = list(keys_set)
    if not strict:
        _step(console, 2, "AI", "free keys; each is tested now")
        for n, (env, name, why, provider) in enumerate(cloud):
            if n == 1 and not ask.confirm("  Add fallback keys (Gemini, OpenRouter)?", False):
                break
            _ask_key(console, ask, env, name, why, provider, lister, env_path, have_keys)

    # 3. Speech: decided by the machine; asked only when there's a real choice.
    _step(console, 3, "Speech")
    if gpu or strict:
        speech = "local"
    elif "GROQ_API_KEY" in have_keys:
        cloud_ok = ask.confirm(
            "  No GPU: use Groq's Whisper (fast; your audio is sent to Groq)?", True
        )
        speech = "groq" if cloud_ok else "local"
    else:
        speech = "local"
    if data.get("stt", {}).get("mode") == "external" and speech == "local":
        speech = "external"  # set up for another dictation app (voxtex watch): keep it
    _set(data, "stt.mode", speech)
    console.print("  " + {
        "local": "Whisper on " + ("your GPU (large-v3-turbo)" if gpu else "the CPU (small)"),
        "groq": "Groq's Whisper (large-v3-turbo, in the cloud)",
        "external": "your own dictation app, with voxtex watch",
    }[speech])  # fmt: skip

    # 4. Local AI
    local = config.providers.get("local")
    if local is not None and (ollama or strict):
        _step(console, 4, "Local AI", "offline and --private")
        _setup_local(console, ask, local, data, lister)

    # 5. Microphone
    if speech != "external" and mic_test is not None:
        _step(console, 5, "Microphone", "say a few words")
        mic_test(console, data)

    write_config(cfg_path, data)
    final = load_config(None, global_dir=gdir)
    chain = [p.name for p in final.chain_providers() if p.resolve_key() is not None]
    console.print()
    console.print(f"[green]✓[/green] saved [dim]{cfg_path}[/dim]")
    console.print(f"  [dim]AI[/dim]       {' → '.join(chain) or 'none'}")
    console.print(f"  [dim]privacy[/dim]  {data['privacy']}")
    if not chain:
        console.print(
            "[yellow]![/yellow] No AI yet: add a key (voxtex init) or install Ollama. Until"
            " then, what you say goes in exactly as heard."
        )
    command = "voxtex watch" if speech == "external" else "voxtex"
    console.print(
        f"\nNext: [bold]cd[/bold] into your thesis folder and run [bold]{command}[/bold]."
        " [dim]voxtex doctor checks everything.[/dim]"
    )
    return 0


def _found(console: Console, what: str, ok: bool, yes: str, no: str) -> None:
    mark = "[green]✓[/green]" if ok else "[dim]·[/dim]"
    console.print(f"  {mark} {what:<8} [dim]{escape(yes if ok else no)}[/dim]")


def _step(console: Console, n: int, title: str, note: str = "") -> None:
    console.print(f"\n[bold]{n}  {title}[/bold]" + (f" [dim]· {note}[/dim]" if note else ""))


def _key_name(env: str) -> str:
    return next((name for e, name, _ in CLOUD_KEYS if e == env), env)


def _ask_key(console, ask, env, name, why, provider, lister, env_path, have_keys) -> None:
    if env in have_keys:
        if ask.confirm(f"  {name}: a key is already set. Keep it?", True):
            return
        have_keys.remove(env)
    console.print(f"  {name} ({why}): get a key at {KEY_URLS.get(env, '')}")
    for _attempt in range(2):
        key = ask.secret(f"  Paste your {env} (Enter to skip)")
        if not key:
            return
        result = check_provider(provider.model_copy(update={"api_key": key}), lister)
        _print_result(console, result)
        if result.status in ("ok", "warn"):  # warn: e.g. offline now; the key may be fine
            save_env_value(env_path, env, key)
            os.environ[env] = key  # so the steps below and the summary see it
            have_keys.append(env)
            return


def _print_result(console: Console, c: Check) -> None:
    icon = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]"}
    console.print(f"    {icon.get(c.status, '·')} {escape(c.detail)}")
    if c.fix and c.status == "fail":
        console.print(f"      [cyan]→ {escape(c.fix)}[/cyan]")


def _setup_local(console: Console, ask: Prompts, local, data: dict, lister) -> None:
    import openai

    if not shutil.which("ollama"):
        console.print("  [yellow]![/yellow] Strict privacy needs a local AI. Install Ollama:")
        console.print(f"    {OLLAMA_INSTALL}")
        console.print(f"    ollama pull {local.model}")
        return
    try:
        pulled = lister(local, "")
    except openai.APIConnectionError:
        console.print(
            "  Ollama is installed but not running. Start it with [bold]ollama serve[/bold],"
        )
        console.print(f"  then [bold]ollama pull {local.model}[/bold] if you haven't.")
        return
    except Exception as e:
        console.print(f"  [yellow]![/yellow] could not ask Ollama for its models: {escape(str(e))}")
        return
    if local.model in pulled or f"{local.model}:latest" in pulled:
        console.print(f"  [green]✓[/green] Ollama has {local.model}")
        return
    options = [(m, m + _size_note(m)) for m in pulled]
    options.append(
        ("", f"Pull {local.model} later (recommended; ~9 GB): ollama pull {local.model}")
    )
    choice = ask.choice(
        f"  Ollama doesn't have {local.model}. Which model should the local fallback use?",
        options,
        default=next(
            (m for m in pulled if (model_size_b(m) or SMALL_MODEL_B) >= SMALL_MODEL_B), ""
        ),
    )
    if choice:
        _set(data, "providers.local.model", choice)
    else:
        console.print(f"  Run [bold]ollama pull {local.model}[/bold] when you can.")


def _size_note(model: str) -> str:
    size = model_size_b(model)
    if size is not None and size < SMALL_MODEL_B:
        return f"  [dim](only {size:g}B: often wrong; fine as a last resort)[/dim]"
    return ""


def mic_test(console: Console, data: dict) -> None:
    """Record 3 s; if it's silent, offer the other inputs."""
    from voxtex.config import load_config as _load
    from voxtex.doctor import print_check, record_clip

    config = _load(None)
    config.mic = str(data.get("listen", {}).get("mic", config.mic))
    check, _audio = record_clip(config, console, seconds=3)
    print_check(console, check)
    if check.status == "ok":
        return
    sources = _mic_names()
    if not sources:
        return
    console.print("  Inputs found:")
    for i, name in enumerate(sources, 1):
        console.print(f"    {i}. {escape(name)}")
    pick = Prompt.ask("  Use which one (Enter to keep the default)", default="", show_default=False)
    if pick.isdigit() and 1 <= int(pick) <= len(sources):
        _set(data, "listen.mic", sources[int(pick) - 1])
        console.print(f"  [green]✓[/green] microphone: {escape(sources[int(pick) - 1])}")


def _mic_names() -> list[str]:
    import subprocess

    from voxtex.audio.capture import list_inputs, prefer_parec

    if prefer_parec():
        out = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True)
        rows = [line.split("\t") for line in out.stdout.splitlines()]
        return [r[1] for r in rows if len(r) > 1 and not r[1].endswith(".monitor")]
    try:
        return [name for _, name in list_inputs()]
    except OSError:
        return []


def first_run_needed(global_dir: Path | None = None) -> bool:
    return not ((global_dir or GLOBAL_CONFIG) / CONFIG_FILE).exists()


def skip_first_run(global_dir: Path | None = None) -> None:
    """Remember that setup was declined, so it isn't offered again."""
    write_config((global_dir or GLOBAL_CONFIG) / CONFIG_FILE, {})
