"""Configuration: built-in defaults ← ~/.config/voxtex/config.toml ← <project>/.voxtex.toml.

API keys come from environment variables, or from `.env` files (KEY=value lines) in the
project folder, the current directory, or ~/.config/voxtex/. Real environment variables win,
then the first file that defines the key.
"""

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GLOBAL_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voxtex"
PROJECT_CONFIG = ".voxtex.toml"
ENV_FILE = ".env"

DEFAULTS: dict[str, Any] = {
    "privacy": "normal",  # normal | strict (only private providers)
    "llm": {
        "chain": ["groq-gptoss", "groq-qwen", "gemini", "openrouter", "local"],
        "timeout": 8.0,
    },
    "providers": {
        "groq-gptoss": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "model": "openai/gpt-oss-120b",
        },
        "groq-qwen": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "model": "qwen/qwen3.8-27b",
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemini-2.5-flash",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
        },
        "local": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5-coder:14b",
            "private": True,
            "timeout": 60.0,  # local models are slower, especially on first load
        },
    },
    "writing": {
        "style": "academic",  # clean | academic
        "language": "auto",  # auto | en | el
    },
    "validate": {
        "compile": "auto",  # auto (only for unfamiliar commands) | always | off
    },
}


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    model: str
    api_key_env: str = ""
    api_key: str = ""
    private: bool = False
    json_mode: bool = True  # send response_format=json_object; auto-disabled if rejected
    timeout: float | None = None  # overrides llm.timeout

    def resolve_key(self) -> str | None:
        """The API key, "" if none is needed, or None if one is needed but missing."""
        if self.api_key:
            return self.api_key
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env) or None


class Config(BaseModel):
    privacy: str = "normal"
    chain: list[str]
    timeout: float = 8.0
    providers: dict[str, ProviderConfig]
    style: str = "academic"
    language: str = "auto"
    compile_check: str = "auto"
    sources: list[str] = []  # config files that were merged in, for `doctor`

    def chain_providers(self, private_only: bool = False) -> list[ProviderConfig]:
        """Providers in chain order; unknown names are skipped."""
        strict = private_only or self.privacy == "strict"
        out = []
        for name in self.chain:
            p = self.providers.get(name)
            if p and (p.private or not strict):
                out.append(p)
        return out


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader: `KEY=value`, optional `export`, quotes, `#` comments."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        out[key.strip()] = value
    return out


def load_env_files(paths: list[Path]) -> tuple[dict[str, str], list[str]]:
    """Merge .env files; earlier paths win. Returns (values, files that existed)."""
    env: dict[str, str] = {}
    found = []
    for p in paths:
        if p.is_file():
            found.append(str(p))
            for k, v in parse_env_file(p).items():
                env.setdefault(k, v)
    return env, found


def load_config(project_root: Path | None = None, global_dir: Path | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    sources = []
    gdir = global_dir or GLOBAL_CONFIG
    env_paths = [Path.cwd() / ENV_FILE, gdir / ENV_FILE]
    if project_root is not None:
        env_paths.insert(0, project_root / ENV_FILE)
    dotenv, env_sources = load_env_files(list(dict.fromkeys(env_paths)))
    paths = [gdir / "config.toml"]
    if project_root is not None:
        paths.append(project_root / PROJECT_CONFIG)
    for p in paths:
        if p.is_file():
            with p.open("rb") as fh:
                data = _merge(data, tomllib.load(fh))
            sources.append(str(p))
    providers = {}
    for name, cfg in data["providers"].items():
        p = ProviderConfig(name=name, **cfg)
        if not p.api_key and p.api_key_env and not os.environ.get(p.api_key_env):
            p.api_key = dotenv.get(p.api_key_env, "")
        providers[name] = p
    return Config(
        privacy=data.get("privacy", "normal"),
        chain=data["llm"]["chain"],
        timeout=data["llm"].get("timeout", 8.0),
        providers=providers,
        style=data["writing"]["style"],
        language=data["writing"]["language"],
        compile_check=data["validate"]["compile"],
        sources=sources + env_sources,
    )
