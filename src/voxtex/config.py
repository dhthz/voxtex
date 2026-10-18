"""Configuration: built-in defaults ← ~/.config/voxtex/config.toml ← <project>/.voxtex.toml."""

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GLOBAL_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voxtex"
PROJECT_CONFIG = ".voxtex.toml"

DEFAULTS: dict[str, Any] = {
    "privacy": "normal",  # normal | strict (only private providers)
    "llm": {
        "chain": ["groq-gptoss", "groq-llama", "gemini", "openrouter", "local"],
        "timeout": 8.0,
    },
    "providers": {
        "groq-gptoss": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "model": "openai/gpt-oss-120b",
        },
        "groq-llama": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "model": "llama-3.3-70b-versatile",
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


def load_config(project_root: Path | None = None, global_dir: Path | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    sources = []
    paths = [(global_dir or GLOBAL_CONFIG) / "config.toml"]
    if project_root is not None:
        paths.append(project_root / PROJECT_CONFIG)
    for p in paths:
        if p.is_file():
            with p.open("rb") as fh:
                data = _merge(data, tomllib.load(fh))
            sources.append(str(p))
    return Config(
        privacy=data.get("privacy", "normal"),
        chain=data["llm"]["chain"],
        timeout=data["llm"].get("timeout", 8.0),
        providers={
            name: ProviderConfig(name=name, **cfg) for name, cfg in data["providers"].items()
        },
        style=data["writing"]["style"],
        language=data["writing"]["language"],
        sources=sources,
    )
