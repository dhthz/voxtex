import stat
import tomllib

import pytest
from rich.console import Console

from voxtex import initwizard
from voxtex.config import load_config, parse_env_file
from voxtex.initwizard import first_run_needed, model_size_b, run_init, save_env_value


class Scripted:
    """Answers questions from a list, in order; records what was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def _next(self, question):
        self.asked.append(question)
        return self.answers.pop(0)

    def choice(self, question, options, default):
        a = self._next(question)
        return default if a is None else a

    def secret(self, question):
        return self._next(question)

    def confirm(self, question, default=True):
        a = self._next(question)
        return default if a is None else a


def lister(models_by_url):
    def _list(p, key):
        if key == "bad":
            import httpx2 as httpx
            import openai

            req = httpx.Request("GET", "http://x")
            raise openai.AuthenticationError(
                "bad", response=httpx.Response(401, request=req), body=None
            )
        return models_by_url.get(p.base_url, [])

    return _list


GROQ = "https://api.groq.com/openai/v1"
OLLAMA = "http://localhost:11434/v1"


@pytest.fixture
def gdir(tmp_path, monkeypatch):
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr("voxtex.audio.stt.cuda_available", lambda: True)
    monkeypatch.setattr(
        initwizard.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "ollama" else None
    )
    return tmp_path / "global"


def test_model_size():
    assert model_size_b("qwen2.5-coder:14b") == 14
    assert model_size_b("llama3.2:3b") == 3
    assert model_size_b("mistral:latest") is None


def test_save_env_value_replaces_and_is_private(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keys\nexport GROQ_API_KEY=old\nOTHER=1\n", encoding="utf-8")
    save_env_value(env, "GROQ_API_KEY", "new")
    save_env_value(env, "GEMINI_API_KEY", "g")
    assert parse_env_file(env) == {"GROQ_API_KEY": "new", "OTHER": "1", "GEMINI_API_KEY": "g"}
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_wizard_tests_and_saves_keys(gdir):
    ask = Scripted([
        None,  # speech: default (local, GPU)
        "bad", "gsk_good",  # Groq: rejected, then accepted
        "",  # Gemini: skip
        "",  # OpenRouter: skip
        "mistral:latest",  # local model: one that's pulled
        None,  # privacy: normal
    ])  # fmt: skip
    models = {
        GROQ: ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        OLLAMA: ["llama3.2:3b", "mistral:latest"],
    }
    out = Console(width=200, record=True)
    assert run_init(out, ask, lister(models), gdir) == 0
    assert parse_env_file(gdir / ".env") == {"GROQ_API_KEY": "gsk_good"}
    data = tomllib.loads((gdir / "config.toml").read_text(encoding="utf-8"))
    assert data["stt"]["mode"] == "local"
    assert data["privacy"] == "normal"
    assert data["providers"]["local"]["model"] == "mistral:latest"
    text = out.export_text()
    assert "AI chain: groq-gptoss → groq-qwen → local" in text
    assert "rejected" in text


def test_wizard_keeps_an_existing_key(gdir, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "already")
    ask = Scripted([None, True, "", "", "", None])
    models = {OLLAMA: ["qwen2.5-coder:14b"]}
    assert run_init(Console(width=200), ask, lister(models), gdir) == 0
    assert not (gdir / ".env").exists()
    assert any("already set" in q for q in ask.asked)


def test_strict_privacy_with_cloud_speech_switches_to_local(gdir):
    ask = Scripted(["groq", "", "", "", "", "strict"])
    run_init(Console(width=200), ask, lister({OLLAMA: []}), gdir)
    data = tomllib.loads((gdir / "config.toml").read_text(encoding="utf-8"))
    assert data["stt"]["mode"] == "local"
    assert load_config(None, global_dir=gdir).privacy == "strict"


def test_rerun_keeps_other_settings(gdir):
    gdir.mkdir(parents=True)
    (gdir / "config.toml").write_text("[listen]\ndecision_pause = 3.0\n", encoding="utf-8")
    run_init(Console(width=200), Scripted(["external", "", "", "", "", None]), lister({}), gdir)
    data = tomllib.loads((gdir / "config.toml").read_text(encoding="utf-8"))
    assert data["listen"]["decision_pause"] == 3.0
    assert data["stt"]["mode"] == "external"


def test_first_run_needed(gdir):
    assert first_run_needed(gdir)
    initwizard.skip_first_run(gdir)
    assert not first_run_needed(gdir)
    load_config(None, global_dir=gdir)  # an empty config file is valid
