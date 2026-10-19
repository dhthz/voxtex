import httpx2 as httpx
import openai
import pytest

from voxtex import doctor
from voxtex.config import load_config
from voxtex.doctor import _same_model, check_ai, check_provider, run_doctor


def _req():
    return httpx.Request("GET", "http://x/models")


def lister_for(models=(), error=None):
    def lister(p, key):
        if error is not None:
            raise error
        return list(models)

    return lister


@pytest.fixture
def config(tmp_path, monkeypatch):
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    return load_config(None, global_dir=tmp_path / "global")


def test_same_model_handles_gemini_and_ollama_names():
    assert _same_model("gemini-2.5-flash", "models/gemini-2.5-flash")
    assert _same_model("mistral", "mistral:latest")
    assert not _same_model("qwen2.5-coder:14b", "qwen2.5-coder:7b")


def test_provider_without_key_is_skipped_with_a_link(config):
    c = check_provider(config.providers["gemini"], lister_for())
    assert c.status == "info"
    assert "aistudio.google.com" in c.fix


def test_provider_with_working_key_and_model(config):
    p = config.providers["groq-gptoss"].model_copy(update={"api_key": "k"})
    c = check_provider(p, lister_for(["openai/gpt-oss-120b", "whisper-large-v3-turbo"]))
    assert c.status == "ok"


def test_retired_model_is_a_failure_with_suggestions(config):
    p = config.providers["groq-qwen"].model_copy(update={"api_key": "k"})
    c = check_provider(p, lister_for(["qwen/qwen3.9-27b", "openai/gpt-oss-120b"]))
    assert c.status == "fail"
    assert "qwen/qwen3.9-27b" in c.detail
    assert "[providers.groq-qwen]" in c.fix


def test_rejected_key(config):
    p = config.providers["groq-gptoss"].model_copy(update={"api_key": "bad"})
    err = openai.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=_req()), body=None
    )
    c = check_provider(p, lister_for(error=err))
    assert c.status == "fail"
    assert "GROQ_API_KEY" in c.detail


def test_ollama_not_running_or_missing(config, monkeypatch):
    err = openai.APIConnectionError(request=_req())
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/ollama")
    c = check_provider(config.providers["local"], lister_for(error=err))
    assert (c.status, c.fix) == ("warn", "ollama serve")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    c = check_provider(config.providers["local"], lister_for(error=err))
    assert "install.sh" in c.fix


def test_ollama_model_not_pulled_lists_what_is(config):
    c = check_provider(config.providers["local"], lister_for(["mistral:latest"]))
    assert c.status == "warn"
    assert "mistral:latest" in c.detail
    assert "ollama pull qwen2.5-coder:14b" in c.fix


def test_no_working_provider_fails(config):
    checks = check_ai(config, lister_for(error=openai.APIConnectionError(request=_req())))
    assert checks[-1].status == "fail"


def test_chain_summary_lists_working_providers(config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    checks = check_ai(config, lister_for(["openai/gpt-oss-120b", "qwen2.5-coder:14b"]))
    assert checks[-1].label == "chain: groq-gptoss → local"


def test_run_doctor_quick(sample_thesis, monkeypatch, capsys):
    from rich.console import Console

    monkeypatch.setattr("voxtex.audio.stt.cuda_available", lambda: False)
    monkeypatch.setattr(doctor, "whisper_cached", lambda model: False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    console = Console(width=200)
    models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    code = run_doctor(console, sample_thesis, quick=True, lister=lister_for(models))
    out = capsys.readouterr().out
    assert code == 0
    assert "main.tex" in out
    assert "groq-gptoss" in out
    assert "groq whisper-large-v3-turbo (cloud)" in out, out  # no GPU, but a Groq key
