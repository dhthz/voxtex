from voxtex.config import load_config


def test_defaults_without_files(tmp_path):
    c = load_config(None, global_dir=tmp_path / "none")
    assert c.chain == ["groq-gptoss", "groq-llama", "gemini", "openrouter", "local"]
    assert c.providers["local"].private
    assert not c.providers["groq-gptoss"].private
    assert c.style == "academic" and c.sources == []


def test_global_then_project_merge(tmp_path):
    g = tmp_path / "global"
    g.mkdir()
    (g / "config.toml").write_text(
        'privacy = "normal"\n'
        '[llm]\nchain = ["groq-gptoss", "local"]\n'
        '[providers.local]\nmodel = "llama3.2:3b"\n'
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".voxtex.toml").write_text('[writing]\nstyle = "clean"\n')
    c = load_config(proj, global_dir=g)
    assert c.chain == ["groq-gptoss", "local"]
    assert c.providers["local"].model == "llama3.2:3b"
    assert c.providers["local"].base_url == "http://localhost:11434/v1"  # default kept
    assert c.style == "clean"
    assert len(c.sources) == 2


def test_chain_privacy(tmp_path):
    c = load_config(None, global_dir=tmp_path)
    assert [p.name for p in c.chain_providers(private_only=True)] == ["local"]
    c.privacy = "strict"
    assert [p.name for p in c.chain_providers()] == ["local"]


def test_resolve_key(tmp_path, monkeypatch):
    c = load_config(None, global_dir=tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert c.providers["groq-gptoss"].resolve_key() is None
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert c.providers["groq-gptoss"].resolve_key() == "gsk_x"
    assert c.providers["local"].resolve_key() == ""  # no key needed
