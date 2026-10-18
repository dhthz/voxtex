"""Live smoke test against Groq. Skipped unless GROQ_API_KEY is set (env or repo .env)."""

import os
from pathlib import Path

import pytest

from voxtex.config import load_config, load_env_files
from voxtex.engine.decide import Insert, Replace, UncertainInsert
from voxtex.engine.engine import Engine
from voxtex.io.marker import find_marker
from voxtex.project.index import load_index
from voxtex.providers import Chain

REPO_ENV = Path(__file__).resolve().parents[1] / ".env"
GROQ_KEY = os.environ.get("GROQ_API_KEY") or load_env_files([REPO_ENV])[0].get("GROQ_API_KEY")

pytestmark = pytest.mark.skipif(not GROQ_KEY, reason="needs GROQ_API_KEY")


def test_table_and_correction_live(sample_thesis, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    idx, _ = load_index(sample_thesis)
    cfg = load_config(None, global_dir=tmp_path)
    groq = [p for p in cfg.chain_providers() if p.name.startswith("groq")]
    engine = Engine(idx, Chain(groq, cfg.timeout), cfg, find_marker(idx.root_path, idx.body_files))

    s = engine.feed(
        "add a table comparing legal-bert 0.71 and bert-base 0.63 on macro F1, "
        "cite the chalkidis paper",
        force=True,
    )
    assert isinstance(s.decision, Insert | UncertainInsert)
    latex = s.block.latex
    assert "\\toprule" in latex and "\\hline" not in latex
    assert "\\parencite{chalkidis20" in latex or "\\textcite{chalkidis20" in latex
    assert "\\label{tab:" in latex
    assert "0.71" in latex and "0.63" in latex

    s = engine.feed("no wait, the second one was 0.61", force=True)
    assert isinstance(s.decision, Replace) and s.block.id == "b1"
    # "The second one" may mean the second model spoken or the second table row; either
    # way exactly one value becomes 0.61 and the other is kept.
    new = s.block.latex
    assert "0.61" in new and (("0.71" in new) != ("0.63" in new))
