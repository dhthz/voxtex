"""Engine with a mocked LLM: canned decisions, real prompts, real buffer updates."""

import json

import pytest
from typer.testing import CliRunner

from voxtex.config import ProviderConfig, load_config
from voxtex.engine.decide import Delete, Ignore, Insert, Replace, UncertainInsert, Wait
from voxtex.engine.engine import Engine
from voxtex.engine.prompts import estimate_tokens
from voxtex.io.marker import find_marker
from voxtex.project.index import load_index
from voxtex.providers import Chain, Reply

TABLE = r"""\begin{table}[t]
  \centering
  \caption{Macro-$F_1$ of \legalbert and \model{BERT-base} \parencite{chalkidis2020}.}
  \label{tab:legalbert-vs-bert}
  \begin{tabular}{lS[table-format=1.2]}
    \toprule
    Model & {Macro-$F_1$} \\
    \midrule
    \legalbert & 0.71 \\
    \model{BERT-base} & 0.63 \\
    \bottomrule
  \end{tabular}
\end{table}"""


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def __call__(self, p, key, messages, json_mode, timeout):
        self.seen.append(messages)
        return Reply(json.dumps(self.replies.pop(0)), input_tokens=900, output_tokens=120)


@pytest.fixture
def make_engine(sample_thesis, tmp_path):
    def _make(*replies):
        idx, _ = load_index(sample_thesis)
        llm = ScriptedLLM(*replies)
        local = ProviderConfig(name="local", base_url="http://localhost:11434/v1", model="m")
        cfg = load_config(None, global_dir=tmp_path / "noconfig")
        loc = find_marker(idx.root_path, idx.body_files)
        return Engine(idx, Chain([local], send=llm), cfg, loc), llm

    return _make


def test_table_then_correction(make_engine):
    """The M2 acceptance scenario, with the model's answers mocked."""
    engine, llm = make_engine(
        {"action": "insert", "latex": TABLE},
        {"action": "replace", "block_id": "b1", "latex": TABLE.replace("0.63", "0.61")},
    )
    s1 = engine.feed(
        "add a table comparing legal-bert 0.71 and bert-base 0.63 on macro F1, "
        "cite the chalkidis paper"
    )
    assert isinstance(s1.decision, Insert) and s1.block.id == "b1"
    system, user = llm.seen[0][0]["content"], llm.seen[0][1]["content"]
    assert "booktabs" in system and "\\parencite" in system and "tab:baselines" in system
    assert "chalkidis2020 | Chalkidis | 2020" in user and "chalkidis2019" in user
    assert "SPEECH:\nadd a table" in user
    assert engine.buffer.pending == []

    s2 = engine.feed("no wait, the second one was 0.61")
    assert isinstance(s2.decision, Replace) and s2.block.id == "b1"
    assert "0.61" in engine.buffer.get("b1").latex
    user2 = llm.seen[1][1]["content"]
    assert "RECENT BLOCKS" in user2 and "[b1] \\begin{table}" in user2
    assert "chalkidis2020" in user2  # retrieval also looks at the block being corrected
    assert llm.seen[0][0] == llm.seen[1][0]  # stable system prompt (prompt caching)


def test_wait_keeps_speech_until_forced(make_engine):
    engine, llm = make_engine(
        {"action": "wait"},
        {"action": "insert", "latex": "We use two encoders."},
    )
    assert isinstance(engine.feed("so we use").decision, Wait)
    assert engine.buffer.pending == ["so we use"]
    s = engine.feed("two encoders", force=True)
    assert isinstance(s.decision, Insert)
    user = llm.seen[1][1]["content"]
    assert "SPEECH:\nso we use two encoders" in user
    assert 'Do not answer "wait"' in user


def test_ignore_and_delete(make_engine):
    engine, _ = make_engine(
        {"action": "insert", "latex": "First."},
        {"action": "ignore"},
        {"action": "delete", "block_id": "b1"},
    )
    engine.feed("first sentence")
    assert isinstance(engine.feed("hmm let me think").decision, Ignore)
    assert engine.buffer.pending == []
    s = engine.feed("scratch that")
    assert isinstance(s.decision, Delete) and engine.buffer.blocks == []


def test_unknown_block_ids_degrade_safely(make_engine):
    engine, _ = make_engine(
        {"action": "replace", "block_id": "b9", "latex": "Text."},
        {"action": "delete", "block_id": "b7"},
    )
    s = engine.feed("rewrite that")
    assert isinstance(s.decision, UncertainInsert) and "b9" in s.block.note
    s = engine.feed("delete it")
    assert isinstance(s.decision, Ignore) and s.notices == ["no block b7 to delete"]


def test_uncertain_insert_keeps_note(make_engine):
    engine, _ = make_engine(
        {
            "action": "uncertain_insert",
            "latex": "\\parencite{chalkidis2019}",
            "note": "2019 or 2020?",
        }
    )
    s = engine.feed("cite chalkidis")
    assert s.block.note == "2019 or 2020?"


def test_flush_and_empty_input(make_engine):
    engine, llm = make_engine({"action": "insert", "latex": "X."})
    assert engine.feed("   ") is None
    assert engine.flush() is None
    engine.buffer.add_utterance("pending text", 0)
    assert isinstance(engine.flush().decision, Insert)


def test_prompt_budget(make_engine):
    """Addendum §5: ≤ 2,000 input tokens for a typical call."""
    engine, llm = make_engine({"action": "insert", "latex": "X."})
    engine.feed("add a table comparing legal-bert and bert-base, cite the chalkidis paper")
    assert estimate_tokens(llm.seen[0]) < 2000


def test_cli_text_mode(sample_thesis, monkeypatch, tmp_path):
    from voxtex import config
    from voxtex.cli import app

    monkeypatch.setattr(config, "GLOBAL_CONFIG", tmp_path / "noconfig")
    llm = ScriptedLLM(
        {"action": "insert", "latex": "Results are shown in \\cref{tab:baselines}."},
        {"action": "ignore"},
    )
    monkeypatch.setattr("voxtex.providers.send_openai", llm)
    r = CliRunner().invoke(
        app,
        ["text", str(sample_thesis), "--provider", "local"],
        input="mention the baseline table\nhmm\n",
    )
    assert r.exit_code == 0, r.output
    assert "chapters/methodology.tex at %> HERE" in r.output
    assert "+ inserted b1" in r.output
    assert "ignored" in r.output
    assert "Session: 2 calls" in r.output
    log = sample_thesis / ".voxtex" / "timings.log"
    assert len(log.read_text().splitlines()) == 2


def test_cli_text_no_provider(sample_thesis, monkeypatch, tmp_path):
    from voxtex import config
    from voxtex.cli import app

    monkeypatch.setattr(config, "GLOBAL_CONFIG", tmp_path / "noconfig")
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (sample_thesis / ".voxtex.toml").write_text('[llm]\nchain = ["groq-gptoss"]\n')
    r = CliRunner().invoke(app, ["text", str(sample_thesis)], input="")
    assert r.exit_code == 1
    assert "no usable AI provider" in r.output
