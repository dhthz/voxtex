"""Engine with a mocked LLM: canned decisions, real prompts, real buffer updates."""

import json
from pathlib import Path

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


def dictate(root, llm, utterances, monkeypatch, tmp_path, file=None, dry_run=False):
    """listen without the microphone: the same workspace, engine, session and display,
    fed transcripts directly. Returns (exit code, printed output)."""
    from rich.console import Console

    from voxtex import config
    from voxtex.app import open_workspace
    from voxtex.ui import UI

    monkeypatch.setattr(config, "GLOBAL_CONFIG", tmp_path / "noconfig")
    if llm is not None:
        monkeypatch.setattr("voxtex.providers.send_openai", llm)
    console = Console(record=True, width=120)
    ws = open_workspace(console, root, False, "local", file, dry_run, interactive=False)
    if ws is None:
        return 1, console.export_text()
    ui = UI(console)
    for u in utterances:
        ui.heard(u)
        step = ws.engine.feed(u, force=True)
        if step is not None:
            ui.step(step, ws.session, rel=ws.location.rel if ws.location else "")
    return 0, console.export_text()


def test_dictation_shows_writes(sample_thesis, monkeypatch, tmp_path):
    llm = ScriptedLLM(
        {"action": "insert", "latex": "Results are shown in \\cref{tab:baselines}."},
        {"action": "ignore"},
    )
    code, out = dictate(
        sample_thesis, llm, ["mention the baseline table", "hmm"], monkeypatch, tmp_path
    )
    assert code == 0, out
    assert "Write(chapters/methodology.tex)" in out and "(b1)" in out
    assert "Not content, nothing written" in out
    log = sample_thesis / ".voxtex" / "timings.log"
    assert len(log.read_text().splitlines()) == 2


def test_no_provider(sample_thesis, monkeypatch, tmp_path):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (sample_thesis / ".voxtex.toml").write_text('[llm]\nchain = ["groq-gptoss"]\n')
    from rich.console import Console

    from voxtex import config
    from voxtex.app import open_workspace

    monkeypatch.setattr(config, "GLOBAL_CONFIG", tmp_path / "noconfig")
    console = Console(record=True, width=120)
    assert open_workspace(console, sample_thesis, False, None, None, False, False) is None
    assert "no usable AI provider" in console.export_text()


def quiet(engine):
    """No compile check: keeps these tests fast and independent of a TeX install."""
    from voxtex.validate import Validator

    engine.validator = Validator(engine.idx, "off")
    return engine


def test_fake_cite_key_is_retried_and_fixed(make_engine):
    engine, llm = make_engine(
        {"action": "insert", "latex": "As shown \\parencite{chalkidis2021}."},
        {"action": "insert", "latex": "As shown \\parencite{chalkidis2020}."},
    )
    s = quiet(engine).feed("as chalkidis showed")
    assert isinstance(s.decision, Insert)
    assert s.block.latex.endswith("{chalkidis2020}.") and s.block.note == ""
    assert s.retried and "unknown citation key 'chalkidis2021'" in s.first_errors[0]
    retry = llm.seen[1]
    assert retry[:2] == llm.seen[0]  # same prompt, then the bad answer and the feedback
    assert retry[2]["role"] == "assistant" and "chalkidis2021" in retry[2]["content"]
    assert "unknown citation key 'chalkidis2021'" in retry[3]["content"]
    assert s.input_tokens == 1800  # both calls counted


def test_invented_number_twice_becomes_uncertain(make_engine):
    bad = {"action": "insert", "latex": "Accuracy improved to 0.93."}
    engine, llm = make_engine(bad, bad)
    s = quiet(engine).feed("accuracy improved a lot")
    assert isinstance(s.decision, UncertainInsert)
    assert s.block.note == "number 0.93 was not said by the speaker"
    assert s.report.errors and len(llm.seen) == 2


def test_retry_that_gives_up_keeps_first_answer_with_note(make_engine):
    engine, _ = make_engine(
        {"action": "insert", "latex": "See \\cref{fig:ghost}."},
        {"action": "ignore"},
    )
    s = quiet(engine).feed("see the ghost figure")
    assert isinstance(s.decision, UncertainInsert)
    assert "fig:ghost" in s.block.note


def test_replace_with_problems_keeps_block_and_notes_them(make_engine):
    engine, _ = make_engine(
        {"action": "insert", "latex": "Score 0.71."},
        {"action": "replace", "block_id": "b1", "latex": "Score 0.75."},
        {"action": "replace", "block_id": "b1", "latex": "Score 0.75."},
    )
    quiet(engine).feed("score 0.71")
    s = engine.feed("change it")
    assert isinstance(s.decision, Replace) and s.block.id == "b1"
    assert s.block.note == "number 0.75 was not said by the speaker"
    assert len(engine.buffer.blocks) == 1


def test_valid_output_is_not_retried(make_engine):
    engine, llm = make_engine({"action": "insert", "latex": "Legal-BERT reaches 0.71."})
    s = quiet(engine).feed("legal bert reaches 0.71")
    assert not s.retried and s.report.ok and len(llm.seen) == 1


def test_labels_from_this_session_are_offered(make_engine):
    engine, llm = make_engine(
        {"action": "insert", "latex": "\\begin{table}\\caption{x}\\label{tab:compare}\\end{table}"},
        {"action": "insert", "latex": "See \\cref{tab:compare}."},
    )
    quiet(engine).feed("add a table")
    s = engine.feed("as shown in the results table")
    assert "tab:compare (table)" in llm.seen[1][1]["content"]
    assert s.report.ok  # referencing a label from an earlier block validates


def test_dictation_writes_file_and_undo_restores(sample_thesis, monkeypatch, tmp_path):
    """M4 acceptance: dictation writes into the real file; undo restores the exact content."""
    from voxtex.cli import app

    (sample_thesis / ".voxtex.toml").write_text('[validate]\ncompile = "off"\n')
    f = sample_thesis / "chapters" / "methodology.tex"
    original = f.read_text()
    llm = ScriptedLLM(
        {"action": "insert", "latex": "\\legalbert reaches 0.71."},
        {"action": "replace", "block_id": "b1", "latex": "\\legalbert reaches 0.73."},
        {
            "action": "uncertain_insert",
            "latex": "See \\parencite{chalkidis2019}.",
            "note": "2019 or 2020?",
        },
    )
    code, out = dictate(
        sample_thesis, llm, ["legal bert reaches 0.71", "no, 0.73", "cite chalkidis"],
        monkeypatch, tmp_path,
    )  # fmt: skip
    assert code == 0, out
    assert "Update(chapters/methodology.tex)" in out
    text = f.read_text()
    assert (
        "\\legalbert reaches 0.73.\n\nSee \\parencite{chalkidis2019}.\n"
        "% voxtex-check: 2019 or 2020?\n\n%> HERE" in text
    )
    assert "0.71" not in text

    r = CliRunner().invoke(app, ["undo", "3", "--path", str(sample_thesis)])
    assert r.exit_code == 0, r.output
    assert "removed b2" in r.output and "restored b1" in r.output
    assert f.read_text() == original


def test_dry_run_changes_nothing(sample_thesis, monkeypatch, tmp_path):
    f = sample_thesis / "chapters" / "methodology.tex"
    original = f.read_text()
    llm = ScriptedLLM({"action": "insert", "latex": "X."})
    code, out = dictate(sample_thesis, llm, ["x"], monkeypatch, tmp_path, dry_run=True)
    assert code == 0 and "b1: would add 1 line" in out
    assert f.read_text() == original
    assert not (sample_thesis / ".voxtex" / "history.jsonl").exists()


def test_without_marker_needs_file(sample_thesis, monkeypatch, tmp_path):
    m = sample_thesis / "chapters" / "methodology.tex"
    m.write_text(m.read_text().replace("%> HERE\n", ""))
    code, out = dictate(sample_thesis, None, [], monkeypatch, tmp_path)
    assert code == 1 and "no %> HERE marker found" in out
    llm = ScriptedLLM({"action": "insert", "latex": "X."})
    intro = Path("chapters/introduction.tex")
    code, out = dictate(sample_thesis, llm, ["x"], monkeypatch, tmp_path, file=intro)
    assert code == 0, out
    assert (sample_thesis / "chapters" / "introduction.tex").read_text().endswith("X.\n\n%> HERE\n")


def test_retry_answering_replace_of_unknown_block_counts_as_insert(make_engine):
    engine, _ = make_engine(
        {"action": "insert", "latex": "Year 2002 \\parencite{chalkidis2020}."},
        {"action": "replace", "block_id": "b1", "latex": "As shown \\parencite{chalkidis2020}."},
    )
    s = quiet(engine).feed("as chalkidis showed")
    assert isinstance(s.decision, Insert) and s.block.note == "" and s.block.id == "b1"
