import json

from voxtex.config import ProviderConfig, load_config
from voxtex.engine.engine import Engine
from voxtex.io.dictation import find_dictations
from voxtex.io.writer import Writer, read_file
from voxtex.project.index import load_index
from voxtex.providers import Chain, Reply
from voxtex.validate import Validator
from voxtex.watchmode import DictationProcessor, is_wsl_windows_drive

TEXT = """\\chapter{Results}

Some text.
%> legal bert gets 0.71
%>   and bert base 0.63

%> hmm let me think
%> HERE
Normal text.
%>
"""


def test_find_dictations():
    found = find_dictations(TEXT)
    assert [(d.line, d.transcript) for d in found] == [
        (4, "legal bert gets 0.71 and bert base 0.63"),
        (7, "hmm let me think"),
    ]
    assert found[0].raw(TEXT) == "%> legal bert gets 0.71\n%>   and bert base 0.63\n"


def test_find_dictations_crlf():
    found = find_dictations("a\r\n%> one\r\n%> two\r\nb\r\n")
    assert found[0].transcript == "one two" and found[0].raw(
        "a\r\n%> one\r\n%> two\r\nb\r\n"
    ).endswith("\r\n")


class LLM:
    def __init__(self, *replies):
        self.replies = list(replies)

    def __call__(self, p, key, messages, json_mode, timeout):
        return Reply(json.dumps(self.replies.pop(0)))


def processor(sample_thesis, tmp_path, *replies):
    idx, _ = load_index(sample_thesis)
    local = ProviderConfig(name="local", base_url="http://localhost:11434/v1", model="m")
    cfg = load_config(None, global_dir=tmp_path / "none")
    engine = Engine(idx, Chain([local], send=LLM(*replies)), cfg, None)
    engine.validator = Validator(idx, "off")
    return DictationProcessor(engine, Writer(idx.root_path))


def test_process_converts_corrects_ignores_and_undoes(sample_thesis, tmp_path):
    f = sample_thesis / "chapters" / "methodology.tex"
    original = read_file(f)
    f.write_text(original + "\n%> legal bert reaches 0.71\n\n%> um wait\n", encoding="utf-8")
    proc = processor(
        sample_thesis,
        tmp_path,
        {"action": "insert", "latex": "\\legalbert reaches 0.71."},
        {"action": "ignore"},
        {"action": "replace", "block_id": "b1", "latex": "\\legalbert reaches 0.73."},
    )
    assert proc.process(f) == 2
    assert read_file(f) == original + "\n\\legalbert reaches 0.71.\n\n"

    # A correction dictated later edits the earlier block and removes its own %> line.
    f.write_text(read_file(f) + "%> no, it was 0.73\n", encoding="utf-8")
    assert proc.process(f) == 1
    assert read_file(f) == original + "\n\\legalbert reaches 0.73.\n\n"

    # Undo brings everything back, %> lines included.
    w = Writer(sample_thesis)
    done, problem = w.undo(10)
    assert problem is None
    assert read_file(f) == (
        original + "\n%> legal bert reaches 0.71\n\n%> um wait\n%> no, it was 0.73\n"
    )


def test_failed_block_is_left_alone(sample_thesis, tmp_path):
    from voxtex.providers import Unavailable

    f = sample_thesis / "chapters" / "methodology.tex"
    f.write_text(read_file(f) + "%> something\n", encoding="utf-8")
    proc = processor(sample_thesis, tmp_path)

    def down(*a):
        raise Unavailable("down")

    proc.engine.chain.send = down
    before = read_file(f)
    assert proc.process(f) == 1
    assert read_file(f) == before
    assert proc.process(f) == 0  # not retried until the file changes


def test_is_wsl_windows_drive(tmp_path):
    from pathlib import Path

    assert is_wsl_windows_drive(Path("/mnt/e/thesis")) is True
    assert is_wsl_windows_drive(tmp_path) is False
