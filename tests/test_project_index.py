import os

from typer.testing import CliRunner

from voxtex.cli import app
from voxtex.project.index import load_index


def test_build_and_cache(sample_thesis):
    idx, cached = load_index(sample_thesis)
    assert not cached
    assert (sample_thesis / ".voxtex" / "index.json").is_file()
    assert idx.main == "main.tex"
    assert len(idx.bib) == 5
    assert {lab.name for lab in idx.labels} >= {"fig:pipeline", "tab:baselines", "eq:prediction"}
    assert {m.name for m in idx.preamble.macros} >= {"\\legalbert", "\\softmax", "\\metric"}

    again, cached = load_index(sample_thesis)
    assert cached
    assert again == idx


def test_cache_invalidated_on_change(sample_thesis):
    load_index(sample_thesis)
    ch = sample_thesis / "chapters" / "methodology.tex"
    ch.write_text(ch.read_text() + "\n\\section{New}\\label{sec:new}\n")
    st = ch.stat()
    os.utime(ch, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    idx, cached = load_index(sample_thesis)
    assert not cached
    assert "sec:new" in {lab.name for lab in idx.labels}


def test_rebuild_flag(sample_thesis):
    load_index(sample_thesis)
    assert not load_index(sample_thesis, rebuild=True)[1]


def test_corrupt_cache_is_rebuilt(sample_thesis):
    load_index(sample_thesis)
    (sample_thesis / ".voxtex" / "index.json").write_text("{not json")
    idx, cached = load_index(sample_thesis)
    assert not cached and idx.main == "main.tex"


def test_cli_index_summary(sample_thesis):
    result = CliRunner().invoke(app, ["index", str(sample_thesis)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "biblatex (authoryear)" in out
    assert "5 bib entries" in out
    assert "used in chapters" in out
    assert "\\parencite" in out and "\\cref" in out


def test_cli_index_json_and_verbose(sample_thesis):
    r = CliRunner().invoke(app, ["index", str(sample_thesis), "--json"])
    assert r.exit_code == 0 and '"main": "main.tex"' in r.output
    r = CliRunner().invoke(app, ["index", str(sample_thesis), "-v"])
    assert r.exit_code == 0 and "Sample paragraph" in r.output


def test_cli_index_no_project(tmp_path):
    r = CliRunner().invoke(app, ["index", str(tmp_path)])
    assert r.exit_code == 1
