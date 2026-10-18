import pytest

from voxtex.project.discover import DiscoveryError, discover, find_main


def doc(pre: str = "", body: str = "", cls: str = "report") -> str:
    return f"\\documentclass{{{cls}}}\n{pre}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"


def test_fixture(sample_thesis):
    pf = discover(sample_thesis)
    assert pf.main.name == "main.tex"
    assert pf.class_name == "report"
    assert [f.rel for f in pf.body] == [
        "main.tex",
        "chapters/introduction.tex",
        "chapters/methodology.tex",
    ]
    assert [b.name for b in pf.bib_files] == ["references.bib"]
    assert pf.missing == []


def test_main_preferred_over_standalone_and_subfiles(make_project):
    root = make_project(
        {
            "fig/plot.tex": doc(cls="standalone"),
            "chapters/ch1.tex": "\\documentclass[../thesis.tex]{subfiles}\n"
            "\\begin{document}\nHello\n\\end{document}\n",
            "thesis.tex": doc(body="\\subfile{chapters/ch1}"),
            "notes/scratch.tex": doc(),
        }
    )
    assert find_main(root).name == "thesis.tex"


def test_no_main_raises(make_project):
    root = make_project({"a.tex": "just text"})
    with pytest.raises(DiscoveryError):
        find_main(root)


def test_preamble_inputs_local_sty_and_cls(make_project):
    root = make_project(
        {
            "main.tex": doc(pre="\\input{setup/preamble}", cls="uniThesis"),
            "uniThesis.cls": "\\LoadClass{report}\\RequirePackage{mymacros}",
            "mymacros.sty": "\\newcommand{\\foo}{bar}",
            "setup/preamble.tex": "\\usepackage{booktabs}",
        }
    )
    pf = discover(root)
    assert [f.rel for f in pf.preamble] == [
        "main.tex",
        "setup/preamble.tex",
        "uniThesis.cls",
        "mymacros.sty",
    ]


def test_include_forms_and_missing(make_project):
    root = make_project(
        {
            "main.tex": doc(
                pre="\\newcommand{\\chap}[1]{\\input{#1}}\n\\bibliography{refs,other}",
                body="\\include{ch/a}\n\\input ch/b\n\\import{parts/}{c}\n\\input{nope}",
            ),
            "ch/a.tex": "\\input{ch/nested}",
            "ch/nested.tex": "nested",
            "ch/b.tex": "b",
            "parts/c.tex": "\\subimport{sub/}{d}",
            "parts/sub/d.tex": "d",
            "refs.bib": "",
        }
    )
    pf = discover(root)
    assert [f.rel for f in pf.body] == [
        "main.tex",
        "ch/a.tex",
        "ch/nested.tex",
        "ch/b.tex",
        "parts/c.tex",
        "parts/sub/d.tex",
    ]
    assert [b.name for b in pf.bib_files] == ["refs.bib"]
    assert sorted(pf.missing) == ["nope", "other.bib"]


def test_commented_includes_ignored(make_project):
    root = make_project({"main.tex": doc(body="% \\input{gone}\n\\input{here}"), "here.tex": "x"})
    pf = discover(root)
    assert [f.rel for f in pf.body] == ["main.tex", "here.tex"]
    assert pf.missing == []


def test_include_cycle_does_not_loop(make_project):
    root = make_project(
        {"main.tex": doc(body="\\input{a}"), "a.tex": "\\input{b}", "b.tex": "\\input{a}"}
    )
    assert [f.rel for f in discover(root).body] == ["main.tex", "a.tex", "b.tex"]
