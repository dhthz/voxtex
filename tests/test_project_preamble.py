from pathlib import Path

import pytest

from voxtex.project.discover import SourceFile
from voxtex.project.preamble import parse_preamble, xparse_arity


def pre(text: str):
    return parse_preamble([SourceFile(Path("main.tex"), text)])


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("m", (1, 0)),
        ("m m", (2, 0)),
        ("s m O{default}", (3, 2)),
        ("o m", (2, 1)),
        ("D<>{x} m", (2, 1)),
        ("r() m", (2, 0)),
        ("t+ m", (2, 1)),
        ("+m >{\\SplitList{,}}m", (2, 0)),
        ("", (0, 0)),
    ],
)
def test_xparse_arity(spec, expected):
    assert xparse_arity(spec) == expected


def test_packages_with_options_merged():
    p = pre(
        "\\documentclass[11pt,a4paper]{report}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\usepackage[style=authoryear, backend=biber]{biblatex}\n"
        "\\PassOptionsToPackage{x}{y}\\usepackage[final]{fontenc}\n"
    )
    assert p.documentclass == "report"
    assert p.class_options == ["11pt", "a4paper"]
    assert [x.name for x in p.packages] == ["fontenc", "amsmath", "amssymb", "biblatex"]
    assert p.package("fontenc").options == ["T1", "final"]
    assert p.package("biblatex").option("style") == "authoryear"
    assert p.package("biblatex").option("missing") is None


def test_macro_forms():
    p = pre(
        "\\newcommand{\\a}{x}\n"
        "\\newcommand\\b[2]{#1#2}\n"
        "\\renewcommand*{\\c}[2][opt]{#1}\n"
        "\\providecommand{\\d}{}\n"
        "\\DeclareMathOperator{\\e}{e}\n"
        "\\DeclareMathOperator*{\\f}{f}\n"
        "\\NewDocumentCommand{\\g}{s o m}{}\n"
        "\\NewDocumentCommand\\h{m}{}\n"
        "\\def\\i#1#2{}\n"
        "\\let\\j\\relax\n"
        "\\newcommand{\\vect}[1]{\\mathbf{#1}}\n"
        "\\newcommand{\\score}[1]{$F_1$ #1}\n"
        "\\def\\my@internal{}\n"
    )
    got = {m.name: (m.arity, m.optional, m.kind) for m in p.macros}
    assert got == {
        "\\a": (0, 0, "newcommand"),
        "\\b": (2, 0, "newcommand"),
        "\\c": (2, 1, "renewcommand"),
        "\\d": (0, 0, "providecommand"),
        "\\e": (0, 0, "DeclareMathOperator"),
        "\\f": (0, 0, "DeclareMathOperator"),
        "\\g": (3, 2, "NewDocumentCommand"),
        "\\h": (1, 0, "NewDocumentCommand"),
        "\\i": (2, 0, "def"),
        "\\j": (0, 0, "let"),
        "\\vect": (1, 0, "newcommand"),
        "\\score": (1, 0, "newcommand"),
    }
    math = {m.name for m in p.macros if m.math}
    assert math == {"\\e", "\\f", "\\vect"}


def test_environments():
    p = pre(
        "\\newenvironment{note}[1]{}{}\n"
        "\\NewDocumentEnvironment{boxed}{o m}{}{}\n"
        "\\newtheorem{theorem}{Theorem}[chapter]\n"
    )
    assert {e.name: e.arity for e in p.environments} == {"note": 1, "boxed": 2, "theorem": 0}


def test_macro_body_and_uses():
    from voxtex.project.preamble import count_macro_uses

    p = pre("\\newcommand{\\en}[1]{\\foreignlanguage{english}{#1}}\\def\\x{y}\\newcommand{\\enx}{}")
    bodies = {m.name: m.body for m in p.macros}
    assert bodies == {"\\en": "\\foreignlanguage{english}{#1}", "\\x": "y", "\\enx": ""}
    count_macro_uses(p.macros, [SourceFile(Path("c.tex"), "\\en{a} \\en{b} \\enx \\english")])
    assert {m.name: m.uses for m in p.macros} == {"\\en": 2, "\\x": 0, "\\enx": 1}
