from pathlib import Path

from voxtex.project.discover import SourceFile
from voxtex.project.labels import collect_labels, label_conventions

BODY = r"""
\chapter{Intro}\label{chap-intro}
\section{Background}
\label{sec-background}
Text \label{loose}
\begin{figure}
  \begin{subfigure}{0.5\textwidth}\label{fig-a}\end{subfigure}
  \caption{x}\label{fig-main}
\end{figure}
\begin{table}\caption{t}\label{tab-results}\end{table}
\begin{align}
  a &= b \label{eq-first}
\end{align}
"""


def labels():
    return collect_labels([SourceFile(Path("ch.tex"), BODY)])


def test_label_kinds_and_lines():
    got = {lab.name: (lab.kind, lab.line) for lab in labels()}
    assert got == {
        "chap-intro": ("chapter", 2),
        "sec-background": ("section", 4),
        "loose": ("section", 5),  # just after a heading, still counts as that section
        "fig-a": ("figure", 7),
        "fig-main": ("figure", 8),
        "tab-results": ("table", 10),
        "eq-first": ("equation", 12),
    }


def test_conventions_detect_dash_separator():
    c = label_conventions(labels())
    assert c.separator == "-"
    assert c.prefix_counts == {"fig": 2, "chap": 1, "sec": 1, "tab": 1, "eq": 1}
    assert c.prefix_by_kind == {
        "chapter": "chap",
        "section": "sec",
        "figure": "fig",
        "table": "tab",
        "equation": "eq",
    }
    assert c.summary.startswith("fig-")


def test_conventions_fixture(sample_thesis):
    from voxtex.project.discover import discover

    c = label_conventions(collect_labels(discover(sample_thesis).body))
    assert c.separator == ":"
    assert c.prefix_by_kind["figure"] == "fig"
    assert c.prefix_by_kind["table"] == "tab"
    assert c.prefix_by_kind["equation"] == "eq"
    assert c.prefix_by_kind["chapter"] == "sec"


def test_no_labels():
    c = label_conventions([])
    assert c.separator == ":" and c.prefix_counts == {}


def test_prefix_without_separator():
    from voxtex.project.labels import Label, split_prefix

    assert split_prefix("figure4.1") == ("figure", "")
    assert split_prefix("tableAppA.01") == ("table", "")
    assert split_prefix("fig:pipeline") == ("fig", ":")
    assert split_prefix("intro") is None
    assert split_prefix("fig:") is None
    c = label_conventions(
        [
            Label(name="figure4.1", kind="figure"),
            Label(name="figureB.1", kind="figure"),
            Label(name="table07.01", kind="table"),
        ]
    )
    assert c.separator == ""
    assert c.prefix_by_kind == {"figure": "figure", "table": "table"}
    assert c.example_by_kind == {"figure": "figure4.1", "table": "table07.01"}
    assert c.summary == "figure…, table…"
