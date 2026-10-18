"""Unit tests for each validator, against the fixture project's index."""

import shutil

import pytest

from voxtex.project.index import build_index
from voxtex.validate import ValidationContext, Validator
from voxtex.validate.citations import check_citations
from voxtex.validate.compile import Compiler
from voxtex.validate.numbers import check_numbers, prose_numbers, spoken_numbers
from voxtex.validate.refs import check_labels, check_refs
from voxtex.validate.syntax import check_syntax

HAS_TEX = shutil.which("pdflatex") is not None


@pytest.fixture
def idx(sample_thesis):
    return build_index(sample_thesis)


def ctx(idx, transcript="", replaced="", other=()):
    return ValidationContext(idx, transcript, replaced, set(other))


def msgs(issues):
    return [i.message for i in issues]


# --- citations ----------------------------------------------------------------------------------


def test_citations(idx):
    ok = r"\parencite[see][p.~3]{chalkidis2019, chalkidis2020} and \textcite{devlin2019}"
    assert check_citations(ok, ctx(idx)) == []
    bad = check_citations(r"\cite{chalkidis2021,vaswani2017} \citep{madeup}", ctx(idx))
    assert msgs(bad) == [
        "unknown citation key 'chalkidis2021' (did you mean chalkidis2020 or chalkidis2019?)",
        "unknown citation key 'madeup'",
    ]


# --- refs and labels ----------------------------------------------------------------------------


def test_refs(idx):
    assert check_refs(r"\cref{tab:baselines,fig:pipeline} \eqref{eq:prediction}", ctx(idx)) == []
    # a label defined in this output, or in another block this session, is fine
    assert check_refs(r"\begin{table}\label{tab:new}\end{table} \cref{tab:new}", ctx(idx)) == []
    assert check_refs(r"\cref{tab:other}", ctx(idx, other={"tab:other"})) == []
    assert msgs(check_refs(r"see \Cref{tab:ghost}", ctx(idx))) == [
        "\\Cref{tab:ghost} points to no existing label"
    ]


def test_labels_pattern_and_uniqueness(idx):
    good = r"\begin{table}\caption{x}\label{tab:results}\end{table}"
    assert check_labels(good, ctx(idx)) == []
    wrong = r"\begin{figure}\caption{x}\label{figure-overview}\end{figure}"
    assert msgs(check_labels(wrong, ctx(idx))) == [
        "figure label 'figure-overview' doesn't follow the pattern, e.g. fig:pipeline"
    ]
    dup = r"\begin{table}\label{tab:baselines}\end{table}"
    assert msgs(check_labels(dup, ctx(idx))) == ["label 'tab:baselines' already exists"]
    # replacing the block that already defined it is not a duplicate
    assert (
        check_labels(
            r"\begin{table}\label{tab:x}\end{table}",
            ctx(idx, replaced=r"\label{tab:x}", other={"tab:x"}),
        )
        == []
    )
    assert msgs(check_labels(r"\label{sec:a}\label{sec:a}", ctx(idx))) == [
        "label 'sec:a' already exists"
    ]


# --- numbers ------------------------------------------------------------------------------------


def test_spoken_numbers():
    nums = spoken_numbers(
        "legal-bert 0.71 and bert-base zero point six three, three epochs, "
        "μηδέν κόμμα εξήντα ένα, twenty five; two hundred. 2e-5 and 0,5"
    )
    assert nums >= {0.71, 0.63, 3, 0.61, 25, 200, 2e-5, 0.5}
    assert 0.633 not in nums


def test_prose_numbers_ignore_structure():
    latex = (
        r"\begin{table}[t]\caption{Macro-$F_1$ \parencite{chalkidis2020}.}\label{tab:x2}"
        r"\begin{tabular}{lS[table-format=1.2]}\toprule \legalbert & 0.71 \\ GPT-4 F1 3D \\"
        r"\multicolumn{2}{c}{x} \cmidrule(lr){2-3} \end{tabular}\end{table} "
        r"\includegraphics[width=0.8\textwidth]{fig2} \hspace{2mm} \cref{fig:a1} "
        r"trained for \SI{3}{epochs} at \num{2e-5} on 12\% of $x^2 + 1$"
    )
    assert prose_numbers(latex) == ["0.71", "4", "3", "2e-5", "12", "1"]


def test_numbers(idx):
    latex = r"\legalbert scores 0.71 and BERT-base 0.61, over \SI{3}{epochs}."
    assert check_numbers(latex, ctx(idx, "legal bert 0.71, bert base 0.61, three epochs")) == []
    assert msgs(check_numbers(latex, ctx(idx, "legal bert 0.71 and bert base 0.61"))) == [
        "number 3 was not said by the speaker"
    ]
    # numbers already in the block being replaced are allowed
    assert check_numbers(latex, ctx(idx, "make it more formal", replaced=latex)) == []


# --- syntax -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latex, expected",
    [
        (r"\textbf{ok} \{literal\} $x$ \[ y \] 50\% \\ next", []),
        (r"\textbf{open", ["1 unclosed '{'"]),
        (r"close}", ["unmatched '}'"]),
        (
            r"\begin{table}\begin{tabular}{l}\end{table}",
            [
                "\\begin{tabular} closed by \\end{table}",
            ],
        ),
        (r"\begin{itemize}\item a", ["\\begin{itemize} is never closed"]),
        (r"\end{figure}", ["\\end{figure} without \\begin"]),
        (r"cost \$5 and $x", ["unbalanced $ (inline math not closed)"]),
        (
            r"\usepackage{x} \newcommand{\a}{b}",
            [
                "\\usepackage is not allowed in body text",
                "\\newcommand is not allowed in body text",
            ],
        ),
        (r"\verb|{| and % a { comment" + "\n" + r"\begin{verbatim}}\end{verbatim}", []),
    ],
)
def test_syntax(latex, expected):
    assert msgs(check_syntax(latex)) == expected


# --- compile ------------------------------------------------------------------------------------


def test_compile_log_parsing(idx):
    c = Compiler(idx, engine="pdflatex")
    n = c.preamble_lines
    log = (
        "! Undefined control sequence.\n"
        f"l.{n + 2} We use \\foo\n"
        "! LaTeX Error: File `plot.png' not found.\n"
        f"l.{n + 3} \\includegraphics{{plot.png}}\n"
        "! LaTeX Error: Environment tabularx undefined.\n"
        f"l.{n + 4} \\begin{{tabularx}}\n"
        "! Emergency stop.\n"
    )
    assert [(i.message, i.error) for i in c.parse_log(log)] == [
        ("LaTeX error: undefined command \\foo (line 2 of the output)", True),
        ("file 'plot.png' not found", False),
        ("LaTeX error: Environment tabularx undefined. (line 4 of the output)", True),
    ]


@pytest.mark.skipif(not HAS_TEX, reason="needs pdflatex")
def test_compile_real(idx):
    c = Compiler(idx)
    assert c.check(r"Text \cite{nokey} and \cref{tab:nolabel} and \legalbert.") == []
    assert msgs(c.check(r"Bad \undefinedcmd here.")) == [
        "LaTeX error: undefined command \\undefinedcmd (line 1 of the output)"
    ]


@pytest.mark.skipif(not HAS_TEX, reason="needs pdflatex")
def test_validator_compiles_only_unfamiliar_output(idx):
    v = Validator(idx, "auto")
    assert not v.needs_compile(r"\legalbert beats \model{BERT} \parencite{chalkidis2020}.")
    assert v.needs_compile(r"\begin{tabularx}{\linewidth}{X}\end{tabularx}")
    report = v.validate(r"We use \notdefined.", ctx(idx))
    assert report.compiled and [i.code for i in report.errors] == ["compile"]
    # structural errors skip the (slow, redundant) compile
    assert not v.validate(r"\notdefined{", ctx(idx)).compiled
    assert not Validator(idx, "off").needs_compile(r"\notdefined")


def test_numbers_already_in_the_document_are_allowed(idx):
    c = ValidationContext(
        idx, "sesame is faster", context=r"\en{Jena} & 120 \\ \en{Sesame} & 85 \\"
    )
    assert check_numbers("Sesame (85 ms) beats Jena (120 ms).", c) == []


@pytest.mark.parametrize(
    "latex, wrapped",
    [
        (
            r"Ο Broekstra και το \en{Sesame} για RDF δεδομένα \cite{Broekstra02}.",
            r"Ο \en{Broekstra} και το \en{Sesame} για \en{RDF} δεδομένα \cite{Broekstra02}.",
        ),
        (r"Το Jena Framework και το OWL-DL.", r"Το \en{Jena Framework} και το \en{OWL-DL}."),
        (
            r"\begin{table}[!tb]\caption{Για Jena}\begin{tabular}{| c | c |}\textbf{Sesame} & 85"
            r"\end{tabular}\label{tableAppA.02}\end{table}",
            r"\begin{table}[!tb]\caption{Για \en{Jena}}\begin{tabular}{| c | c |}"
            r"\textbf{\en{Sesame}} & 85\end{tabular}\label{tableAppA.02}\end{table}",
        ),
        (  # math, units, macro args, options, comments and refs are left alone
            r"σε $x_{max}$ \SI{3}{ms} \gloss{term}{ορ} \includegraphics[width=0.5\textwidth]{a.png}"
            r" \ref{figure4.1} % English comment",
            r"σε $x_{max}$ \SI{3}{ms} \gloss{term}{ορ} \includegraphics[width=0.5\textwidth]{a.png}"
            r" \ref{figure4.1} % English comment",
        ),
        (
            r"\begin{equation} f(x) = ax \end{equation}",
            r"\begin{equation} f(x) = ax \end{equation}",
        ),
    ],
)
def test_wrap_latin(latex, wrapped):
    from voxtex.validate.latin import wrap_latin

    assert wrap_latin(latex, "\\en") == wrapped
    assert wrap_latin(wrapped, "\\en") == wrapped  # idempotent
    assert wrap_latin(latex, "") == latex
