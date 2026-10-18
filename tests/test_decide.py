import pytest

from voxtex.engine.decide import Delete, Insert, Replace, UncertainInsert, Wait, parse_decision
from voxtex.providers import ParseError


def test_plain_json():
    assert parse_decision('{"action":"wait"}') == Wait(action="wait")


def test_fenced_and_surrounded_by_prose():
    d = parse_decision('Sure!\n```json\n{"action": "insert", "latex": "A {b} c"}\n```\nDone.')
    assert d == Insert(action="insert", latex="A {b} c")


def test_braces_inside_strings():
    d = parse_decision('{"action":"replace","block_id":"b2","latex":"\\\\textbf{x} }"}')
    assert isinstance(d, Replace) and d.latex == "\\textbf{x} }"


def test_action_normalised_and_optional_note():
    assert parse_decision('{"action":" Delete ","block_id":"b1"}') == Delete(
        action="delete", block_id="b1"
    )
    assert parse_decision('{"action":"uncertain_insert","latex":"x"}') == UncertainInsert(
        action="uncertain_insert", latex="x", note=""
    )


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"action":"insert"}',  # missing latex
        '{"action":"insert","latex":""}',
        '{"action":"dance"}',
        '{"action": "insert", "latex": "unterminated}',
        "[1, 2]",
    ],
)
def test_invalid(text):
    with pytest.raises(ParseError):
        parse_decision(text)


@pytest.mark.parametrize(
    "raw, latex",
    [
        # correctly escaped JSON is untouched
        (r'"\\toprule \\cite{a} x \\\\ y\nZ"', "\\toprule \\cite{a} x \\\\ y\nZ"),
        # model forgot to double backslashes: \t, \b, \f, \n, \r must not become control chars
        (r'"\toprule \cite{a} x \\ y\nZ"', "\\toprule \\cite{a} x \\\\ y\nZ"),
        (
            r'"\begin{table}\n\centering\n\bottomrule \frac{1}{2} \noindent \ref{t} caf\u00e9"',
            "\\begin{table}\n\\centering\n\\bottomrule \\frac{1}{2} \\noindent \\ref{t} café",
        ),
        # mostly correct with one stray: only the stray is fixed
        (
            r'"\\toprule \\midrule \\bottomrule \cite{a}"',
            "\\toprule \\midrule \\bottomrule \\cite{a}",
        ),
        (r'"A.\n\nWe \"quote\""', 'A.\n\nWe "quote"'),
        (r'"see \cref{x} and \textcite{y}"', "see \\cref{x} and \\textcite{y}"),
    ],
)
def test_latex_escape_repair(raw, latex):
    assert parse_decision('{"action":"insert","latex":' + raw + "}").latex == latex


def test_over_escaped_latex_is_unescaped_once():
    from voxtex.engine.decide import fix_over_escaped

    bad = "\\\\begin{itemize}\\n\\\\item Η αρχή.\\n\\\\end{itemize}"
    assert fix_over_escaped(bad) == "\\begin{itemize}\n\\item Η αρχή.\n\\end{itemize}"
    # normal LaTeX with line breaks is left alone
    for ok in ("a \\\\ b \\textbf{x} \\\\ c", "0.71 \\\\\n\\midrule \\noindent"):
        assert fix_over_escaped(ok) == ok
    d = parse_decision('{"action":"insert","latex":"\\\\\\\\section{A}\\\\n\\\\\\\\label{sec:a}"}')
    assert d.latex == "\\section{A}\n\\label{sec:a}"
