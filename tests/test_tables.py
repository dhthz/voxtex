from voxtex.editor.tablepane import _add_spec_column, _drop_spec_column
from voxtex.editor.tables import ends_with_break, find_table, parse, render

BOOKTABS = r"""Before.
\begin{table}[t]
  \centering
  \begin{tabular}{lrr}
    \toprule
    Model & Macro F1 & Micro F1 \\
    \midrule
    \textbf{legal-bert} & 0.71 & 0.80 \\
    bert-base & 0.63 \% & \multicolumn{1}{c}{0.75} \\[2pt]
    \bottomrule
  \end{tabular}
  \caption{Results \& more}\label{tab:res}
\end{table}
After."""


def table_source():
    i = BOOKTABS.index("legal-bert")
    start, end = find_table(BOOKTABS, i)
    return start, end, BOOKTABS[start:end]


def test_find_table_only_inside_it():
    start, end, source = table_source()
    assert source.startswith(r"\begin{tabular}{lrr}") and source.endswith(r"\end{tabular}")
    assert find_table(BOOKTABS, BOOKTABS.index("Before")) is None
    assert find_table(BOOKTABS, BOOKTABS.index("Results")) is None  # the caption isn't the table


def test_parse_reads_cells_rules_and_row_ends():
    t = parse(table_source()[2])
    assert t.head == r"\begin{tabular}{lrr}"
    assert t.rows == [
        ["Model", "Macro F1", "Micro F1"],
        [r"\textbf{legal-bert}", "0.71", "0.80"],
        ["bert-base", r"0.63 \%", r"\multicolumn{1}{c}{0.75}"],
    ]
    assert t.rules == {0: [r"\toprule"], 1: [r"\midrule"], 3: [r"\bottomrule"]}
    assert t.ends == ["", "", "[2pt]"]
    assert t.header_rows() == 1


def test_render_round_trips_and_aligns():
    source = table_source()[2]
    t = parse(source)
    out = render(t, ends_with_break(source))
    assert parse(out).rows == t.rows and parse(out).rules == t.rules
    assert "    Model               & Macro F1 & Micro F1 \\\\" in out
    assert out.endswith("  \\end{tabular}")
    assert r"\\[2pt]" in out


def test_column_spec_follows_column_edits():
    assert _add_spec_column(r"\begin{tabular}{| c | c |}", 0) == r"\begin{tabular}{| c | c | c |}"
    assert _add_spec_column(r"\begin{tabular}{lrr}", 0) == r"\begin{tabular}{llrr}"
    assert _drop_spec_column(r"\begin{tabular}{|l|p{3cm}|}", 0) == r"\begin{tabular}{|p{3cm}|}"
    assert _drop_spec_column(r"\begin{tabularx}{\textwidth}{lX}", 1) == (
        r"\begin{tabularx}{\textwidth}{l}"
    )
