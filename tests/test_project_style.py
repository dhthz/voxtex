from pathlib import Path

from voxtex.project.discover import SourceFile, discover
from voxtex.project.style import pick_samples

PROSE = (
    "Domain adaptation of language models has become a standard step in applied work. "
    "It typically involves continued pre-training on in-domain text before fine-tuning, "
    "and several studies report consistent improvements \\parencite{a}. However, the "
    "magnitude of these gains varies across tasks and corpora, which motivates a careful "
    "comparison of the available options before committing compute to a single approach."
)


def test_fixture_samples(sample_thesis):
    s = pick_samples(discover(sample_thesis).body[1:])
    assert 1 <= len(s.paragraphs) <= 3
    assert all("\\begin" not in p.text for p in s.paragraphs)
    assert s.table and "\\toprule" in s.table.text
    assert s.figure and "\\includegraphics" in s.figure.text
    assert s.sources == ["chapters/introduction.tex", "chapters/methodology.tex"]


def test_skips_headings_lists_and_short_text():
    body = "\n\n".join(
        [
            "\\section{Intro}",
            "Short line.",
            "\\begin{itemize}\\item " + PROSE + "\\end{itemize}",
            PROSE,
        ]
    )
    s = pick_samples([SourceFile(Path("c.tex"), body)])
    assert [p.text for p in s.paragraphs] == [PROSE]
    assert s.table is None and s.figure is None


def test_single_file_thesis_keeps_body():
    body = "\\begin{document}\n\n" + PROSE + "\n\n\\end{document}"
    assert len(pick_samples([SourceFile(Path("main.tex"), body)]).paragraphs) == 1


def test_prefers_different_files():
    other = PROSE.replace("Domain", "Model")
    files = [
        SourceFile(Path("a.tex"), PROSE + "\n\n" + PROSE.replace("standard", "common")),
        SourceFile(Path("b.tex"), other),
    ]
    s = pick_samples(files)
    assert {p.file for p in s.paragraphs[:2]} == {"a.tex", "b.tex"}
