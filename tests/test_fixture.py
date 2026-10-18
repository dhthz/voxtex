"""Sanity checks that the fixture project has what later milestones rely on."""


def test_fixture_layout(sample_thesis):
    assert (sample_thesis / "main.tex").is_file()
    assert (sample_thesis / "references.bib").is_file()
    assert (sample_thesis / "chapters" / "introduction.tex").is_file()
    assert (sample_thesis / "chapters" / "methodology.tex").is_file()


def test_fixture_has_marker(sample_thesis):
    assert "%> HERE" in (sample_thesis / "chapters" / "methodology.tex").read_text()


def test_fixture_has_same_author_ambiguity(sample_thesis):
    bib = (sample_thesis / "references.bib").read_text()
    assert "@inproceedings{chalkidis2019" in bib
    assert "@inproceedings{chalkidis2020" in bib
