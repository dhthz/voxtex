import pytest

from voxtex.project.bib import clean_latex, parse_bib, surnames


@pytest.mark.parametrize(
    "field, expected",
    [
        ("Chalkidis, Ilias and Androutsopoulos, Ion", ["Chalkidis", "Androutsopoulos"]),
        ("Jacob Devlin and Ming-Wei Chang", ["Devlin", "Chang"]),
        ("Ludwig van Beethoven", ["van Beethoven"]),
        ("Kaiser, {\\L}ukasz and others", ["Kaiser"]),
        ("{Google Research}", ["Google Research"]),
        ("M{\\\"u}ller, Hans and Gonz\\'alez, Ana", ["Müller", "González"]),
    ],
)
def test_surnames(field, expected):
    assert surnames(field) == expected


def test_clean_latex():
    assert clean_latex("{LEGAL-BERT}: The {Muppets}") == "LEGAL-BERT: The Muppets"
    assert clean_latex("{\\L}ukasz") == "Łukasz"
    assert clean_latex("\\emph{Deep}  Learning") == "Deep Learning"
    assert clean_latex("Ca\\c{c}a and \\'{\\i} and \\v Sabina") == "Caça and í and Šabina"
    assert clean_latex("see \\url{x}") == "see x"


def test_fixture_bib(sample_thesis):
    entries = {e.key: e for e in parse_bib(sample_thesis / "references.bib")}
    assert set(entries) == {
        "chalkidis2019",
        "chalkidis2020",
        "devlin2019",
        "vaswani2017",
        "zheng2021",
    }
    c = entries["chalkidis2020"]
    assert c.authors[0] == "Chalkidis" and c.year == "2020"
    assert c.title.startswith("LEGAL-BERT")
    assert (
        c.short
        == "chalkidis2020 | Chalkidis | 2020 | LEGAL-BERT: The Muppets straight out of Law School"
    )


def test_biblatex_date_and_strings(tmp_path):
    p = tmp_path / "x.bib"
    p.write_text(
        "@string{acl = {ACL}}\n"
        "@online{web2023, author={Doe, Jane}, title={A Page}, date={2023-05-01}, url={x}}\n"
        "@misc{nobody, title={Anon}}\n",
        encoding="utf-8",
    )
    entries = {e.key: e for e in parse_bib(p)}
    assert entries["web2023"].year == "2023"
    assert entries["nobody"].authors == [] and entries["nobody"].short.startswith("nobody | ? |")
