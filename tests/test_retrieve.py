from voxtex.engine.retrieve import (
    kinds_mentioned,
    macro_signature,
    match_bib,
    match_labels,
    match_macros,
    translit,
)
from voxtex.project.bib import BibEntry
from voxtex.project.labels import Label
from voxtex.project.preamble import Macro

BIB = [
    BibEntry(
        key="devlin2019", authors=["Devlin", "Chang"], year="2019", title="BERT: Pre-training"
    ),
    BibEntry(
        key="chalkidis2019",
        authors=["Chalkidis"],
        year="2019",
        title="Neural Legal Judgment Prediction",
    ),
    BibEntry(
        key="chalkidis2020", authors=["Chalkidis"], year="2020", title="LEGAL-BERT: The Muppets"
    ),
    BibEntry(
        key="vaswani2017", authors=["Vaswani"], year="2017", title="Attention Is All You Need"
    ),
]


def keys(entries):
    return [e.key for e in entries]


def test_translit():
    assert translit("Χαλκίδης") == "chalkidis"
    assert translit("Ζαρολιάγκης") == "zaroliagis"


def test_surname_returns_both_same_author_papers():
    assert set(keys(match_bib("cite the chalkidis paper", BIB))) == {
        "chalkidis2019",
        "chalkidis2020",
    }


def test_year_disambiguates():
    assert keys(match_bib("as Chalkidis showed in 2020", BIB))[0] == "chalkidis2020"


def test_greek_and_misheard_surnames():
    assert keys(match_bib("όπως έδειξε ο Χαλκίδης το 2019", BIB))[0] == "chalkidis2019"
    assert "chalkidis2020" in keys(match_bib("the Halkidis legal bert paper", BIB))


def test_title_words_without_surname():
    assert keys(match_bib("the attention is all you need paper", BIB)) == ["vaswani2017"]


def test_nothing_relevant():
    assert match_bib("we trained for three epochs", BIB) == []


def test_labels_by_name_and_kind():
    labels = [
        Label(name="tab:baselines", kind="table"),
        Label(name="fig:pipeline", kind="figure"),
        Label(name="sec:methodology", kind="chapter"),
    ]
    assert [lab.name for lab in match_labels("as in the pipeline overview", labels)] == [
        "fig:pipeline"
    ]
    assert [lab.name for lab in match_labels("βλέπε τον πίνακα", labels)] == ["tab:baselines"]
    assert kinds_mentioned("add a table and a Σχήμα") == {"table", "figure"}


def test_macros_always_named_and_used():
    macros = [
        Macro(name="\\en", arity=1, uses=78),
        Macro(name="\\legalbert"),
        Macro(name="\\softmax", kind="DeclareMathOperator"),
        Macro(name="\\baselinestretch", kind="renewcommand"),
        Macro(name="\\unused", arity=2),
    ]
    got = [m.name for m in match_macros("compare legal-bert and softmax", "", macros, ["\\en"])]
    assert got == ["\\en", "\\legalbert", "\\softmax"]


def test_macro_signature():
    assert macro_signature(Macro(name="\\metric", arity=2, optional=1)) == "\\metric[…]{…}"
    assert macro_signature(Macro(name="\\x")) == "\\x"
