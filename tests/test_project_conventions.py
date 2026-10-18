"""Golden tests: preamble (+ a little body text) -> expected rules."""

from pathlib import Path

from voxtex.project.conventions import derive_conventions
from voxtex.project.discover import SourceFile
from voxtex.project.labels import Label
from voxtex.project.preamble import parse_preamble

ENGLISH = "This is ordinary English prose about legal language models. " * 10
GREEK = "Αυτό είναι ένα κείμενο στα ελληνικά για νομικά γλωσσικά μοντέλα. " * 10


def conv(preamble: str, body: str = ""):
    pre_src = [SourceFile(Path("main.tex"), preamble)]
    body_src = [SourceFile(Path("ch.tex"), body)]
    return derive_conventions(parse_preamble(pre_src), pre_src, body_src, [])


def test_biblatex_cleveref_booktabs_siunitx():
    c = conv(
        "\\usepackage[english]{babel}\\usepackage[style=authoryear]{biblatex}"
        "\\usepackage{cleveref}\\usepackage{booktabs}\\usepackage{siunitx}"
    )
    assert (c.bib_package, c.bib_style) == ("biblatex", "authoryear")
    assert (c.cite_paren, c.cite_text) == ("\\parencite", "\\textcite")
    assert (c.ref, c.ref_capital, c.ref_style) == ("\\cref", "\\Cref", "cleveref")
    assert c.tables == "booktabs"
    assert (c.units, c.unit_cmd, c.num_cmd) == ("siunitx", "\\SI", "\\num")
    assert (c.language, c.language_source) == ("en", "babel")
    assert c.rules() == [
        "cite: \\parencite (parenthetical), \\textcite (in text)",
        "ref: \\cref{label} (\\Cref at sentence start), no ~\\ref",
        "tables: booktabs (\\toprule/\\midrule/\\bottomrule), no vertical rules",
        "units: siunitx (\\SI{value}{unit}, \\num{n})",
        "language: en",
    ]


def test_natbib():
    c = conv("\\usepackage[round]{natbib}")
    assert (c.bib_package, c.cite_paren, c.cite_text) == ("natbib", "\\citep", "\\citet")
    assert c.ref_style == "plain"


def test_plain_cite_and_hline():
    c = conv("\\usepackage{graphicx}", "\\begin{tabular}{ll}\\hline a & b\\\\\\hline\\end{tabular}")
    assert (c.bib_package, c.cite_paren, c.cite_text) == ("none", "\\cite", "\\cite")
    assert c.tables == "hline"
    assert c.rules()[0] == "cite: \\cite{key}"


def test_biblatex_numeric_default_style():
    assert conv("\\usepackage{biblatex}").bib_style == "numeric"


def test_observed_usage_overrides_package_default():
    body = (
        "As shown \\cite{a}, \\cite{b} and \\cite[p.~3]{c}; "
        "see \\autoref{x}, \\autoref{y}, \\autoref{z}."
    )
    c = conv("\\usepackage{biblatex}\\usepackage{cleveref}", body)
    assert c.cite_paren == "\\cite"
    assert c.cite_text == "\\textcite"  # nothing observed, keep the package default
    assert c.ref == "\\autoref"


def test_cleveref_loaded_but_unused():
    c = conv("\\usepackage{cleveref}", "Figure~\\ref{a}, Table~\\ref{b}, Section~\\ref{c} \\ref{d}")
    assert c.ref_style == "plain"


def test_siunitx_qty_observed():
    c = conv("\\usepackage{siunitx}", "\\qty{3}{\\s} and \\qty{4}{\\m}")
    assert c.unit_cmd == "\\qty"


def test_babel_main_language_is_last_option():
    c = conv("\\usepackage[english,greek]{babel}")
    assert (c.languages, c.language) == (["greek", "english"], "el")


def test_babel_main_option():
    c = conv("\\usepackage[greek,english,main=greek]{babel}")
    assert c.languages[0] == "greek" and c.language == "el"


def test_babel_language_from_class_options():
    pre = parse_preamble(
        [SourceFile(Path("m.tex"), "\\documentclass[greek]{report}\\usepackage{babel}")]
    )
    src = [SourceFile(Path("m.tex"), "")]
    assert derive_conventions(pre, src, src, []).language == "el"


def test_polyglossia():
    c = conv("\\usepackage{polyglossia}\\setdefaultlanguage{greek}\\setotherlanguage{english}")
    assert (c.languages, c.language, c.language_source) == (
        ["greek", "english"],
        "el",
        "polyglossia",
    )


def test_language_from_text():
    assert conv("", GREEK).language == "el"
    assert conv("", ENGLISH).language == "en"
    assert conv("", ENGLISH).language_source == "text"
    assert conv("", "too short").language_source == "default"


# --- Greek thesis in the style of the CEID Patras template -------------------------------------

CEID_PRE = (
    "\\documentclass[english,greek]{report}\\usepackage[greek]{babel}\\usepackage{amsmath}"
    "\\bibliographystyle{static/hellas}"
    "\\newcommand{\\en}[1]{\\foreignlanguage{english}{#1}}"
    "\\newcommand{\\tl}[1]{\\textlatin{#1}}"
)
CEID_BODY = (
    "Το Σχήμα~\\ref{figure4.1} δείχνει το \\en{RDF} και το \\en{XML} \\en{schema}. "
    "Ο Πίνακας~\\ref{table07.01} και η (\\ref{equation08.01}) \\tl{x} \\cite{a}."
)
CEID_LABELS = [
    Label(name="figure4.1", kind="figure"),
    Label(name="table07.01", kind="table"),
    Label(name="equation08.01", kind="equation"),
]


def ceid():
    pre_src = [SourceFile(Path("main.tex"), CEID_PRE)]
    body_src = [SourceFile(Path("ch.tex"), CEID_BODY)]
    return derive_conventions(parse_preamble(pre_src), pre_src, body_src, CEID_LABELS)


def test_bibtex_style_from_bibliographystyle():
    c = ceid()
    assert (c.bib_package, c.bib_style, c.cite_paren) == ("bibtex", "hellas", "\\cite")


def test_latin_wrapper_is_most_used_switching_macro():
    c = ceid()
    assert c.language == "el"
    assert c.latin_wrapper == "\\en"
    assert c.rules()[-1] == "English/Latin-script terms: wrap in \\en{…}"


def test_latin_wrapper_falls_back_to_textlatin():
    pre_src = [SourceFile(Path("m.tex"), "\\usepackage[greek]{babel}")]
    src = [SourceFile(Path("c.tex"), "")]
    c = derive_conventions(parse_preamble(pre_src), pre_src, src, [])
    assert c.latin_wrapper == "\\textlatin"


def test_no_latin_wrapper_for_english_or_polyglossia():
    assert conv("\\usepackage[english]{babel}").latin_wrapper == ""
    assert conv("\\usepackage{polyglossia}\\setmainlanguage{greek}").latin_wrapper == ""


def test_ref_nouns_observed_and_defaults():
    c = ceid()
    assert c.ref_nouns == {
        "figure": "Σχήμα",
        "table": "Πίνακας",
        "chapter": "Κεφάλαιο",
        "section": "Ενότητα",
    }
    assert c.eq_ref == "(\\ref{label})"  # observed, even though amsmath is loaded
    assert conv("\\usepackage{amsmath}").eq_ref == "\\eqref{label}"
    assert conv("").ref_nouns["figure"] == "Figure"


def test_label_examples_in_rules():
    rule = next(r for r in ceid().rules() if r.startswith("labels like:"))
    assert rule == "labels like: equation equation08.01, figure figure4.1, table table07.01"
