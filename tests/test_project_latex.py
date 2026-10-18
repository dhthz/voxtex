from voxtex.project.latex import read_group, read_name_arg, read_optional, strip_comments


def test_strip_comments_keeps_escaped_percent():
    assert strip_comments("50\\% done % a comment\nnext") == "50\\% done \nnext"


def test_read_group_nested_and_escaped():
    text = r"  {a {b} \} c}rest"
    content, end = read_group(text, 0)
    assert content == r"a {b} \} c"
    assert text[end:] == "rest"


def test_read_group_absent_or_unbalanced():
    assert read_group("x{a}", 0) is None
    assert read_group("{a", 0) is None


def test_read_optional_with_braces_inside():
    content, end = read_optional("[a={x]y}, b]{z}", 0)
    assert content == "a={x]y}, b"


def test_read_name_arg_both_forms():
    assert read_name_arg(r"{\foo}[1]", 0)[0] == r"\foo"
    assert read_name_arg(r"\bar[1]", 0)[0] == r"\bar"
