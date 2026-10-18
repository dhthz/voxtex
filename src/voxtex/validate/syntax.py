"""Structural checks: balanced braces, \\begin/\\end pairs, math delimiters, no preamble."""

import re

from voxtex.project.latex import strip_comments
from voxtex.validate.base import Issue, ValidationContext

_ENV = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")
_FORBIDDEN = re.compile(
    r"\\(documentclass|usepackage|RequirePackage|begin\s*\{document\}|end\s*\{document\}|"
    r"newcommand|renewcommand|providecommand|DeclareMathOperator|NewDocumentCommand|def)"
    r"(?![A-Za-z@])"
)
_VERBATIM = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted)\}.*?\\end\{\1\}|\\verb(.).*?\2", re.S
)


def check_syntax(latex: str, ctx: ValidationContext | None = None) -> list[Issue]:
    text = _VERBATIM.sub("", strip_comments(latex))
    issues: list[Issue] = []

    depth = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2  # skip escaped char: \{ \} \\ \%
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                issues.append(Issue("syntax", "unmatched '}'"))
                depth = 0
        i += 1
    if depth > 0:
        issues.append(Issue("syntax", f"{depth} unclosed '{{'"))

    stack: list[str] = []
    for m in _ENV.finditer(text):
        kind, name = m.group(1), m.group(2).strip()
        if kind == "begin":
            stack.append(name)
        elif not stack:
            issues.append(Issue("syntax", f"\\end{{{name}}} without \\begin"))
        elif stack[-1] != name:
            issues.append(Issue("syntax", f"\\begin{{{stack[-1]}}} closed by \\end{{{name}}}"))
            if name in stack:  # unwind to the matching \begin
                while stack.pop() != name:
                    pass
            else:
                stack.pop()
        else:
            stack.pop()
    for name in stack:
        issues.append(Issue("syntax", f"\\begin{{{name}}} is never closed"))

    no_escaped = re.sub(r"\\\$", "", text)
    if no_escaped.replace("$$", "").count("$") % 2:
        issues.append(Issue("syntax", "unbalanced $ (inline math not closed)"))
    if no_escaped.count("$$") % 2:
        issues.append(Issue("syntax", "unbalanced $$"))
    if len(re.findall(r"\\\[", text)) != len(re.findall(r"\\\]", text)):
        issues.append(Issue("syntax", "unbalanced \\[ \\]"))
    if len(re.findall(r"\\left(?![a-z])", text)) != len(re.findall(r"\\right(?![a-z])", text)):
        issues.append(Issue("syntax", "unbalanced \\left / \\right"))

    for m in _FORBIDDEN.finditer(text):
        issues.append(Issue("syntax", f"\\{m.group(1)} is not allowed in body text"))
    if "```" in latex:
        issues.append(Issue("syntax", "markdown code fence in output"))
    return issues
