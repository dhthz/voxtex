"""Every key in a \\cite-like command must exist in the bibliography."""

import re
from difflib import get_close_matches

from voxtex.validate.base import CITE_CMDS, Issue, ValidationContext, command_args, split_keys


def check_citations(latex: str, ctx: ValidationContext) -> list[Issue]:
    known = {e.key for e in ctx.idx.bib}
    issues = []
    for c in command_args(latex, CITE_CMDS):
        for key in split_keys(c.arg):
            if key == "*" or key in known:
                continue
            close = get_close_matches(key, known, n=2, cutoff=0.6)
            hint = f" (did you mean {' or '.join(close)}?)" if close else ""
            issues.append(Issue("citation", f"unknown citation key '{key}'{hint}"))
    return issues


def normalize_cite_keys(latex: str, known: set[str]) -> str:
    """Fix keys that are right except for escaping or case (Berners\\-Lee01 → Berners-Lee01)."""
    by_lower = {k.lower(): k for k in known}
    out, last = [], 0
    for c in command_args(latex, CITE_CMDS):
        keys = split_keys(c.arg)
        fixed = []
        for key in keys:
            plain = re.sub(r"\\([-_&%#])", r"\1", key)
            fixed.append(key if key in known else by_lower.get(plain.lower(), key))
        if fixed != keys:
            arg_start = c.end - 1 - len(c.arg)
            out.append(latex[last:arg_start])
            out.append(",".join(fixed))
            last = c.end - 1
    out.append(latex[last:])
    return "".join(out)
