"""Every key in a \\cite-like command must exist in the bibliography."""

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
