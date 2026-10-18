"""Cross-references must point at existing labels; new labels must be unique and on-pattern."""

import re

from voxtex.project.discover import SourceFile
from voxtex.project.labels import collect_labels, split_prefix
from voxtex.validate.base import REF_CMDS, Issue, ValidationContext, command_args, split_keys

_LABEL = re.compile(r"\\label\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")


def labels_in(latex: str) -> list[str]:
    return [m.group(1).strip() for m in _LABEL.finditer(latex)]


def check_refs(latex: str, ctx: ValidationContext) -> list[Issue]:
    known = {lab.name for lab in ctx.idx.labels} | ctx.other_labels | set(labels_in(latex))
    issues = []
    for c in command_args(latex, REF_CMDS):
        for target in split_keys(c.arg):
            if target not in known:
                issues.append(Issue("ref", f"\\{c.name}{{{target}}} points to no existing label"))
    return issues


def check_labels(latex: str, ctx: ValidationContext) -> list[Issue]:
    project = {lab.name for lab in ctx.idx.labels}
    # A label that was already in the block being replaced isn't a duplicate.
    previously_here = set(labels_in(ctx.replaced))
    conv = ctx.idx.conventions.labels
    issues = []
    seen: set[str] = set()
    for lab in collect_labels([SourceFile(path=ctx.idx.root_path / "<output>", text=latex)]):
        name = lab.name
        if name in seen or (name in project | ctx.other_labels and name not in previously_here):
            issues.append(Issue("label", f"label '{name}' already exists"))
        seen.add(name)
        want = conv.prefix_by_kind.get(lab.kind)
        if not want:
            continue
        got = split_prefix(name)
        example = conv.example_by_kind.get(lab.kind, want + conv.separator + "…")
        if got is None or got[0] != want or got[1] != conv.separator:
            issues.append(
                Issue(
                    "label", f"{lab.kind} label '{name}' doesn't follow the pattern, e.g. {example}"
                )
            )
    return issues
