"""Validation of model output before anything touches a file (plan §5.4)."""

import re
import time
from dataclasses import dataclass, field

from voxtex.project.index import ProjectIndex
from voxtex.validate.base import Issue, ValidationContext
from voxtex.validate.citations import check_citations
from voxtex.validate.compile import Compiler
from voxtex.validate.numbers import check_numbers
from voxtex.validate.refs import check_labels, check_refs, labels_in
from voxtex.validate.syntax import check_syntax

__all__ = ["Issue", "Report", "ValidationContext", "Validator", "labels_in"]

CHECKS = (check_syntax, check_citations, check_refs, check_labels, check_numbers)


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    compiled: bool = False
    seconds: float = 0.0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.error]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if not i.error]

    @property
    def ok(self) -> bool:
        return not self.errors


class Validator:
    """compile: 'auto' (only when the output uses commands/environments the thesis never
    uses), 'always', or 'off'. Compiling needs pdflatex/xelatex and costs ~1s."""

    def __init__(self, idx: ProjectIndex, compile: str = "auto"):
        self.idx = idx
        self.mode = compile
        self.compiler = Compiler(idx) if compile != "off" else None
        self.known = set(idx.known_commands)

    def needs_compile(self, latex: str) -> bool:
        if not (self.compiler and self.compiler.available):
            return False
        if self.mode == "always":
            return True
        cmds = set(re.findall(r"\\([A-Za-z@]+)", latex))
        envs = {"env:" + e for e in re.findall(r"\\begin\s*\{([^}]+)\}", latex)}
        return bool((cmds | envs) - self.known)

    def validate(self, latex: str, ctx: ValidationContext) -> Report:
        start = time.monotonic()
        report = Report()
        for check in CHECKS:
            report.issues += check(latex, ctx)
        # A compile would only repeat structural errors, so skip it then.
        if not any(i.code == "syntax" for i in report.errors) and self.needs_compile(latex):
            report.issues += self.compiler.check(latex)
            report.compiled = True
        report.seconds = time.monotonic() - start
        return report
