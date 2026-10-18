"""Optional: compile the snippet inside the real preamble and report LaTeX errors.

Runs from the project root (so local .cls/.sty files and \\input paths resolve) with
output in a temp directory. Undefined references/citations are fine (there's no
bibliography run); missing image files are warnings, since figures are often dictated
before the image exists.
"""

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from voxtex.project.index import ProjectIndex
from voxtex.project.latex import read_text
from voxtex.validate.base import Issue

TIMEOUT_S = 30
MAX_ERRORS = 3
JOB = "voxtex-check"

_BEGIN_DOC = re.compile(r"\\begin\s*\{document\}")
_MISSING_FILE = re.compile(r"File [`'](.+?)' not found", re.I)
_UNICODE_ENGINES = ("fontspec", "polyglossia", "unicode-math")


def pick_engine(idx: ProjectIndex) -> str | None:
    """xelatex for fontspec/polyglossia projects, otherwise pdflatex; None if TeX is missing."""
    unicode = any(idx.preamble.has(p) for p in _UNICODE_ENGINES)
    candidates = ("xelatex", "lualatex") if unicode else ("pdflatex",)
    return next((e for e in candidates if shutil.which(e)), None)


class Compiler:
    def __init__(self, idx: ProjectIndex, engine: str | None = None):
        self.idx = idx
        self.engine = engine or pick_engine(idx)
        self.root = idx.root_path
        main = read_text(self.root / idx.main)
        m = _BEGIN_DOC.search(main)
        self.preamble = main[: m.start()] if m else main
        self.preamble_lines = self.preamble.count("\n") + 1  # + the \begin{document} line
        self.last_seconds = 0.0

    @property
    def available(self) -> bool:
        return self.engine is not None

    def document(self, latex: str) -> str:
        return f"{self.preamble}\\begin{{document}}\n{latex}\n\\end{{document}}\n"

    def check(self, latex: str) -> list[Issue]:
        if not self.available:
            return []
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="voxtex-") as tmp:
            tex = Path(tmp) / f"{JOB}.tex"
            tex.write_text(self.document(latex), encoding="utf-8")
            cmd = [
                self.engine,
                "-interaction=nonstopmode",
                "-draftmode" if self.engine == "pdflatex" else "-no-pdf",
                f"-output-directory={tmp}",
                f"-jobname={JOB}",
                str(tex),
            ]
            try:
                subprocess.run(
                    cmd, cwd=self.root, capture_output=True, timeout=TIMEOUT_S, check=False
                )
            except subprocess.TimeoutExpired:
                self.last_seconds = time.monotonic() - start
                return [Issue("compile", f"compile check timed out after {TIMEOUT_S}s", False)]
            log_path = Path(tmp) / f"{JOB}.log"
            log = (
                log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            )
        self.last_seconds = time.monotonic() - start
        return self.parse_log(log)

    def parse_log(self, log: str) -> list[Issue]:
        issues: list[Issue] = []
        lines = log.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("! "):
                continue
            msg = line[2:].strip()
            if msg.startswith(("Emergency stop", "==> Fatal error")):
                continue
            where, context = "", ""
            for follow in lines[i + 1 : i + 8]:
                m = re.match(r"l\.(\d+) ?(.*)", follow)
                if m:
                    snippet_line = int(m.group(1)) - self.preamble_lines
                    if snippet_line > 0:
                        where = f" (line {snippet_line} of the output)"
                    context = m.group(2).strip()
                    break
            if msg.startswith("Undefined control sequence") and context:
                cmds = re.findall(r"\\[A-Za-z@]+", context)
                if cmds:
                    msg = f"undefined command {cmds[-1]}"
            missing = _MISSING_FILE.search(msg)
            if missing:
                issues.append(Issue("compile", f"file '{missing.group(1)}' not found", False))
                continue
            msg = msg.removeprefix("LaTeX Error: ").removeprefix("Package ")
            issue = Issue("compile", f"LaTeX error: {msg}{where}")
            if all(issue.message != x.message for x in issues):
                issues.append(issue)
            if sum(1 for x in issues if x.error) >= MAX_ERRORS:
                break
        return issues
