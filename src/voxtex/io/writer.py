"""Atomic, backed-up, undoable edits to project files.

Files are read and written as raw UTF-8 bytes (a BOM survives) and new text follows
the file's own line endings. Each edit re-reads the file first, so changes the author
made in their editor in the meantime are kept.
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voxtex.io.history import CONTEXT_CHARS, Edit, History, UndoError, locate, sha
from voxtex.io.marker import MARKER

_MARKER_LINE = re.compile(r"^[ \t]*%>[ \t]*HERE\b.*$", re.M)


def read_file(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="surrogateescape")


def write_atomic(path: Path, content: str) -> None:
    """Write to a temp file in the same folder, then rename over the original."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".voxtex")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content.encode("utf-8", errors="surrogateescape"))
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def newline_of(content: str) -> str:
    return "\r\n" if "\r\n" in content else "\n"


@dataclass
class TextSpan:
    file: Path
    offset: int
    text: str


class Writer:
    def __init__(self, root: Path, history: History | None = None):
        self.root = root
        self.history = history or History(root)
        self.backups = root / ".voxtex" / "backups"
        self._backed_up: set[Path] = set()
        self.last_written: dict[Path, str] = {}  # path -> sha, so watch mode ignores own writes

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def backup(self, path: Path) -> None:
        if path in self._backed_up:
            return
        self.backups.mkdir(parents=True, exist_ok=True)
        name = f"{self.history.session}-{self.rel(path).replace('/', '__')}"
        shutil.copy2(path, self.backups / name)
        self._backed_up.add(path)

    # --- the one primitive --------------------------------------------------------------------

    def edit(
        self, path: Path, offset: int, removed: str, added: str, kind: str, block: str = ""
    ) -> Edit:
        """Replace `removed` (which must sit at `offset`) with `added`, atomically."""
        content = read_file(path)
        nl = newline_of(content)
        if nl != "\n":
            added = added.replace("\r\n", "\n").replace("\n", nl)
        if content[offset : offset + len(removed)] != removed:
            raise UndoError(f"{self.rel(path)} changed under voxtex; edit skipped")
        self.backup(path)
        new = content[:offset] + added + content[offset + len(removed) :]
        write_atomic(path, new)
        self.last_written[path.resolve()] = sha(new)
        return self.history.record(
            kind=kind,
            file=self.rel(path),
            offset=offset,
            removed=removed,
            added=added,
            before_ctx=content[max(0, offset - CONTEXT_CHARS) : offset],
            after_sha=sha(new),
            block=block,
        )

    # --- operations built on it ---------------------------------------------------------------

    def ensure_marker(self, path: Path) -> int:
        """Offset of the marker line in `path`, appending one at the end if missing."""
        content = read_file(path)
        m = _MARKER_LINE.search(content)
        if m:
            return m.start()
        nl = newline_of(content)
        sep = "" if content.endswith(nl) or not content else nl
        self.edit(path, len(content), "", f"{sep}{nl}{MARKER}{nl}", kind="marker")
        return _MARKER_LINE.search(read_file(path)).start()

    def insert_at_marker(self, path: Path, text: str, block: str) -> TextSpan:
        """Insert `text` as its own paragraph just above the marker, so the marker stays last."""
        offset = self.ensure_marker(path)
        e = self.edit(path, offset, "", text.rstrip("\n") + "\n\n", kind="insert", block=block)
        return TextSpan(path, offset, e.added.rstrip("\r\n"))

    def replace_span(
        self, span: TextSpan, new_text: str, kind: str, block: str = ""
    ) -> TextSpan | None:
        """Replace text written earlier. Found by exact match nearest its last known offset;
        None if the author has since changed it."""
        content = read_file(span.file)
        hits = [m.start() for m in re.finditer(re.escape(span.text), content)] if span.text else []
        if not hits:
            return None
        offset = min(hits, key=lambda h: abs(h - span.offset))
        removed = span.text
        if not new_text:
            # Deleting: also take the blank line voxtex added after it.
            tail = content[offset + len(removed) :]
            m = re.match(r"(\r?\n){1,2}", tail)
            if m:
                removed += m.group(0)
        e = self.edit(span.file, offset, removed, new_text, kind=kind, block=block)
        return TextSpan(span.file, offset, e.added)

    def undo(self, n: int = 1) -> tuple[list[Edit], str | None]:
        """Revert the last `n` edits (across sessions), newest first.

        Stops at the first edit that can't be undone safely; returns (undone, problem).
        """
        done = []
        for edit in reversed(self.history.pending()[-n:] if n > 0 else []):
            path = self.root / edit.file
            try:
                content = read_file(path)
                at = locate(content, edit)
            except (OSError, UndoError) as e:
                return done, str(e)
            new = content[:at] + edit.removed + content[at + len(edit.added) :]
            write_atomic(path, new)
            self.last_written[path.resolve()] = sha(new)
            self.history.mark_undone(edit)
            done.append(edit)
        return done, None
