"""Undo history in .voxtex/history.jsonl.

Every write is one edit: at `offset` in `file`, the text `removed` was replaced by
`added`. Undo applies the inverse. If the file is byte-identical to how voxtex left
it, that's exact; otherwise the added text (or, for deletions, the text just before
it) is searched for near the original offset, so undo still works after the author
edits elsewhere in the file.
"""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

CONTEXT_CHARS = 80


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Edit:
    id: str
    ts: float
    session: str
    kind: str  # insert | replace | delete | marker | watch | ignore
    file: str  # relative to the project root
    offset: int
    removed: str
    added: str
    before_ctx: str  # text just before `offset`, to relocate deletions
    after_sha: str  # hash of the whole file right after this edit
    block: str = ""


class UndoError(Exception):
    pass


class History:
    def __init__(self, root: Path, session: str | None = None):
        self.root = root
        self.path = root / ".voxtex" / "history.jsonl"
        self.session = session or time.strftime("%Y%m%d-%H%M%S")

    def record(self, **fields) -> Edit:
        edit = Edit(id=uuid.uuid4().hex[:12], ts=time.time(), session=self.session, **fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"edit": asdict(edit)}, ensure_ascii=False) + "\n")
        return edit

    def _entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def pending(self) -> list[Edit]:
        """Edits not yet undone, oldest first."""
        entries = self._entries()
        undone = {e["undo"] for e in entries if "undo" in e}
        return [Edit(**e["edit"]) for e in entries if "edit" in e and e["edit"]["id"] not in undone]

    def mark_undone(self, edit: Edit) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"undo": edit.id, "ts": time.time()}) + "\n")


def locate(content: str, edit: Edit) -> int:
    """Where the inverse of `edit` applies in `content` now."""
    if sha(content) == edit.after_sha:
        return edit.offset
    if edit.added:
        hits = _find_all(content, edit.added)
    else:
        ctx = edit.before_ctx
        hits = [h + len(ctx) for h in _find_all(content, ctx)] if ctx else []
    if not hits:
        raise UndoError(f"{edit.file} changed too much to undo this edit safely")
    return min(hits, key=lambda h: abs(h - edit.offset))


def _find_all(text: str, needle: str) -> list[int]:
    hits, start = [], 0
    while (i := text.find(needle, start)) >= 0:
        hits.append(i)
        start = i + 1
    return hits
