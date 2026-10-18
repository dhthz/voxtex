"""Rolling transcript buffer plus the record of blocks written this session."""

from dataclasses import dataclass, field


@dataclass
class Block:
    id: str
    latex: str
    note: str = ""  # uncertain_insert: what the user should check


@dataclass
class Buffer:
    pending: list[str] = field(default_factory=list)
    pending_since: float | None = None
    blocks: list[Block] = field(default_factory=list)  # live blocks, in document order
    _next_id: int = 1

    @property
    def pending_text(self) -> str:
        return " ".join(self.pending).strip()

    def add_utterance(self, text: str, now: float) -> None:
        text = text.strip()
        if not text:
            return
        if not self.pending:
            self.pending_since = now
        self.pending.append(text)

    def clear_pending(self) -> None:
        self.pending.clear()
        self.pending_since = None

    def new_block(self, latex: str, note: str = "") -> Block:
        b = Block(id=f"b{self._next_id}", latex=latex, note=note)
        self._next_id += 1
        self.blocks.append(b)
        return b

    def get(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    def replace(self, block_id: str, latex: str, note: str = "") -> Block | None:
        b = self.get(block_id)
        if b:
            b.latex, b.note = latex, note
        return b

    def delete(self, block_id: str) -> Block | None:
        b = self.get(block_id)
        if b:
            self.blocks.remove(b)
        return b

    def recent(self, n: int = 3) -> list[Block]:
        return self.blocks[-n:]
