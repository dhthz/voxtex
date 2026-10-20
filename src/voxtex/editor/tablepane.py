"""The table pane: the table around the cursor as a grid you can edit cell by cell.
Every change is written back into the chapter as LaTeX (see tables.py)."""

import copy
import re

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.widgets import DataTable, Input, Static

from voxtex.editor import words
from voxtex.editor.tables import Table


def shown(cell: str, bold: bool, pal) -> Text:
    """A cell as it looks in the PDF: its visible text (maths and odd commands as written)."""
    if not cell.strip():
        return Text("·", style=pal.overlay0)  # empty, but there
    text, _ = words.visible_text(cell)
    text = " ".join(text.split()) or cell.strip()
    bold = bold or "\\textbf" in cell or "\\bfseries" in cell
    return Text(text, style=f"bold {pal.text}" if bold else pal.text)


class TablePane(Vertical):
    """Arrows move between cells; ↵ edits one; r/c add a row/column after the cursor,
    d/x delete them; esc goes back to the text."""

    DEFAULT_CSS = """
    TablePane #shape { height: 1; color: $vx-overlay0; }
    TablePane DataTable {
        height: auto; max-height: 30; background: ansi_default; scrollbar-size: 1 1;
    }
    TablePane DataTable > .datatable--cursor { background: $vx-surface1; }
    TablePane DataTable > .datatable--hover { background: $vx-surface0; }
    TablePane #cell { height: 1; border: none; padding: 0; display: none;
        background: $vx-surface0; color: $vx-text; }
    TablePane #cell:focus { border: none; }
    """
    BINDINGS = [
        Binding("r", "add_row", "row", show=False),
        Binding("c", "add_column", "column", show=False),
        Binding("d", "delete_row", "delete row", show=False),
        Binding("x", "delete_column", "delete column", show=False),
        Binding("escape", "back", "back", show=False),
    ]

    class Edited(Message):
        """The table changed here; the app writes it back into the chapter."""

        def __init__(self, table: Table) -> None:
            super().__init__()
            self.table = table

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table: Table | None = None
        self.source = ""
        self.editing: Coordinate | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="shape")
        yield DataTable(id="cells", cursor_type="cell", show_header=False)
        yield Input(id="cell", placeholder="the cell's LaTeX")

    @property
    def grid(self) -> DataTable:
        return self.query_one("#cells", DataTable)

    def show(self, table: Table, source: str) -> None:
        """Show `table` (read from `source`); keeps the cursor on the same cell."""
        if source == self.source:
            return
        self.table, self.source = table, source
        pal = self.app.pal
        grid = self.grid
        cursor = grid.cursor_coordinate
        grid.clear(columns=True)
        width = max(table.columns, 1)
        grid.add_columns(*[str(i + 1) for i in range(width)])
        head = table.header_rows()
        for i, row in enumerate(table.rows):
            cells = [shown(c, i < head, pal) for c in row]
            grid.add_row(*(cells + [Text("")] * (width - len(cells)))[:width])
        if table.rows:
            grid.cursor_coordinate = Coordinate(
                min(cursor.row, len(table.rows) - 1), min(cursor.column, width - 1)
            )
        rules = sum(len(r) for r in table.rules.values())
        self.query_one("#shape", Static).update(
            f"{table.env} · {len(table.rows)} rows × {width} columns"
            + (f" · {rules} rules" if rules else "")
        )

    # --- editing -------------------------------------------------------------------------

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """↵ on a cell: edit its LaTeX in the box below the grid."""
        if self.table is None:
            return
        r, c = event.coordinate
        row = self.table.rows[r] if r < len(self.table.rows) else []
        self.editing = event.coordinate
        box = self.query_one("#cell", Input)
        box.value = row[c] if c < len(row) else ""
        box.display = True
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.editing is None or self.table is None:
            return
        r, c = self.editing
        table = copy.deepcopy(self.table)
        row = table.rows[r]
        row.extend([""] * (c + 1 - len(row)))
        row[c] = event.value.strip()
        self._done_editing()
        self.post_message(self.Edited(table))

    def _done_editing(self) -> None:
        self.editing = None
        box = self.query_one("#cell", Input)
        box.display = False
        self.grid.focus()

    def _changed(self, change) -> None:
        if self.table is None:
            return
        table = copy.deepcopy(self.table)
        change(table, self.grid.cursor_coordinate)
        self.post_message(self.Edited(table))

    def action_add_row(self) -> None:
        def add(t: Table, at: Coordinate) -> None:
            t.rows.insert(at.row + 1, [""] * t.columns)
            t.ends.insert(at.row + 1, "")
            t.rules = {(k + 1 if k > at.row + 1 else k): v for k, v in t.rules.items()}

        self._changed(add)

    def action_add_column(self) -> None:
        def add(t: Table, at: Coordinate) -> None:
            for row in t.rows:
                row.insert(min(at.column + 1, len(row)), "")
            t.head = _add_spec_column(t.head, at.column)

        self._changed(add)

    def action_delete_row(self) -> None:
        def delete(t: Table, at: Coordinate) -> None:
            if len(t.rows) > 1:
                del t.rows[at.row]
                del t.ends[at.row]
                t.rules = {(k - 1 if k > at.row else k): v for k, v in t.rules.items()}

        self._changed(delete)

    def action_delete_column(self) -> None:
        def delete(t: Table, at: Coordinate) -> None:
            if t.columns > 1:
                for row in t.rows:
                    if at.column < len(row):
                        del row[at.column]
                t.head = _drop_spec_column(t.head, at.column)

        self._changed(delete)

    def action_back(self) -> None:
        if self.editing is not None:
            self._done_editing()
            return
        self.app.area.focus()


# The column spec: one letter per column (l c r X, or p{..}/m{..}/b{..}), plus | and @{..}.
_SPEC_COLUMN = r"[lcrXLCRJS]|[pmbw]\{[^}]*\}"


def _spec_parts(head: str) -> tuple[str, str, str] | None:
    """(before the spec, the spec, after it): the last {...} of the \\begin line."""
    end = head.rfind("}")
    start = head.rfind("{", 0, end)
    if end < 0 or start < 0:
        return None
    # A spec with p{3cm} has nested braces: walk back to the brace that opens the spec.
    depth = 0
    for i in range(end, -1, -1):
        if head[i] == "}":
            depth += 1
        elif head[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    return head[: start + 1], head[start + 1 : end], head[end:]


def _columns_in(spec: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(_SPEC_COLUMN, spec)]


def _add_spec_column(head: str, after: int) -> str:
    """A copy of column `after`'s letter right after it, with the spec's own separator
    (`| c | c |` → `| c | c | c |`, `lr` → `llr`)."""
    parts = _spec_parts(head)
    if parts is None:
        return head
    before, spec, rest = parts
    cols = _columns_in(spec)
    if not cols:
        return head
    i = min(after, len(cols) - 1)
    start, end = cols[i]
    if i + 1 < len(cols):
        sep = spec[end : cols[i + 1][0]]
    elif i > 0:
        sep = spec[cols[i - 1][1] : start]
    else:
        sep = ""
    letter = spec[start:end] if end - start == 1 else "l"
    return before + spec[:end] + sep + letter + spec[end:] + rest


def _drop_spec_column(head: str, column: int) -> str:
    parts = _spec_parts(head)
    if parts is None:
        return head
    before, spec, rest = parts
    cols = _columns_in(spec)
    if column >= len(cols):
        return head
    start, end = cols[column]
    if column + 1 < len(cols):  # take the separator after it too
        end = cols[column + 1][0]
    elif column > 0:
        start = cols[column - 1][1]
    return before + spec[:start] + spec[end:] + rest
