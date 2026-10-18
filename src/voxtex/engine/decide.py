"""The model's decision, validated with pydantic, and lenient JSON extraction."""

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from voxtex.providers import ParseError


class Wait(BaseModel):
    action: Literal["wait"]


class Ignore(BaseModel):
    action: Literal["ignore"]


class Insert(BaseModel):
    action: Literal["insert"]
    latex: str = Field(min_length=1)


class Replace(BaseModel):
    action: Literal["replace"]
    block_id: str
    latex: str = Field(min_length=1)


class Delete(BaseModel):
    action: Literal["delete"]
    block_id: str


class UncertainInsert(BaseModel):
    action: Literal["uncertain_insert"]
    latex: str = Field(min_length=1)
    note: str = ""


Decision = Annotated[
    Wait | Ignore | Insert | Replace | Delete | UncertainInsert, Field(discriminator="action")
]
_ADAPTER: TypeAdapter[Decision] = TypeAdapter(Decision)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _first_object(text: str) -> str | None:
    """The first balanced {...} in text, respecting JSON strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_JSON_ESCAPES = set('"\\/bfnrtu')


def _escape_kind(blob: str, i: int) -> str:
    """Classify the escape starting at blob[i] == '\\' inside a JSON string.

    'latex'  — really a LaTeX command the model forgot to escape (\\toprule, \\cite, \\ref)
    'json'   — a genuine JSON escape (\\n before a capital/space, \\", \\u00e9)
    'pair'   — an escaped backslash (\\\\)
    """
    c = blob[i + 1] if i + 1 < len(blob) else ""
    nxt = blob[i + 2] if i + 2 < len(blob) else ""
    if c == "\\":
        return "pair"
    if c == "u":
        return "json" if re.fullmatch(r"[0-9a-fA-F]{4}", blob[i + 2 : i + 6]) else "latex"
    if c not in _JSON_ESCAPES:
        return "latex" if c.isalpha() else "json"
    # \\b \\f \\n \\r \\t followed by a lowercase letter is almost always \\begin, \\frac,
    # \\noindent, \\ref, \\toprule...; before anything else it's a real control character.
    return "latex" if c in "bfnrt" and nxt.islower() else "json"


def _string_escapes(blob: str):
    """Yield (index, kind) for every backslash escape inside JSON string literals."""
    in_str, i = False, 0
    while i < len(blob):
        c = blob[i]
        if not in_str:
            in_str = c == '"'
            i += 1
            continue
        if c == '"':
            in_str = False
        elif c == "\\":
            kind = _escape_kind(blob, i)
            yield i, kind
            i += 1 if kind == "latex" else 2
            continue
        i += 1


def repair_escapes(blob: str) -> str:
    """Fix JSON whose LaTeX backslashes weren't doubled.

    Mostly-correct output (\\\\toprule ... one stray \\cite) only gets the strays fixed.
    Mostly-unescaped output is taken literally: every backslash is LaTeX (\\\\ is a
    line break), except real \\n newlines, \\" and \\uXXXX.
    """
    escapes = list(_string_escapes(blob))
    sloppy = sum(1 for _, kind in escapes if kind == "latex")
    if not sloppy:
        return blob
    proper = sum(1 for i, kind in escapes if kind == "pair" and blob[i + 2 : i + 3].isalpha())
    out, last = [], 0
    for i, kind in escapes:
        if proper >= sloppy and kind != "latex":
            continue  # keep as written
        out.append(blob[last:i])
        c = blob[i + 1] if i + 1 < len(blob) else ""
        if kind == "pair":
            out.append("\\\\\\\\")  # LaTeX \\\\ -> JSON "\\\\\\\\"
            last = i + 2
        elif kind == "json" and c in '"nu':
            out.append(blob[i : i + 2])
            last = i + 2
        else:
            out.append("\\\\")
            last = i + 1
    out.append(blob[last:])
    return "".join(out)


def parse_decision(text: str) -> Decision:
    """Parse the model's reply; raises ParseError so the chain can retry."""
    cleaned = _FENCE.sub("", text.strip())
    blob = _first_object(cleaned)
    if blob is None:
        raise ParseError(f"no JSON object in reply: {text[:120]!r}")
    try:
        data = json.loads(repair_escapes(blob))
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from e
    if isinstance(data, dict) and isinstance(data.get("action"), str):
        data["action"] = data["action"].strip().lower()
    try:
        return _ADAPTER.validate_python(data)
    except ValidationError as e:
        raise ParseError(f"bad decision: {e.errors()[0]['msg']}") from e
