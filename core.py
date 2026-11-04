"""Pure logic for preserve-whitespace (no Hermes imports, so it is testable on its own).

Capture: pull the model's raw assistant text out of a provider response.
Restore: in an outgoing request, replace an assistant text ``t`` with a recorded raw text ``r``
only when ``r.strip() == t``. ``str.strip`` removes outer whitespace and nothing else, so the
replacement can only re-add leading/trailing whitespace. Text Hermes changed in any other way
(think blocks removed, secrets redacted, ...) has no matching key and is left untouched.
"""

from __future__ import annotations

import fnmatch
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

# Content-part types that carry assistant text (Chat Completions blocks / Responses parts).
_TEXT_PART_TYPES = {"text", "output_text", "input_text"}
# Mirrors agent/codex_responses_adapter.py _OUTPUT_TEXT_TYPES (what Hermes reads on capture).
_RESPONSES_CAPTURE_TYPES = {"output_text", "text"}
DEFAULT_API_MODES = ("chat_completions", "codex_responses")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def worth_recording(raw: Any) -> bool:
    """Only texts whose trim actually changes something, and that are not whitespace-only."""
    return isinstance(raw, str) and bool(raw.strip()) and raw != raw.strip()


# ---------------------------------------------------------------- capture

def extract_raw_texts(response: Any) -> list[str]:
    """Raw assistant texts in a provider response, in output order.

    Chat Completions (SDK object, dict, or Hermes' assembled stream namespace):
    ``choices[0].message.content``. Responses API: one text per assistant ``message`` item,
    joined exactly the way Hermes joins it before trimming (``_extract_responses_message_text``),
    or ``output_text`` when the output list is empty.
    """
    choices = _field(response, "choices")
    if isinstance(choices, list) and choices:
        content = _field(_field(choices[0], "message"), "content")
        return [content] if isinstance(content, str) else []

    output = _field(response, "output")
    if isinstance(output, list) and output:
        texts = []
        for item in output:
            if _field(item, "type") != "message" or (_field(item, "role") or "assistant") != "assistant":
                continue
            chunks = []
            for part in _as_list(_field(item, "content")):
                ptype = _field(part, "type")
                text = _field(part, "refusal") if ptype == "refusal" else (
                    _field(part, "text") if ptype in _RESPONSES_CAPTURE_TYPES else None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                texts.append("".join(chunks))
        return texts

    out_text = _field(response, "output_text")
    return [out_text] if isinstance(out_text, str) else []


# ---------------------------------------------------------------- restore

class _Slot:
    """One assistant text location in a request: a string ``content`` or a single text part."""

    __slots__ = ("holder", "key")

    def __init__(self, holder: dict, key: str):
        self.holder, self.key = holder, key

    @property
    def text(self) -> str:
        return self.holder[self.key]

    def set(self, value: str) -> None:
        self.holder[self.key] = value


def _slot_for(message: dict) -> Optional[_Slot]:
    content = message.get("content")
    if isinstance(content, str):
        return _Slot(message, "content")
    if isinstance(content, list):
        # Only an unambiguous single text part (Hermes stores one part; cache markers keep one block).
        text_parts = [p for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if len(text_parts) == 1 and text_parts[0].get("type", "text") in _TEXT_PART_TYPES:
            return _Slot(text_parts[0], "text")
    return None


def _assistant_slots(request: dict) -> list[_Slot]:
    """Assistant text slots in ``messages`` (Chat Completions) or ``input`` (Responses).

    The final row is skipped when it is an assistant row: some APIs reject trailing whitespace on
    a final assistant message (prefill), and nothing after it depends on its exact bytes.
    """
    rows = request.get("messages")
    if not isinstance(rows, list):
        rows = request.get("input")
    if not isinstance(rows, list):
        return []
    slots = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("role") != "assistant":
            continue
        if row.get("type", "message") != "message":
            continue
        if idx == len(rows) - 1:
            continue
        slot = _slot_for(row)
        if slot is not None and slot.text:
            slots.append(slot)
    return slots


def restore_request(request: dict, records: Iterable[tuple[str, str]]) -> int:
    """Restore outer whitespace in ``request`` in place; returns how many texts were changed.

    ``records`` are ``(key, raw)`` pairs in capture order, ``key == raw.strip()``. When a key maps
    to one distinct raw text, every match gets it. When the same key was emitted with different
    whitespace, matches are aligned with the most recent records (older turns are the ones
    compaction removes); a match that cannot be aligned is left unchanged.
    """
    by_key: dict[str, list[str]] = defaultdict(list)
    for key, raw in records:
        if raw.strip() == key:
            by_key[key].append(raw)
    if not by_key:
        return 0

    slots = [s for s in _assistant_slots(request) if s.text in by_key]
    occurrences: dict[str, int] = defaultdict(int)
    for s in slots:
        occurrences[s.text] += 1
    seen: dict[str, int] = defaultdict(int)
    changed = 0
    for s in slots:
        key = s.text
        raws = by_key[key]
        i = seen[key]
        seen[key] += 1
        if len(set(raws)) == 1:
            raw = raws[0]
        else:
            idx = len(raws) - occurrences[key] + i
            if idx < 0:
                continue
            raw = raws[idx]
        if raw != key and raw.strip() == key:  # the whitespace-only invariant, checked per write
            s.set(raw)
            changed += 1
    return changed


# ---------------------------------------------------------------- routing

def _matches_any(value: Any, patterns: Iterable[str]) -> bool:
    value = str(value or "").lower()
    return any(fnmatch.fnmatchcase(value, str(p).lower()) for p in patterns)


def route_enabled(settings: dict, *, model: Any, provider: Any, base_url: Any, api_mode: Any) -> bool:
    """True when this request's route is one the operator listed.

    ``models`` / ``providers`` / ``base_urls`` are glob lists; every non-empty list must match.
    With all three empty the plugin does nothing unless ``all_routes: true``.
    """
    if not settings.get("enabled", True):
        return False
    api_modes = settings.get("api_modes") or DEFAULT_API_MODES
    if str(api_mode or "chat_completions") not in api_modes:
        return False
    rules = [(settings.get("models") or [], model), (settings.get("providers") or [], provider),
             (settings.get("base_urls") or [], base_url)]
    if not any(patterns for patterns, _ in rules):
        return bool(settings.get("all_routes", False))
    return all(_matches_any(value, patterns) for patterns, value in rules if patterns)


# ---------------------------------------------------------------- storage

class RecordStore:
    """Per-session ``(key, raw)`` records. In memory, mirrored to SQLite when ``db_path`` is given
    so a resumed session (new process) still restores. Sessions without an id stay in memory."""

    def __init__(self, db_path: Optional[Path] = None, retain_days: float = 30.0):
        self._lock = threading.Lock()
        self._mem: dict[str, list[tuple[str, str]]] = {}
        self._db: Optional[sqlite3.Connection] = None
        if db_path is not None:
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, key TEXT NOT NULL, raw TEXT NOT NULL, created REAL NOT NULL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS records_session ON records(session_id, seq)")
            if retain_days and retain_days > 0:
                self._db.execute("DELETE FROM records WHERE created < ?", (time.time() - retain_days * 86400,))
            self._db.commit()

    def _load(self, session_id: str) -> list[tuple[str, str]]:
        if session_id not in self._mem:
            rows: list[tuple[str, str]] = []
            if self._db is not None and session_id:
                rows = [tuple(r) for r in self._db.execute(
                    "SELECT key, raw FROM records WHERE session_id = ? ORDER BY seq", (session_id,))]
            self._mem[session_id] = rows
        return self._mem[session_id]

    def record(self, session_id: str, raw: str) -> bool:
        if not worth_recording(raw):
            return False
        key = raw.strip()
        with self._lock:
            rows = self._load(session_id)
            if rows and rows[-1] == (key, raw):  # a retry of the same text adds nothing
                return False
            rows.append((key, raw))
            if self._db is not None and session_id:
                self._db.execute("INSERT INTO records (session_id, key, raw, created) VALUES (?, ?, ?, ?)",
                                 (session_id, key, raw, time.time()))
                self._db.commit()
        return True

    def records(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._load(session_id))
