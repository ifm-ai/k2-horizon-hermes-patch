"""Unit tests for core.py (no Hermes needed): ``pytest tests/test_core.py``."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import RecordStore, extract_raw_texts, restore_request, route_enabled, worth_recording  # noqa: E402

RAW = "\n\nhello\n"


def chat_request(*assistant_texts, tail_user=True):
    msgs = [{"role": "system", "content": "sys"}]
    for i, t in enumerate(assistant_texts):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": t}]
    if tail_user:
        msgs.append({"role": "user", "content": "next"})
    return {"model": "m", "messages": msgs}


# ---- capture
def test_extract_chat_stream_namespace_and_dict():
    ns = NS(choices=[NS(message=NS(content=RAW, tool_calls=[]))])
    assert extract_raw_texts(ns) == [RAW]
    assert extract_raw_texts({"choices": [{"message": {"content": RAW}}]}) == [RAW]
    assert extract_raw_texts(NS(choices=[NS(message=NS(content=None))])) == []


def test_extract_responses_items_join_like_hermes():
    resp = NS(output=[
        NS(type="reasoning", summary=[]),
        NS(type="message", role="assistant", content=[NS(type="output_text", text="\nA"), NS(type="output_text", text="B\n")]),
        NS(type="function_call", name="x", arguments="{}"),
        NS(type="message", role="assistant", content=[NS(type="refusal", refusal=" no ")]),
    ])
    assert extract_raw_texts(resp) == ["\nAB\n", " no "]
    assert extract_raw_texts(NS(output=[], output_text="\nhi\n")) == ["\nhi\n"]


def test_worth_recording():
    assert worth_recording(RAW)
    assert not worth_recording("hello")       # nothing to restore
    assert not worth_recording("\n \n")       # whitespace-only: Hermes drops/pads these, leave alone
    assert not worth_recording(None)


# ---- restore: the whitespace-only guarantee
def test_restores_exact_raw():
    req = chat_request("hello")
    assert restore_request(req, [("hello", RAW)]) == 1
    assert req["messages"][2]["content"] == RAW


def test_changed_text_is_never_touched():
    # Hermes removed a think block / redacted a secret: stored text != raw.strip() -> no key match.
    req = chat_request("hello", "key=***")
    before = copy.deepcopy(req)
    records = [("<think>x</think>\n\nhello", "<think>x</think>\n\nhello\n"), ("key=sk-abc", "\nkey=sk-abc\n")]
    assert restore_request(req, records) == 0
    assert req == before


def test_forged_record_cannot_change_characters():
    req = chat_request("hello")
    assert restore_request(req, [("hello", "\nHELLO\n")]) == 0  # raw.strip() != key -> ignored
    assert req["messages"][2]["content"] == "hello"


def test_user_and_tool_rows_untouched():
    req = {"messages": [{"role": "user", "content": "hello"}, {"role": "tool", "content": "hello"},
                        {"role": "assistant", "content": "hello"}, {"role": "user", "content": "x"}]}
    assert restore_request(req, [("hello", RAW)]) == 1
    assert [m["content"] for m in req["messages"]] == ["hello", "hello", RAW, "x"]


def test_final_assistant_row_skipped():
    req = chat_request("hello", tail_user=False)
    assert restore_request(req, [("hello", RAW)]) == 0


def test_single_text_part_list_content():
    req = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]},
                        {"role": "user", "content": "x"}]}
    assert restore_request(req, [("hello", RAW)]) == 1
    assert req["messages"][0]["content"][0] == {"type": "text", "text": RAW, "cache_control": {"type": "ephemeral"}}


def test_tool_call_turn_keeps_tool_calls():
    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    req = {"messages": [{"role": "assistant", "content": "hello", "tool_calls": tc},
                        {"role": "tool", "tool_call_id": "c1", "content": "ok"}]}
    assert restore_request(req, [("hello", RAW)]) == 1
    assert req["messages"][0]["tool_calls"] == tc


def test_ambiguous_key_aligned_to_latest_records():
    # Same visible text twice with different whitespace; the earlier turn was compacted away.
    req = chat_request("hello")
    assert restore_request(req, [("hello", "\nhello"), ("hello", "\n\nhello\n")]) == 1
    assert req["messages"][2]["content"] == "\n\nhello\n"
    req2 = chat_request("hello", "hello")
    restore_request(req2, [("hello", "\nhello"), ("hello", "\n\nhello\n")])
    assert [req2["messages"][2]["content"], req2["messages"][4]["content"]] == ["\nhello", "\n\nhello\n"]


def test_responses_input_items():
    req = {"input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "reasoning", "encrypted_content": "e", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": "hello"},
        {"type": "message", "role": "assistant", "content": "hello"},  # fallback string form
        {"type": "message", "role": "user", "content": "next"},
    ]}
    assert restore_request(req, [("hello", RAW)]) == 2
    assert req["input"][2]["content"][0]["text"] == RAW
    assert req["input"][5]["content"] == RAW
    assert req["input"][4]["output"] == "hello"


# ---- routing
def test_route_scoping():
    kw = dict(model="ifm-7b", provider="custom", base_url="http://localhost:8000/v1", api_mode="chat_completions")
    assert not route_enabled({}, **kw)                             # unscoped -> inactive
    assert route_enabled({"all_routes": True}, **kw)
    assert route_enabled({"models": ["ifm-*"]}, **kw)
    assert not route_enabled({"models": ["gpt-*"]}, **kw)
    assert route_enabled({"models": ["IFM-*"], "base_urls": ["http://localhost:*"]}, **kw)
    assert not route_enabled({"models": ["ifm-*"], "providers": ["openrouter"]}, **kw)
    assert not route_enabled({"models": ["ifm-*"]}, **{**kw, "api_mode": "anthropic_messages"})
    assert route_enabled({"models": ["ifm-*"]}, **{**kw, "api_mode": "codex_responses"})
    assert not route_enabled({"models": ["ifm-*"], "enabled": False}, **kw)


# ---- storage
def test_store_persists_and_dedupes(tmp_path):
    db = tmp_path / "r.db"
    s = RecordStore(db)
    assert s.record("s1", RAW)
    assert not s.record("s1", RAW)          # immediate repeat (retry)
    assert not s.record("s1", "hello")      # nothing to restore
    s.record("", "\nmem-only\n")            # no session id: memory only
    s2 = RecordStore(db)                    # new process
    assert s2.records("s1") == [("hello", RAW)]
    assert s2.records("") == []
