"""Integration test against a Hermes checkout: ``HERMES_REPO=/path/to/hermes-agent pytest tests/``.

Runs the plugin through Hermes' real ``hermes_cli/middleware.py`` (the code the turn loop calls),
and uses Hermes' real Responses capture function (``_extract_responses_message_text``, pulled
from the adapter source) to produce the stored text. The plugin registry is a small stand-in that
mirrors ``PluginManager.invoke_middleware`` (per-callback copies, rewrites compose), so the full
Hermes dependency set is not needed. Skipped when HERMES_REPO is not set.
"""

import ast
import importlib.util
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

REPO = os.environ.get("HERMES_REPO")
pytestmark = pytest.mark.skipif(not REPO, reason="set HERMES_REPO to a hermes-agent checkout")
PLUGIN_DIR = Path(__file__).resolve().parents[1]
RAW = "\n\nhello\n"
ROUTE = dict(model="ifm-7b", provider="custom", base_url="http://localhost:8000/v1")


class _Manager:
    """Stand-in for the plugin manager's middleware registry/dispatch."""

    def __init__(self):
        self._middleware: dict[str, list] = {}

    def invoke_middleware(self, kind, *, _payload_key=None, **kwargs):
        from hermes_cli.middleware import _safe_copy
        results = []
        for cb in self._middleware.get(kind, []):
            call_kwargs = kwargs
            if _payload_key:
                ok = "original_" + _payload_key
                call_kwargs = {**kwargs, _payload_key: _safe_copy(kwargs[_payload_key]), ok: _safe_copy(kwargs[ok])}
            try:
                ret = cb(**call_kwargs)
            except Exception:
                continue  # fail-open, like Hermes
            if ret is not None:
                results.append(ret)
                if _payload_key and isinstance(ret, dict) and isinstance(ret.get(_payload_key), dict):
                    kwargs = {**kwargs, _payload_key: ret[_payload_key]}
        return results

    def _report_hook_failure(self, *a, **k):
        pass


@pytest.fixture()
def hermes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(REPO)
    for name in [m for m in sys.modules if m == "hermes_cli" or m.startswith("hermes_cli.")]:
        monkeypatch.delitem(sys.modules, name)
    import hermes_cli.middleware as mw  # the real module

    manager = _Manager()
    fake = types.ModuleType("hermes_cli.plugins")
    fake.invoke_middleware = lambda kind, **kw: manager.invoke_middleware(kind, **kw)
    fake.has_middleware = lambda kind: bool(manager._middleware.get(kind))
    fake._delivery_manager = lambda: manager
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake)

    settings = {"models": ["ifm-*"], "persist": False}

    class Ctx:
        def register_middleware(self, kind, cb):
            manager._middleware.setdefault(kind, []).append(cb)

        def get_config(self, key, default=None):
            return settings.get(key, default)

    spec = importlib.util.spec_from_file_location(
        "preserve_whitespace_under_test", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    plugin.register(Ctx())
    return mw


def _hermes_responses_capture():
    """Hermes' own ``_extract_responses_message_text`` (exact source), with its helpers."""
    src = Path(REPO, "agent", "codex_responses_adapter.py").read_text()
    wanted = {"_extract_responses_message_text", "_nonempty_str", "_field", "_as_list"}
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted, "adapter layout changed; update this test"
    from typing import Optional, TypeGuard
    ns: dict[str, Any] = {"Any": Any, "TypeGuard": TypeGuard, "Optional": Optional,
                          "_OUTPUT_TEXT_TYPES": {"output_text", "text"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "adapter_extract", "exec"), ns)
    return ns["_extract_responses_message_text"]


def _execute(mw, request, response, api_mode, session="s1", **route):
    route = {**ROUTE, **route}
    return mw.run_llm_execution_middleware(request, lambda req: response, original_request=request,
                                           session_id=session, api_mode=api_mode, **route)


def _send(mw, request, api_mode, session="s1", **route):
    route = {**ROUTE, **route}
    return mw.apply_llm_request_middleware(request, session_id=session, api_mode=api_mode, **route).payload


def test_chat_completions_round_trip(hermes):
    mw = hermes
    tc = [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}]
    # Turn 1: the server split "<ifm|think>..</ifm|think>" into reasoning; content keeps the whitespace.
    resp = NS(choices=[NS(message=NS(content=RAW, tool_calls=tc, reasoning_content="r"))])
    assert _execute(mw, {"messages": []}, resp, "chat_completions") is resp
    # Hermes stores content.strip() and strips again at send time.
    stored = RAW.strip()
    req = {"model": "ifm-7b", "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": stored, "reasoning_content": "r", "tool_calls": tc},
        {"role": "tool", "tool_call_id": "c1", "content": "file.txt"}]}
    out = _send(mw, deepcopy(req), "chat_completions")
    assert out["messages"][2]["content"] == RAW
    assert {k: v for k, v in out["messages"][2].items() if k != "content"} == \
        {k: v for k, v in req["messages"][2].items() if k != "content"}
    assert out["messages"][:2] == req["messages"][:2] and out["messages"][3] == req["messages"][3]


def test_other_routes_and_sessions_untouched(hermes):
    mw = hermes
    _execute(mw, {}, NS(choices=[NS(message=NS(content=RAW))]), "chat_completions")
    req = {"messages": [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "x"}]}
    assert _send(mw, deepcopy(req), "chat_completions", model="gpt-5") == req          # route not listed
    assert _send(mw, deepcopy(req), "chat_completions", session="s2") == req           # other session
    assert _send(mw, deepcopy(req), "anthropic_messages") == req                       # unsupported mode
    # and a model on an unlisted route is never even recorded
    _execute(mw, {}, NS(choices=[NS(message=NS(content="\nonly-gpt\n"))]), "chat_completions", model="gpt-5")
    req2 = {"messages": [{"role": "assistant", "content": "only-gpt"}, {"role": "user", "content": "x"}]}
    assert _send(mw, deepcopy(req2), "chat_completions") == req2


def test_responses_round_trip(hermes):
    mw = hermes
    capture = _hermes_responses_capture()
    item = NS(type="message", role="assistant", status="completed", id="msg_1",
              content=[NS(type="output_text", text=RAW)])
    resp = NS(output=[NS(type="reasoning", encrypted_content="e", summary=[]), item,
                      NS(type="function_call", call_id="c1", name="terminal", arguments="{}")])
    _execute(mw, {"input": []}, resp, "codex_responses")
    stored = capture(item)          # what Hermes keeps in codex_message_items
    assert stored == "hello"
    req = {"model": "ifm-7b", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
        {"type": "reasoning", "encrypted_content": "e", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": stored}]},
        {"type": "function_call", "call_id": "c1", "name": "terminal", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"}]}
    out = _send(mw, deepcopy(req), "codex_responses")
    assert out["input"][2]["content"][0]["text"] == RAW
    assert [out["input"][i] for i in (0, 1, 3, 4)] == [req["input"][i] for i in (0, 1, 3, 4)]


def test_think_block_in_content_is_not_resurrected(hermes):
    mw = hermes
    # Server did NOT parse reasoning out: Hermes stores strip_think(raw).strip() == "hello",
    # but the plugin's key is raw.strip() == "<think>r</think>\n\nhello", so nothing matches.
    _execute(mw, {}, NS(choices=[NS(message=NS(content="<think>r</think>\n\nhello\n"))]), "chat_completions")
    req = {"messages": [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "x"}]}
    assert _send(mw, deepcopy(req), "chat_completions") == req


def test_provider_errors_propagate(hermes):
    mw = hermes

    def boom(_req):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        mw.run_llm_execution_middleware({}, boom, original_request={}, session_id="s1",
                                        api_mode="chat_completions", **ROUTE)
