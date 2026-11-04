"""preserve-whitespace: replay assistant text with the outer whitespace the model emitted.

Hermes trims assistant text when it stores a turn (Chat Completions) or captures a Responses
message item, and trims every string ``content`` again when it builds a request. This plugin:

* ``llm_execution`` middleware: after the provider call returns (streams are fully assembled by
  then), records each raw assistant text whose trim changes it.
* ``llm_request`` middleware: just before the request is sent, after every Hermes pass, puts the
  recorded raw text back wherever the outgoing assistant text equals ``raw.strip()``.

Only routes listed in ``plugins.entries.preserve-whitespace.settings`` are touched. See README.md.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from .core import RecordStore, extract_raw_texts, restore_request, route_enabled

logger = logging.getLogger(__name__)

PLUGIN_NAME = "preserve-whitespace"
_SETTINGS_KEYS = ("enabled", "models", "providers", "base_urls", "api_modes", "all_routes", "persist",
                  "retain_days")
_SETTINGS_TTL_S = 5.0


class _Plugin:
    def __init__(self, ctx: Any):
        self._ctx = ctx
        self._settings: dict = {}
        self._settings_at = 0.0
        self._store: Optional[RecordStore] = None
        self._warned_unscoped = False

    # -- configuration ----------------------------------------------------
    def settings(self) -> dict:
        now = time.monotonic()
        if now - self._settings_at > _SETTINGS_TTL_S:
            fresh = {}
            for key in _SETTINGS_KEYS:
                try:
                    value = self._ctx.get_config(key, None)
                except Exception:  # unreadable config: keep the last good settings
                    return self._settings
                if value is not None:
                    fresh[key] = value
            self._settings, self._settings_at = fresh, now
            if not any(fresh.get(k) for k in ("models", "providers", "base_urls", "all_routes")) \
                    and not self._warned_unscoped:
                self._warned_unscoped = True
                logger.warning("%s: no models/providers/base_urls configured; plugin is inactive "
                               "(set plugins.entries.%s.settings.models, or all_routes: true)",
                               PLUGIN_NAME, PLUGIN_NAME)
        return self._settings

    def store(self) -> RecordStore:
        if self._store is None:
            cfg = self.settings()
            db_path: Optional[Path] = None
            if cfg.get("persist", False):
                try:
                    from plugins.plugin_storage import plugin_data_dir
                    db_path = plugin_data_dir(PLUGIN_NAME) / "records.db"
                except Exception as exc:
                    logger.warning("%s: persistent storage unavailable (%s); keeping records in memory",
                                   PLUGIN_NAME, type(exc).__name__)
            try:
                self._store = RecordStore(db_path, retain_days=float(cfg.get("retain_days", 30)))
            except Exception as exc:
                logger.warning("%s: could not open %s (%s); keeping records in memory",
                               PLUGIN_NAME, db_path, type(exc).__name__)
                self._store = RecordStore(None)
        return self._store

    def _route_on(self, kwargs: dict) -> bool:
        return route_enabled(self.settings(), model=kwargs.get("model"), provider=kwargs.get("provider"),
                             base_url=kwargs.get("base_url"), api_mode=kwargs.get("api_mode"))

    # -- middleware -------------------------------------------------------
    def on_llm_execution(self, **kwargs: Any) -> Any:
        # Provider errors must propagate untouched, so only our own bookkeeping is guarded.
        response = kwargs["next_call"](kwargs["request"])
        try:
            if self._route_on(kwargs):
                session_id = str(kwargs.get("session_id") or "")
                store = self.store()
                for raw in extract_raw_texts(response):
                    store.record(session_id, raw)
        except Exception:
            logger.debug("%s: capture failed", PLUGIN_NAME, exc_info=True)
        return response

    def on_llm_request(self, **kwargs: Any) -> Optional[dict]:
        request = kwargs.get("request")
        if not isinstance(request, dict) or not self._route_on(kwargs):
            return None
        records = self.store().records(str(kwargs.get("session_id") or ""))
        if not records:
            return None
        changed = restore_request(request, records)  # ``request`` is already Hermes' private copy
        if not changed:
            return None
        return {"request": request, "source": PLUGIN_NAME,
                "reason": f"restored outer whitespace on {changed} assistant text(s)"}


def register(ctx: Any) -> None:
    plugin = _Plugin(ctx)
    ctx.register_middleware("llm_execution", plugin.on_llm_execution)
    ctx.register_middleware("llm_request", plugin.on_llm_request)
