# preserve-whitespace

A Hermes plugin that replays assistant text to the model with the leading and trailing
whitespace the model actually emitted. It only acts on the routes you list, and it works with
both Chat Completions and the Responses API.

## The problem

Hermes trims assistant text before it is sent back to the model on later turns. Say the model
emits:

```
...</ifm|think>\n\nhello\n<ifm|tool_calls>...
```

The server parses the reasoning and the tool call into their own fields and returns
`content = "\n\nhello\n"`. On the next turn the model sees `"hello"`, so what it is shown no
longer matches what it produced. For models trained to expect exact replay, this hurts quality
and breaks prefix-cache reuse on the server.

This is where the text gets trimmed (checked against Hermes `main` at `71736c9f32`, 2026-11-03):

| Path | Where | What |
|---|---|---|
| Chat Completions, storage | `agent/chat_completion_helpers.py` `_assistant_content_for_storage` | `strip_think_blocks(content).strip()` |
| Every API mode, send time | `agent/turn_request_assembly.py` (closing normalization) | `content.strip()` on every string content |
| Iteration summary, send time | `agent/chat_completion_helpers.py` (summary request builder) | same `.strip()` |
| Responses API, capture | `agent/codex_responses_adapter.py` `_extract_responses_message_text` | `"".join(parts).strip()`, then stored in `codex_message_items` |

Removing these calls inside Hermes would change behaviour for every provider:

- Anthropic-style APIs reject a request whose final assistant message ends in whitespace.
- The send-time trim keeps prompt-cache prefixes byte-stable.

So this plugin restores the whitespace only for the routes you choose, and leaves Hermes
unchanged.

## How it works

The plugin uses Hermes' supported middleware API
(`website/docs/developer-guide/middleware.md` in the Hermes repo) and never patches
Hermes internals.

1. **Capture, using `llm_execution` middleware.** This wraps the provider call. Once the response
   is back (streams are fully assembled by then), the plugin reads the model's raw text:
   - Chat Completions: `choices[0].message.content`.
   - Responses: each assistant `message` item's text, joined exactly as Hermes joins it.

   If trimming would change that text, the plugin records the pair
   `(key = raw.strip(), raw)` for this session.
2. **Restore, using `llm_request` middleware.** This runs after every Hermes pass, just before
   the request is sent. For each assistant text in `messages` (Chat Completions) or `input`
   (Responses) that is exactly equal to a recorded `key`, the plugin writes `raw` back.

### Why only whitespace can change

`str.strip()` removes leading and trailing whitespace and nothing else. If
`raw.strip() == outgoing_text`, the two strings are identical apart from outer whitespace, so
putting `raw` back cannot add, remove or alter any other character. The plugin checks this
condition again on every write.

If Hermes changed the text in any other way, the key doesn't match and the plugin does nothing:

| Model emitted | Hermes sends | Plugin's key | Result |
|---|---|---|---|
| `"\n\nhello\n"` | `"hello"` | `"hello"` | restored to `"\n\nhello\n"` |
| `"<think>x</think>\n\nhello"` | `"hello"` (think block removed) | `"<think>x</think>\n\nhello"` | unchanged |
| `"\nkey=sk-abc\n"` | `"key=***"` (secret redacted) | `"key=sk-abc"` | unchanged |

It also never touches:

- user, system or tool rows;
- tool-call arguments, ids or reasoning fields;
- the final row of a request when that row is an assistant message (prefill);
- any request on a route you didn't list.

## Install

```bash
hermes plugins install ifm-ai/k2-horizon-hermes-patch --no-enable
hermes plugins enable preserve-whitespace
```

Or by hand: copy this repository to `~/.hermes/plugins/preserve-whitespace/` and run
`hermes plugins enable preserve-whitespace`.

Then add the routes to `~/.hermes/config.yaml`. The plugin does nothing until at least one rule
is set:

```yaml
plugins:
  entries:
    preserve-whitespace:
      settings:
        models: ["ifm-*"]                       # glob, case-insensitive
        # providers: ["custom"]                 # optional, glob
        # base_urls: ["http://localhost:8000*"] # optional, glob
        # api_modes: [chat_completions, codex_responses]   # default
        # persist: false         # true = also keep records on disk across restarts (default false)
        # retain_days: 30        # prune older records at startup (default 30)
        # all_routes: false      # true = apply to every route (not recommended)
        # enabled: true
```

Every non-empty list must match the live route: its model, provider, base URL and API mode.
Fallback providers and model switches are checked per request, so a fallback to an unlisted
provider is left alone automatically. Settings are re-read every few seconds, so a config change
takes effect without restarting.

Restart Hermes (or the gateway) after enabling the plugin.

## Checking it works

Run any session on a listed model and look at what is sent on the second turn. Either:

- set `HERMES_DUMP_REQUESTS=1` and inspect the dumped request, or
- check `middleware_trace` in a `pre_api_request` / `post_api_request` observer.

Each request where the plugin restored text carries a trace entry like this:

```json
{"source": "preserve-whitespace", "reason": "restored outer whitespace on 1 assistant text(s)"}
```

By default records are kept in memory only and are gone when Hermes exits. With `persist: true`
they are also written to `~/.hermes/plugin-data/preserve-whitespace/records.db` (SQLite, one row per
recorded text) and pruned after `retain_days`. That file holds the model's raw replies, unredacted,
so only turn it on where that is acceptable.

## Limitations

- **The saved transcript is still trimmed.** The plugin changes what is sent to the model, not
  what Hermes stores. Anything that reads `state.db` or the gateway transcript directly still
  sees the trimmed text.
- **Only outer whitespace is restored.** Changes Hermes makes for other reasons still reach the
  model, for example:
  - reformatted tool-call arguments;
  - dropped reasoning-only turns;
  - merged consecutive turns.
- **The same text with different whitespace.** If the model says `"hello"` once as `"\nhello"`
  and once as `"\n\nhello\n"`, both have the key `"hello"`. Each occurrence is then matched to
  the most recent records in order, since compaction removes the oldest turns first. An
  occurrence that can't be matched is left unchanged.
- **Responses items with several text parts.** Hermes merges them into one part, so the plugin
  restores the joined text with its outer whitespace. All the characters come back, but the
  original part boundaries don't.
- **Whitespace-only replies** (for example a tool-call turn whose content is just `"\n"`) are not
  restored. Hermes stores those as empty, and its empty-turn handling owns them.
- **Calls that bypass middleware.** The iteration-summary call builds its request without request
  middleware, so it still sees trimmed text. It's a one-off call at the iteration limit.
- **Resumed sessions** (after a Hermes restart) are only restored with `persist: true`. Without it,
  turns from before the restart are sent trimmed. Sessions without an id (some auxiliary or one-shot
  calls) are always kept in memory only.
- **Cost:** Hermes deep-copies the request for each request middleware, which adds a little work
  per call on listed routes. Prompt caching is unaffected, because the restored bytes are the same
  on every turn.

## Tests

```bash
cd ~/.hermes/plugins/preserve-whitespace/tests
python -m pytest                                     # pure logic, no Hermes needed
HERMES_REPO=/path/to/hermes-agent python -m pytest   # also runs through Hermes' real middleware.py
```

The integration test runs the plugin through Hermes' real `hermes_cli/middleware.py` (the module
the turn loop calls). It builds the stored Responses text with Hermes' own
`_extract_responses_message_text`, taken straight from the adapter source. If a Hermes update
moves that function, the test fails loudly instead of passing quietly.

## Uninstall

```bash
hermes plugins disable preserve-whitespace
rm -rf ~/.hermes/plugins/preserve-whitespace ~/.hermes/plugin-data/preserve-whitespace
```

## Files

| File | Purpose |
|---|---|
| `plugin.yaml` | Manifest |
| `__init__.py` | `register(ctx)`: settings, storage, the two middleware callbacks |
| `core.py` | Pure logic: capture, restore, route matching, record store (no Hermes imports) |
| `tests/` | Unit tests and the Hermes integration test |
