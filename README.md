# Claude Prompt Cache Hints

A small, dependency-free Python module for placing explicit Claude prompt-cache breakpoints in growing conversations. It adds request hints only: it does not send API requests, store prompts, or cache responses.

The policy marks stable instructions, tools, and older conversation turns while leaving the current user turn and detected runtime timestamps outside the added cache boundary. Existing client breakpoints are preserved, and the module uses no more than four explicit breakpoints.

## Use with the Anthropic Messages API

```python
import cache_hints

payload = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.\nCurrent date: 2026-11-17",
    "messages": [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "New question"},
    ],
}

changed = cache_hints.anthropic(payload, ttl="5m")
print(changed, cache_hints.layout(payload))
# Send payload with your normal Anthropic client or HTTP request.
```

`anthropic()` **mutates** the supplied payload and returns whether it changed. Pass `ttl="1h"` to request one-hour cache entries; this also upgrades existing five-minute entries so the requested TTL is consistent. It recognizes labelled runtime dates and times such as `Current date: ...` or `当前时间：...`; it cannot identify every kind of changing content. Keep other volatile data in a separate, uncached block yourself.

## Use with an OpenAI-compatible Claude gateway

```python
import cache_hints

config = {"prompt_cache": {
    "enabled": True,
    "static_ttl": "1h",
    "tail_ttl": "5m",
    "partition_turns": 10,
}}
payload = {
    "model": "claude-sonnet-4-6",
    "messages": [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Current question"},
    ],
}

marked = cache_hints.openai(payload, config, "Current question")
# Send marked with your normal gateway client.
```

`openai()` returns a **copy** when it adds hints; it leaves the original untouched. It only applies to model names containing `claude`, and only when `prompt_cache.enabled` is true. It does not replace any pre-existing breakpoints. This path requires a gateway that accepts `cache_control` in OpenAI-style message content; it is not a standard feature of every OpenAI-compatible API. Pass the current user's text exactly as it appears in the payload so the new turn is excluded from caching.

## Limits

- Breakpoints are hints to the provider, not a guarantee of cache hits. Prefix length, model, TTL, request cadence, and provider behavior all matter.
- The module does not calculate savings or token usage. Check the provider's returned cache usage fields.
- This is a request-payload utility, not a Claude Code hook or proxy.
- Test your gateway's handling of `cache_control` before enabling the OpenAI-compatible path in production.

The policy was informed by [Anthropic's prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and the stable-prefix partitioning approach described in [Pawwake](https://github.com/garan0613/pawwake). This implementation is original to the Xiaoke panel project.

## Test

Python 3.8+; no packages to install:

```bash
python3 -m unittest discover -s tests -v
```

MIT licensed. See [LICENSE](LICENSE).
