# Claude Prompt Cache Hints

A small, dependency-free Python module for placing explicit Claude prompt-cache breakpoints in growing conversations, plus an optional loopback proxy for Claude Code. Neither component caches responses or stores prompt content.

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

## Use with Claude Code

The optional `cc_cache_proxy.py` forwards Claude Code's Anthropic-format requests to your Anthropic-compatible API. It keeps Claude Code's existing breakpoints, fills spare slots with stable tool and system prefixes, and never adds a breakpoint to the current user turn. It can observe provider-reported cache tokens from JSON, SSE, gzip, or deflate responses without changing response bytes.

```bash
python3 -m pip install -r requirements-proxy.txt
python3 cc_cache_proxy.py --upstream https://your-anthropic-compatible-api.example \
  --port 18045 --cache-ttl 5m --usage-log ./cache-usage.jsonl
```

In a separate shell, point a Claude Code session at the local proxy:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18045 claude
```

Keep your existing Claude Code authentication configuration; the proxy forwards request authentication headers to the upstream. Set `--upstream` to the **real** API endpoint, not the local proxy URL. The proxy listens on `127.0.0.1` only. It forwards `/v1/messages` and `/v1/messages/count_tokens` after applying the cache policy; other paths pass through. `--no-cache` removes both automatic and block-level cache controls when comparing behavior.

`--usage-log` is optional. Its JSONL file contains model names, request status, timing, and provider-reported token counters only; it contains no prompts, responses, or credentials. Missing usage stays `null`, rather than being reported as zero. The file is created with owner-only permissions. A provider's reported cache tokens are observations, not proof of actual cost savings; compare with its billing records when measuring impact.

This public proxy extracts the cache path from a private panel integration. It does not include that integration's image or PDF conversions. Run the cache proxy only when your upstream already handles the content types Claude Code sends.

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
- The core module modifies request payloads; the optional proxy is a separate component with HTTP dependencies.
- Test your gateway's handling of `cache_control` before enabling the OpenAI-compatible path in production.

The policy was informed by [Anthropic's prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and the stable-prefix partitioning approach described in [Pawwake](https://github.com/garan0613/pawwake). This implementation is original to the Xiaoke panel project.

## Test

Python 3.8+; install `requirements-proxy.txt` for the proxy tests:

```bash
python3 -m unittest discover -s tests -v
```

MIT licensed. See [LICENSE](LICENSE).
