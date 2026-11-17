"""Loopback-only Claude Code cache proxy for an Anthropic-compatible upstream.

This proxy changes cache controls only. It does not lift images, convert PDFs,
or make any panel-specific request changes.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
import zlib

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import cache_hints
from cache_usage import UsageObserver, normalize


HOP_HEADERS = frozenset({
    'host', 'content-length', 'connection', 'transfer-encoding', 'keep-alive',
    'te', 'trailer', 'upgrade', 'proxy-authorization', 'proxy-authenticate',
})
MAX_BODY = 100 * 1024 * 1024


def strip_cache_controls(value):
    """Remove both automatic and block-level cache controls in-place."""
    changed = False
    if isinstance(value, dict):
        if 'cache_control' in value:
            del value['cache_control']
            changed = True
        for part in value.values():
            changed = strip_cache_controls(part) or changed
    elif isinstance(value, list):
        for part in value:
            changed = strip_cache_controls(part) or changed
    return changed


def append_usage(path, metric):
    """Append counters only, never a prompt, response, or credential."""
    raw = (json.dumps(metric, separators=(',', ':')) + '\n').encode()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def create_app(upstream, cache_ttl='5m', no_cache=False, usage_log=None):
    upstream = upstream.rstrip('/')
    if not upstream.startswith(('https://', 'http://')):
        raise ValueError('upstream must be an HTTP URL')
    if cache_ttl not in ('5m', '1h'):
        raise ValueError('cache_ttl must be 5m or 1h')
    usage_log = Path(usage_log) if usage_log else None

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(1200, connect=30), trust_env=False) as client:
            app.state.client = client
            yield

    async def forward(request):
        if request.url.path == '/healthz':
            return JSONResponse({'ok': True, 'prompt_cache': not no_cache,
                                 'cache_ttl': None if no_cache else cache_ttl})
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY:
                return JSONResponse({'error': {'type': 'invalid_request_error',
                                               'message': 'Request exceeds 100MB'}}, status_code=413)
            chunks.append(chunk)
        body = b''.join(chunks)
        started = time.monotonic()
        stamp = time.time()
        model = 'unknown'
        decoded = False
        encoding = request.headers.get('content-encoding', '').lower()
        if encoding in ('gzip', 'deflate'):
            try:
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == 'gzip' else zlib.MAX_WBITS)
                body = decoder.decompress(body, MAX_BODY + 1)
                if len(body) > MAX_BODY or decoder.unconsumed_tail:
                    return JSONResponse({'error': {'type': 'invalid_request_error',
                                                   'message': 'Decoded request exceeds 100MB'}}, status_code=413)
                if not decoder.eof:
                    raise ValueError('Incomplete compressed request')
                decoded = True
            except (zlib.error, ValueError):
                return JSONResponse({'error': {'type': 'invalid_request_error',
                                               'message': 'Invalid compressed request'}}, status_code=400)

        is_message = request.method == 'POST' and request.url.path.rstrip('/') in (
            '/v1/messages', '/v1/messages/count_tokens')
        if is_message:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    model = payload.get('model', 'unknown')
                    changed = strip_cache_controls(payload) if no_cache else cache_hints.anthropic(payload, ttl=cache_ttl)
                    if changed:
                        body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
            except (ValueError, TypeError, AttributeError):
                # The upstream remains responsible for validating malformed requests.
                pass

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        if decoded:
            headers.pop('content-encoding', None)
        raw_path = request.scope.get('raw_path', request.url.path.encode()).decode('ascii')
        url = upstream + raw_path
        if request.url.query:
            url += '?' + request.url.query
        client = request.app.state.client
        try:
            response = await client.send(client.build_request(request.method, url, headers=headers, content=body), stream=True)
        except httpx.TimeoutException:
            return JSONResponse({'error': {'type': 'api_error', 'message': 'Upstream timeout'}}, status_code=504)
        except httpx.HTTPError:
            return JSONResponse({'error': {'type': 'api_error', 'message': 'Upstream connection failed'}}, status_code=502)

        async def stream():
            observe = usage_log is not None and request.method == 'POST' and request.url.path.rstrip('/') == '/v1/messages'
            observer = (UsageObserver(response.headers.get('content-type', ''),
                                      response.headers.get('content-encoding', '')) if observe else None)
            completed = False
            try:
                async for chunk in response.aiter_raw():
                    if observer:
                        observer.feed(chunk)
                    yield chunk
                completed = True
            finally:
                await response.aclose()
                if observer:
                    observer.finish()
                    status = ('error' if response.status_code >= 400 else
                              'complete' if completed and observer.stopped else 'incomplete')
                    metric = {'ts': stamp, 'model': model, 'status': status,
                              'elapsed_ms': round((time.monotonic() - started) * 1000),
                              **normalize(observer.usage)}
                    try:
                        await asyncio.shield(asyncio.to_thread(append_usage, usage_log, metric))
                    except OSError:
                        # A metrics file must not break the model response.
                        pass

        return StreamingResponse(stream(), status_code=response.status_code,
                                 headers={k: v for k, v in response.headers.items()
                                          if k.lower() not in HOP_HEADERS})

    return Starlette(routes=[Route('/{path:path}', forward,
                                   methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD'])],
                     lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, help='Anthropic-compatible HTTP API base URL')
    parser.add_argument('--port', type=int, default=18045)
    parser.add_argument('--cache-ttl', choices=('5m', '1h'), default='5m')
    parser.add_argument('--no-cache', action='store_true', help='Remove cache controls before forwarding')
    parser.add_argument('--usage-log', type=Path, help='Optional JSONL path for provider-reported token counters')
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.upstream, cache_ttl=args.cache_ttl, no_cache=args.no_cache,
                           usage_log=args.usage_log), host='127.0.0.1', port=args.port,
                access_log=False)


if __name__ == '__main__':
    main()
