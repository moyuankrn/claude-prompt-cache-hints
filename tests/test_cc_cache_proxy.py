import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from starlette.testclient import TestClient

from cache_usage import UsageObserver, normalize
from cc_cache_proxy import create_app


class CacheUsageTests(unittest.TestCase):
    def test_split_gzip_sse_tracks_provider_usage_only(self):
        body = (b'data: {"type":"message_start","message":{"usage":{"input_tokens":20,'
                b'"cache_read_input_tokens":80,"cache_creation_input_tokens":10}}}\n\n'
                b'data: {"type":"message_delta","usage":{"output_tokens":4}}\n\n'
                b'data: {"type":"message_stop"}\n\n')
        observer = UsageObserver('text/event-stream', 'gzip')
        zipped = gzip.compress(body)
        for i in range(0, len(zipped), 7):
            observer.feed(zipped[i:i + 7])
        observer.finish()
        self.assertTrue(observer.stopped)
        self.assertEqual(normalize(observer.usage), {
            'input_tokens': 110, 'output_tokens': 4, 'cached_tokens': 80,
            'cache_write_tokens': 10, 'total_tokens': 114,
        })

    def test_missing_and_zero_usage_are_distinct(self):
        self.assertIsNone(normalize({})['cached_tokens'])
        self.assertEqual(normalize({'prompt_tokens_details': {'cached_tokens': 0},
                                    'cache_read_input_tokens': 7})['cached_tokens'], 0)


class ProxyTests(unittest.TestCase):
    def test_preserves_auth_query_status_stream_and_records_counters(self):
        requests = []
        body = (b'data: {"type":"message_start","message":{"usage":{"input_tokens":20,'
                b'"cache_read_input_tokens":80,"cache_creation_input_tokens":10}}}\n\n'
                b'data: {"type":"message_stop"}\n\n')

        class FixtureStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for i in range(0, len(body), 9):
                    yield body[i:i + 9]

        async def upstream(request):
            requests.append(request)
            return httpx.Response(200, stream=FixtureStream(),
                                  headers={'content-type': 'text/event-stream'})

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'usage.jsonl'
            app = create_app('https://gateway.invalid', usage_log=log)
            payload = {'model': 'claude-example', 'tools': [{'name': 'Read'}],
                       'system': [{'type': 'text', 'text': 'fixed instructions'},
                                  {'type': 'text', 'text': 'Current date: 2026-11-17'}],
                       'messages': [{'role': 'user', 'content': 'Current request'}]}
            original = copy.deepcopy(payload)
            with TestClient(app) as client:
                original_client = app.state.client
                app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
                response = client.post('/v1/messages?beta=true', json=payload,
                                       headers={'authorization': 'Bearer synthetic'})
                client.portal.call(app.state.client.aclose)
                app.state.client = original_client
            self.assertEqual(payload, original)
            self.assertEqual(response.content, body)
            self.assertEqual(requests[0].headers['authorization'], 'Bearer synthetic')
            self.assertEqual(str(requests[0].url), 'https://gateway.invalid/v1/messages?beta=true')
            sent = json.loads(requests[0].content)
            self.assertIn('cache_control', sent['tools'][0])
            self.assertIn('cache_control', sent['system'][0])
            self.assertNotIn('cache_control', sent['system'][1])
            self.assertNotIn('cache_control', sent['messages'][0])
            metric = json.loads(log.read_text())
            self.assertEqual(metric['cached_tokens'], 80)
            self.assertEqual(metric['input_tokens'], 110)
            self.assertNotIn('Bearer synthetic', log.read_text())
            self.assertNotIn('Current request', log.read_text())

    def test_gzip_request_and_no_cache_switch(self):
        requests = []

        class FixtureStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"error":"synthetic"}'

        async def upstream(request):
            requests.append(request)
            return httpx.Response(429, stream=FixtureStream(), headers={'retry-after': '2'})

        app = create_app('https://gateway.invalid', no_cache=True)
        payload = {'model': 'claude-example', 'cache_control': {'type': 'ephemeral'},
                   'system': [{'type': 'text', 'text': 'fixed',
                               'cache_control': {'type': 'ephemeral'}}],
                   'messages': [{'role': 'user', 'content': 'new'}]}
        with TestClient(app) as client:
            original_client = app.state.client
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            response = client.post('/v1/messages/count_tokens',
                                   content=gzip.compress(json.dumps(payload).encode()),
                                   headers={'content-encoding': 'gzip', 'x-api-key': 'synthetic'})
            client.portal.call(app.state.client.aclose)
            app.state.client = original_client
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['retry-after'], '2')
        self.assertNotIn('content-encoding', requests[0].headers)
        self.assertEqual(requests[0].headers['x-api-key'], 'synthetic')
        sent = json.loads(requests[0].content)
        self.assertNotIn('cache_control', sent)
        self.assertNotIn('cache_control', sent['system'][0])


if __name__ == '__main__':
    unittest.main()
