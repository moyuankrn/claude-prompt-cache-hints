"""Observe provider-reported cache usage without persisting response text."""

import codecs
import json
import zlib


class UsageObserver:
    """Incrementally read JSON or SSE usage, including compressed responses."""

    def __init__(self, content_type='', encoding=''):
        self.sse = 'text/event-stream' in content_type
        self.usage = {}
        self.stopped = False
        self.buffer = ''
        self.fields = []
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.compression = (zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == 'gzip' else zlib.MAX_WBITS)
                            if encoding in ('gzip', 'deflate') else None)
        self.enabled = encoding in ('', 'identity', 'gzip', 'deflate')

    def event(self, event):
        if not isinstance(event, dict):
            return
        nested = event.get('message', {}).get('usage') if isinstance(event.get('message'), dict) else None
        if isinstance(nested, dict):
            self.usage.update(nested)
        usage = event.get('usage')
        if isinstance(usage, dict):
            self.usage.update(usage)
        if event.get('type') == 'message_stop' or not self.sse:
            self.stopped = True

    def line(self, line):
        line = line.rstrip('\r').lstrip('\ufeff')
        if line.startswith('data:'):
            field = line[5:]
            self.fields.append(field[1:] if field.startswith(' ') else field)
        elif not line and self.fields:
            self.flush()
        if self.fields:
            try:
                value = json.loads('\n'.join(self.fields))
            except ValueError:
                return
            self.fields = []
            self.event(value)

    def flush(self):
        if not self.fields:
            return
        try:
            self.event(json.loads('\n'.join(self.fields)))
        except ValueError:
            pass
        self.fields = []

    def feed(self, raw):
        if not self.enabled:
            return
        try:
            if self.compression:
                raw = self.compression.decompress(raw)
            self.buffer += self.decoder.decode(raw)
            if self.sse:
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    self.line(line)
            # The observer may transiently see answer text. Bound it and never log it.
            if len(self.buffer) > 2 * 1024 * 1024:
                self.buffer = ''
            if sum(map(len, self.fields)) > 2 * 1024 * 1024:
                self.fields = []
        except (UnicodeError, zlib.error):
            self.enabled = False
            self.buffer = ''
            self.fields = []

    def finish(self):
        if not self.enabled:
            return
        if self.sse:
            if self.buffer:
                self.line(self.buffer)
            self.flush()
        elif self.buffer:
            try:
                self.event(json.loads(self.buffer))
            except ValueError:
                pass
        self.buffer = ''
        self.fields = []


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if 0 <= value <= 9223372036854775807 else None


def normalize(usage):
    """Keep missing usage distinct from provider-reported zero tokens."""
    u = usage if isinstance(usage, dict) else {}
    details = u.get('prompt_tokens_details') or u.get('input_tokens_details') or {}
    cached = _number(details.get('cached_tokens'))
    if cached is None:
        cached = _number(u.get('cache_read_input_tokens'))
    if cached is None:
        cached = _number(u.get('prompt_cache_hit_tokens'))
    write = _number(details.get('cached_creation_tokens'))
    if write is None:
        write = _number(details.get('cache_write_tokens'))
    if write is None:
        write = _number(details.get('cache_creation_tokens'))
    if write is None:
        write = _number(u.get('cache_creation_input_tokens'))
    if write is None:
        parts = [_number(u.get(k)) for k in ('claude_cache_creation_5_m_tokens',
                                              'claude_cache_creation_1_h_tokens')]
        if any(x is not None for x in parts):
            write = sum(x or 0 for x in parts)
    inp = _number(u.get('prompt_tokens'))
    if inp is None:
        inp = _number(u.get('input_tokens'))
        if inp is not None and u.get('usage_semantic') != 'openai':
            inp += (cached or 0) + (write or 0)
    out = _number(u.get('completion_tokens'))
    if out is None:
        out = _number(u.get('output_tokens'))
    total = _number(u.get('total_tokens'))
    if total is None and inp is not None and out is not None:
        total = inp + out
    return {'input_tokens': inp, 'output_tokens': out, 'cached_tokens': cached,
            'cache_write_tokens': write, 'total_tokens': total}
