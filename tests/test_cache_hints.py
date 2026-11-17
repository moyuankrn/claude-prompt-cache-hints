import copy
import unittest

import cache_hints as hints


class OpenAICompatibleTests(unittest.TestCase):
    def test_stable_prefix_only_and_no_input_mutation(self):
        messages = [{'role': 'system', 'content': 'Stable rules'}]
        for n in range(8):
            messages += [
                {'role': 'user', 'content': f'Question {n}'},
                {'role': 'assistant', 'content': f'Answer {n}'},
            ]
        messages.append({'role': 'user', 'content': [
            {'type': 'text', 'text': 'Current question'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
        ]})
        payload = {'model': 'claude-example', 'messages': messages}
        before = copy.deepcopy(payload)
        config = {'prompt_cache': {
            'enabled': True, 'static_ttl': '1h', 'tail_ttl': '5m',
            'partition_turns': 3,
        }}
        marked = hints.openai(payload, config, 'Current question')
        self.assertEqual(payload, before)
        self.assertEqual(marked['messages'][-1], before['messages'][-1])
        self.assertEqual(hints.count_marks(marked), 3)
        self.assertEqual(marked['messages'][0]['content'][0]['cache_control']['ttl'], '1h')

    def test_existing_marks_and_disabled_policy_are_untouched(self):
        payload = {'model': 'claude-example', 'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'rules',
                                           'cache_control': {'type': 'ephemeral'}}]},
            {'role': 'user', 'content': 'new'},
        ]}
        config = {'prompt_cache': {'enabled': True}}
        self.assertIs(hints.openai(payload, config, 'new'), payload)
        self.assertIs(hints.openai(payload, {}, 'new'), payload)


class AnthropicTests(unittest.TestCase):
    def test_runtime_timestamp_outside_cache_and_text_preserved(self):
        original = 'Fixed rules.\nCurrent date: 2026-11-17\nLive state.'
        payload = {'model': 'claude-example', 'system': original,
                   'messages': [{'role': 'user', 'content': 'new request'}]}
        self.assertTrue(hints.anthropic(payload))
        self.assertEqual(''.join(x['text'] for x in payload['system']), original)
        self.assertIn('cache_control', payload['system'][0])
        self.assertNotIn('cache_control', payload['system'][1])
        self.assertFalse(hints.anthropic(payload))

    def test_existing_marks_use_only_available_slots(self):
        payload = {'model': 'claude-example',
                   'tools': [{'name': 'Read'}, {'name': 'Write'}],
                   'system': [{'type': 'text', 'text': 'fixed',
                               'cache_control': {'type': 'ephemeral'}},
                              {'type': 'text', 'text': 'fixed too',
                               'cache_control': {'type': 'ephemeral'}}],
                   'messages': [{'role': 'user', 'content': [
                       {'type': 'text', 'text': 'old',
                        'cache_control': {'type': 'ephemeral'}}]},
                       {'role': 'user', 'content': 'current'}]}
        self.assertTrue(hints.anthropic(payload))
        self.assertEqual(hints.count_marks(payload), 4)
        self.assertNotIn('cache_control', payload['messages'][-1])
        self.assertFalse(hints.anthropic(payload))

    def test_longer_ttl_upgrades_existing_marks(self):
        payload = {'model': 'claude-example', 'system': [
            {'type': 'text', 'text': 'fixed',
             'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': 'current'}]}
        self.assertTrue(hints.anthropic(payload, ttl='1h'))
        self.assertEqual(payload['system'][0]['cache_control']['ttl'], '1h')


if __name__ == '__main__':
    unittest.main()
