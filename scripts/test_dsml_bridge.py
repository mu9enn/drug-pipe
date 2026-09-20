import unittest
from dsml_bridge import rescue, convert_response, convert_request, events


class BridgeTests(unittest.TestCase):
    def test_typed_parameters_and_multiline(self):
        text = '<｜DSML｜tool_calls><｜DSML｜invoke name="Read"><｜DSML｜parameter name="path" string="true">/tmp/a\nb</｜DSML｜parameter><｜DSML｜parameter name="offset" string="false">2</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
        remaining, calls = rescue(text, {'Read': {'type': 'object', 'properties': {'offset': {'type': 'integer'}}}})
        self.assertEqual(remaining, '')
        self.assertEqual(calls[0]['input'], {'path': '/tmp/a\nb', 'offset': 2})

    def test_invalid_truncated_quoted_unknown(self):
        valid = '<|DSML|invoke name="Read"></|DSML|invoke>'
        for text, tools in [(valid[:-1], {'Read': {}}), (valid, {}), ('```'+valid+'```', {'Read': {}})]:
            with self.assertRaises(ValueError): rescue(text, tools)

    def test_native_and_sse(self):
        data = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [{'id': 'c1', 'function': {'name': 'Read', 'arguments': '{"path":"x"}'}}], 'reasoning_content': 'read it'}}]}
        message, rescued = convert_response(data, [{'name': 'Read', 'input_schema': {}}], 'model')
        self.assertEqual(rescued, 0)
        self.assertEqual(message['stop_reason'], 'tool_use')
        stream = list(events(message))
        self.assertEqual(stream[-1]['type'], 'message_stop')
        body = {'messages': [{'role': 'assistant', 'content': message['content']}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c1', 'content': 'result'}]}]}
        request = convert_request(body, 'model')
        self.assertEqual(request['messages'][0]['reasoning_content'], 'read it')
        self.assertEqual(request['messages'][1], {'role': 'tool', 'tool_call_id': 'c1', 'content': 'result'})

    def test_no_conversion_without_marker(self):
        self.assertEqual(rescue('normal text', {}), ('normal text', []))


if __name__ == '__main__': unittest.main()
