import unittest
from dsml_bridge import convert_response
class Boundaries(unittest.TestCase):
 def test_truncated_native_call_is_not_executable(self):
  data={'choices':[{'finish_reason':'length','message':{'tool_calls':[{'id':'c1','function':{'name':'test','arguments':'{}'}}]}}]}
  with self.assertRaisesRegex(ValueError,'truncated native'):
   convert_response(data,[{'name':'test','input_schema':{'type':'object'}}],'test')
 def test_text_length_remains_max_tokens(self):
  data={'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}
  msg,_=convert_response(data,[],'test');self.assertEqual(msg['stop_reason'],'max_tokens')
 def test_complete_native_unchanged(self):
  data={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'c1','function':{'name':'test','arguments':'{}'}}]}}]}
  msg,_=convert_response(data,[{'name':'test','input_schema':{'type':'object'}}],'test');self.assertEqual(msg['stop_reason'],'tool_use')
if __name__=='__main__':unittest.main()
