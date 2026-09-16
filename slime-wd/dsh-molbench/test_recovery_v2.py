import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import run_dsh_molbench as r
from recovery_policy import observation,error_class

def call(cid,text,args=None,err=False):
 return [{'type':'tool/call','data':{'callId':cid,'name':'mcp__test','arguments':args or {'x':1}}},
         {'type':'tool/result','data':{'message':{'source':{'callId':cid},'content':[{'isError':err,'content':[{'text':text}]}]}}}]
class RecoveryV2(unittest.TestCase):
 def test_recovered_error_does_not_replay(self):
  e=call('a','Error: fetch failed',err=True)+call('b','{"status":"success","message":"recovered after fetch failed"}')
  self.assertEqual(r.unresolved_infrastructure(e,{}),[])
  self.assertEqual(r.unresolved_infrastructure(call('a','fetch failed appeared in diagnostic history'),{}),[])
  self.assertEqual(r.unresolved_infrastructure([{'type':'adapter/log','data':{'error':'fetch failed'}}]+call('c','{"status":"success"}'),{}),[])
 def test_different_args_do_not_clear(self):
  e=call('a','Error: fetch failed',err=True)+call('b','{"status":"success"}',{'x':2})
  self.assertEqual(len(r.unresolved_infrastructure(e,{})),1)
 def test_deterministic_errors(self):
  for text in ['400 Client Error: Bad Request for url: https://example.com','HTTP 401','HTTP 422','Pre-condition Violation','invalid SMILES','No such file or directory','CUDA out of memory','HTTP 500: FileNotFoundError']:
   with self.subTest(text=text):
    self.assertEqual(r.unresolved_infrastructure(call('a','Error: '+json.dumps({'status':'error','msg':text}),err=True),{}),[])
 def test_known_transient_only(self):
  for text in ['HTTP 429','HTTP 502','HTTP 503','ReadTimeout: timed out','fetch failed','UND_ERR_SOCKET other side closed']:
   self.assertEqual(len(r.unresolved_infrastructure(call('a',text,err=True),{})),1)
  self.assertEqual(r.unresolved_infrastructure(call('a','HTTP 500: unknown internal error',err=True),{}),[])
 def test_latest_definitive_response_supersedes_transport(self):
  e=call('a','fetch failed',err=True)+call('b','Error: invalid parameter',err=True)
  self.assertEqual(r.unresolved_infrastructure(e,{}),[])
 def test_budget_precedes_stale_infra_and_survives_resume(self):
  d={'status':'failed','error':'TimeoutError: task exceeded 14400 seconds','attempt':1,'unresolved_infra':[{'tool':None}],'failure_class':'retryable_infra'}
  self.assertEqual(r.failure_class(d),'trajectory_budget_exhausted');self.assertFalse(r.retry_eligible(d));self.assertEqual(r.unresolved_infrastructure(call('a','fetch failed',err=True),d),[])
 def test_actual_budget_path_cancels_and_stops(self):
  class API:
   def __init__(self):self.calls=[]
   def rpc(self,name,args,**kw):
    self.calls.append(name)
    return {'sessionId':'test'} if name=='session.create' else {'selected':{}}
   def history(self,s):return []
  api=API()
  with tempfile.TemporaryDirectory() as d,patch.object(r.time,'monotonic',side_effect=[0,2]),patch.object(r,'task_prompt',return_value='test'):
   x=r.run_sample(api,Path(d),r.Sample('test','ms1',1,'',''),Path(d),1,'provider','model','preset','',recover_infra=True)
   self.assertEqual(x['failure_class'],'trajectory_budget_exhausted');self.assertEqual(x['budget_exhausted']['retry'],False)
   self.assertEqual(x['unresolved_infra'],[]);self.assertFalse(r.retry_eligible(x));self.assertIn('session.cancel',api.calls)
 def test_bounded_retry_is_not_best_of(self):
  rows=iter([{'attempt':i,'failure_class':'retryable_infra'} for i in [1,2,3]])
  sleeps=[];self.assertEqual(r.recover_task(lambda:next(rows),lambda x:None,sleeps.append)['attempt'],3);self.assertEqual(sleeps,[60,180])
if __name__=='__main__':unittest.main()
