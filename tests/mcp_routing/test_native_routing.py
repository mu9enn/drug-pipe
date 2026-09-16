import asyncio,os,pathlib,sys,unittest
from datetime import timedelta
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'slime-wd/slime'))
from drug_agent.tools.mcp_client import MCPClient
class Routing(unittest.IsolatedAsyncioTestCase):
 async def test_stdio_owner_lifecycle_and_sdk_timeout_units(self):
  for annotation in [float,timedelta]:
   with self.subTest(annotation=annotation):
    calls=[];owner=asyncio.current_task()
    class Transport:
     async def __aenter__(self):assert asyncio.current_task() is owner;return 'read','write'
     async def __aexit__(self,*args):calls.append('transport_closed');assert asyncio.current_task() is owner
    def factory(params):
     assert params.command=='bash' and params.args==[str(P/'runtime/molclaw_mcp.sh')];assert params.env['MOLCLAW_SCP_API_KEY']=='TEST_ONLY';assert params.env['MOLCLAW_SCP_SERVER_URL']=='https://example.invalid/mcp';calls.append('stdio');return Transport()
    class Session:
     def __init__(self,read,write,read_timeout_seconds=None):
      assert read_timeout_seconds==(14400.0 if annotation is float else timedelta(seconds=14400));calls.append('session')
     async def __aenter__(self):return self
     async def __aexit__(self,*args):calls.append('session_closed');assert asyncio.current_task() is owner
     async def initialize(self):calls.append('initialize')
     async def call_tool(self,name,arguments):return {'content':[{'type':'text','text':'{"done":true}'}]}
    Session.__init__.__annotations__['read_timeout_seconds']=annotation
    with patch.dict(os.environ,{'DRUG_PROJECT':str(P),'DRUG_AGENT_ALLOW_TOOL_ENV':'1','DRUG_AGENT_TRAINING_OFFLINE':'0'}),patch('mcp.ClientSession',Session),patch('mcp.client.stdio.stdio_client',factory):
     c=MCPClient(server_url='https://example.invalid/mcp',api_key='TEST_ONLY');await c.connect();assert (await c.call_tool('test',{}))['parsed']=={'done':True};await c.disconnect()
    self.assertEqual(calls,['stdio','session','initialize','session_closed','transport_closed'])
if __name__=='__main__':unittest.main()
