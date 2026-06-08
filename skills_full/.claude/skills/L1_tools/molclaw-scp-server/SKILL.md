---
name: molclaw-scp-server
description: All tools utilized within molclaw skills connect via the SCP protocol. This skill serves as a unified guide for using the SCP Server. This skill must be loaded to create the SCP server before invoking any tools. 
license: MIT license
metadata:
    skill-author: PJLab
---

# SCP Usage

Note: 
- Local files are not directly accessible by the server. Please upload them to the server using `molclaw-file-transfer` before execution. 
- For PDB file inputs, it is recommended to preprocess them using `molclaw-pdbfixer` before execution.
- Please refer to skill `molclaw-scp-server` to complete tool invocation.

### 1. SCP Server Definition

If MCP environment. is not installed, please run `pip install mcp`.

The SCP server is defined as below:

```python
import json
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

class DrugSDAClient:    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        
    async def connect(self):
        print(f"server url: {self.server_url}")
        try:
            self.transport = streamablehttp_client(
                url=self.server_url,
                headers={"SCP-HUB-API-KEY": "sk-a0033dde-b3cd-413b-adbe-980bc78d6126"}
            )
            self.read, self.write, self.get_session_id = await self.transport.__aenter__()
            
            self.session_ctx = ClientSession(self.read, self.write)
            self.session = await self.session_ctx.__aenter__()

            await self.session.initialize()
            session_id = self.get_session_id()
            
            print(f"✓ connect success")
            return True
            
        except Exception as e:
            print(f"✗ connect failure: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def disconnect(self):
        try:
            if self.session:
                await self.session_ctx.__aexit__(None, None, None)
            if hasattr(self, 'transport'):
                await self.transport.__aexit__(None, None, None)
            print("✓ already disconnect")
        except Exception as e:
            print(f"✗ disconnect error: {e}")
    
    def parse_result(self, result):
        try:
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            return str(result)
        except Exception as e:
            return {"error": f"parse error: {e}", "raw": str(result)}
```

### 2. SCP Server Connection

The **initialization** and **shutdown** of the SCP server are shown below:

```python
## When start, connect the SCP server
client = DrugSDAClient(server_url)
if not await client.connect():
    print("connection failed")
    return

## When finish, disconnect the SCP server
await client.disconnect() 
```

**Note**: 

1. For most tools, the default SCP server endpoint (`server_url`) is `https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool`. 

2. Notably, for the tools listed below that operate **over long time periods**, the `server_url` is `http://180.184.86.2:32208/mcp`. 

```tex
pred_binding_affinity_boltz2

evobind_tool
gmx_mmpbsa_workflow
gmx_mmpbsa_propro
run_openawsem_simulation
prolif_analyze_pdb
```
