---
name: molclaw-scp-server
description: All tools utilized within MolClaw skills connect via the MCP protocol. This skill is the unified guide for connecting to the deployed MCP server before invoking tools.
license: MIT license
metadata:
    skill-author: PJLab
---

SCP (Science Context Protocol) is an open-source standard protocol designed to accelerate scientific discovery by building a global collaboration network for autonomous scientific agents, connecting heterogeneous scientific resources (software tools, AI models, datasets, workflow engines, lab instruments, etc.).

### 1. Server Definition

If MCP environment. is not installed, please run `pip install mcp`.

The server is defined as below:

```python
import json
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

DrugSDA_Tool_SERVER_URL = "https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool"
DrugSDA_Tool_API_KEY = "sk-7e8704c0-b838-44e9-ab74-f116037bbf35"

class DrugSDAClient:    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        
    async def connect(self):
        print(f"server url: {self.server_url}")
        try:
            self.transport = streamablehttp_client(
                url=self.server_url,
                headers={"SCP-HUB-API-KEY": DrugSDA_Tool_API_KEY}
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

### 2. Server Connection

The **initialization** and **shutdown** of the MCP server are shown below:

```python
## When start, connect the MCP server
client = DrugSDAClient(DrugSDA_Tool_SERVER_URL)
if not await client.connect():
    print("connection failed")
    return

## When finish, disconnect the MCP server
await client.disconnect() 
```

**Note**: The deployed MolClaw tool endpoint is `https://scp.intern-ai.org.cn/api/v1/mcp/2/DrugSDA-Tool`.
