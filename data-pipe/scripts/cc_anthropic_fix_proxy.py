#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


UPSTREAM_BASE = "http://10.12.111.135:10010"

# 建议运行前 export TOKEN='你的JWT'
# 也可以直接把 token 粘到这里。
TOKEN = os.environ.get("TOKEN", "PASTE_YOUR_JWT_TOKEN_HERE")

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 10011

app = FastAPI()


def content_to_text(content: Any) -> str:
    """
    把 Anthropic content 里的 string / text blocks 转成 system 字符串。
    对非 text block 做保守 JSON 序列化，避免信息丢失太多。
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    parts.append(block.get("thinking", ""))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)

    return str(content)


def normalize_messages_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容 Claude Code 新版可能发出的 messages 内 system role。
    目标网关报错显示它只接受 messages[].role in {"user", "assistant"}。
    """
    body = dict(body)

    system_parts: List[str] = []

    # 保留原本顶层 system
    if body.get("system"):
        system_parts.append(content_to_text(body.get("system")))

    new_messages = []
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue

        role = msg.get("role")

        if role == "system":
            system_parts.append(content_to_text(msg.get("content")))
            continue

        new_messages.append(msg)

    body["messages"] = new_messages

    if system_parts:
        body["system"] = "\n\n".join(p for p in system_parts if p)

    return body


def upstream_headers(request: Request) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    # Claude Code 可能会带 beta header，原样转发。
    if request.headers.get("anthropic-beta"):
        headers["anthropic-beta"] = request.headers["anthropic-beta"]

    return headers


@app.get("/")
async def health():
    return {"ok": True, "upstream": UPSTREAM_BASE}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen", "type": "model", "display_name": "qwen"},
            {"id": "minimax", "type": "model", "display_name": "minimax"},
        ],
    }


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    body = normalize_messages_body(body)

    url = f"{UPSTREAM_BASE}/v1/messages"
    headers = upstream_headers(request)

    is_stream = body.get("stream") is True

    if is_stream:
        client = httpx.AsyncClient(timeout=None)
        req = client.build_request("POST", url, headers=headers, json=body)
        upstream_resp = await client.send(req, stream=True)

        async def iter_upstream():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()
                await client.aclose()

        return StreamingResponse(
            iter_upstream(),
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_resp = await client.post(url, headers=headers, json=body)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    import uvicorn

    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIwODk3NzQyNjgsImlhdCI6MTc3NDQxNDI2OCwiaXNzIjoibGxtLWdhdGV3YXkiLCJwcm9qZWN0IjoiYWk0cy1kaXNjb3ZlcnkiLCJyb2xlIjoicHJvamVjdCIsInVzZXJuYW1lIjoiYWk0cy1kaXNjb3ZlcnkifQ.Vw5EGFE5TxulXVC4rg0AzqfGKEzJ_TO66t4WVwf-rKM"


    # if not TOKEN or TOKEN == "PASTE_YOUR_JWT_TOKEN_HERE":
    #     raise RuntimeError("Please export TOKEN first or paste it into TOKEN in the script.")

    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)