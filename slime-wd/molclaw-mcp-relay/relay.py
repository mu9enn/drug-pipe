#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import time


LISTEN_HOST = os.environ.get("MCP_RELAY_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("MCP_RELAY_LISTEN_PORT", "13208"))

TARGET_HOST = os.environ.get("MCP_RELAY_TARGET_HOST", "httpproxy-headless.kubebrain.svc.pjlab.local")
TARGET_PORT = int(os.environ.get("MCP_RELAY_TARGET_PORT", "3128"))


async def pipe(name: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    total = 0
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                print(f"{name}: EOF, bytes={total}", flush=True)
                break
            total += len(data)
            print(f"{name}: {len(data)} bytes, total={total}", flush=True)
            writer.write(data)
            await writer.drain()
    except Exception as exc:
        print(f"{name}: pipe error: {exc!r}, bytes={total}", flush=True)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def handle_client(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    peer = client_writer.get_extra_info("peername")
    conn_id = f"{peer}-{int(time.time())}"
    print(f"[{conn_id}] new connection", flush=True)
    try:
        target_reader, target_writer = await asyncio.open_connection(TARGET_HOST, TARGET_PORT)
        print(f"[{conn_id}] connected upstream proxy {TARGET_HOST}:{TARGET_PORT}", flush=True)
    except Exception as exc:
        print(f"[{conn_id}] failed to connect upstream proxy: {exc!r}", flush=True)
        client_writer.close()
        await client_writer.wait_closed()
        return

    await asyncio.gather(
        pipe(f"[{conn_id}] client->proxy", client_reader, target_writer),
        pipe(f"[{conn_id}] proxy->client", target_reader, client_writer),
        return_exceptions=True,
    )
    print(f"[{conn_id}] connection closed", flush=True)


async def main() -> None:
    server = await asyncio.start_server(handle_client, LISTEN_HOST, LISTEN_PORT)
    print(
        f"proxy listening on {LISTEN_HOST}:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
