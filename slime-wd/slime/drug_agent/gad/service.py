from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from drug_agent.gad.discriminator import GADDiscriminator

DISCRIMINATOR: GADDiscriminator | None = None
LOCK = asyncio.Lock()
CONFIG: dict[str, Any] = {}
LAST_METRICS: dict[str, Any] = {}


def _require_server():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("GAD service requires fastapi and uvicorn on the discriminator worker") from exc
    return FastAPI, HTTPException


FastAPI, HTTPException = _require_server()
app = FastAPI(title="Drug Agent GAD Discriminator")


@app.get("/health")
async def health():
    return {
        "ok": DISCRIMINATOR is not None,
        "version": None if DISCRIMINATOR is None else DISCRIMINATOR.version,
        "model_path": None if DISCRIMINATOR is None else DISCRIMINATOR.model_path,
    }


@app.get("/metrics")
async def metrics():
    running = {}
    if DISCRIMINATOR is not None:
        variance = DISCRIMINATOR.running_m2 / max(1, DISCRIMINATOR.running_count - 1)
        running = {
            "count": DISCRIMINATOR.running_count,
            "mean": DISCRIMINATOR.running_mean,
            "std": variance**0.5,
        }
    return {
        "version": None if DISCRIMINATOR is None else DISCRIMINATOR.version,
        "metrics": LAST_METRICS,
        "reward_normalization": running,
    }


@app.post("/score-and-update")
async def score_and_update(payload: dict[str, Any]):
    global LAST_METRICS
    if DISCRIMINATOR is None:
        raise HTTPException(status_code=503, detail="discriminator not initialized")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list")
    try:
        states = [item["state_messages"] for item in items]
        teacher = [item["teacher_response"] for item in items]
        student = [item["student_response"] for item in items]
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid aligned pair: {exc}") from exc
    if any(not isinstance(state, list) for state in states):
        raise HTTPException(status_code=400, detail="every state_messages value must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in teacher):
        raise HTTPException(status_code=400, detail="teacher_response must be a non-empty string")
    if any(not isinstance(value, str) for value in student):
        raise HTTPException(status_code=400, detail="student_response must be a string")
    update_steps = int(payload.get("update_steps") or CONFIG["update_steps"])
    if update_steps < 1:
        raise HTTPException(status_code=400, detail="update_steps must be >= 1")
    async with LOCK:
        result = DISCRIMINATOR.score_and_update(
            states,
            teacher,
            student,
            update_steps=update_steps,
            clip_grad=float(payload.get("clip_grad") or CONFIG["clip_grad"]),
            reward_clip=float(payload.get("reward_clip") or CONFIG["reward_clip"]),
        )
        LAST_METRICS = result["metrics"]
        if DISCRIMINATOR.version % CONFIG["save_interval"] == 0:
            DISCRIMINATOR.save(str(Path(CONFIG["output_dir"]) / f"version_{DISCRIMINATOR.version:06d}"))
        return result


@app.post("/checkpoint")
async def checkpoint(payload: dict[str, Any] | None = None):
    if DISCRIMINATOR is None:
        raise HTTPException(status_code=503, detail="discriminator not initialized")
    async with LOCK:
        path = (payload or {}).get("path") or str(Path(CONFIG["output_dir"]) / "latest")
        DISCRIMINATOR.save(path)
        return {"ok": True, "path": path, "version": DISCRIMINATOR.version}


def main() -> int:
    global DISCRIMINATOR, CONFIG
    parser = argparse.ArgumentParser(description="Serve and continuously update the GAD discriminator")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--update-steps", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=2.0)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()
    if args.update_steps < 1 or args.save_interval < 1 or args.reward_clip <= 0:
        parser.error("--update-steps/--save-interval must be >= 1 and --reward-clip must be > 0")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resume_backbone = Path(args.resume) / "backbone" if args.resume else None
    init_path = str(resume_backbone) if resume_backbone and resume_backbone.is_dir() else args.model_path
    DISCRIMINATOR = GADDiscriminator(init_path, lr=args.lr, max_length=args.max_length)
    if args.resume:
        DISCRIMINATOR.load(args.resume)
    CONFIG = vars(args)
    import uvicorn

    print(json.dumps({"event": "gad_service_start", **CONFIG}, ensure_ascii=False), flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
