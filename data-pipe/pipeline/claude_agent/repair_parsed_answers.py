#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
SOLUTION_RE = re.compile(r"<solution>([\s\S]*?)</solution>", re.IGNORECASE)


def extract_text_from_stream_jsonl(session_path: Path) -> str:
    if not session_path.is_file():
        return ""
    chunks: list[str] = []
    with session_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "assistant":
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt)
            elif isinstance(content, str) and content.strip():
                chunks.append(content)
    return "\n".join(chunks)


def extract_answer_block(text: str) -> str:
    raw = text or ""
    m = ANSWER_RE.search(raw)
    if m:
        return (m.group(1) or "").strip()
    m = SOLUTION_RE.search(raw)
    if m:
        return (m.group(1) or "").strip()
    return ""


def parse_answer_array(answer_block: str) -> tuple[list[str] | None, str | None]:
    if not answer_block:
        return None, "no <answer>/<solution> block found"

    block = answer_block.strip()
    parsed: Any = None
    first_error: str | None = None

    def _load_candidate(text: str) -> Any:
        s = (text or "").strip()
        if not s:
            raise json.JSONDecodeError("empty", s, 0)
        return json.loads(s)

    candidates: list[str] = [block]
    try:
        candidates.append(bytes(block, "utf-8").decode("unicode_escape").strip())
    except Exception:
        pass

    for cand in candidates:
        try:
            parsed = _load_candidate(cand)
        except json.JSONDecodeError as e:
            if first_error is None:
                first_error = str(e)
            l = cand.find("[")
            r = cand.rfind("]")
            if l != -1 and r != -1 and r > l:
                sub = cand[l : r + 1].strip()
                try:
                    parsed = _load_candidate(sub)
                except json.JSONDecodeError:
                    continue
                else:
                    break
            continue
        else:
            break

    if parsed is None:
        return None, f"answer is not valid JSON: {first_error or 'unknown parse error'}"

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as e:
            return None, f"answer JSON string is not a valid JSON array: {e}"

    if not isinstance(parsed, list):
        return None, "answer JSON is not an array"
    if not all(isinstance(x, str) for x in parsed):
        return None, "answer array contains non-string entries"

    return [str(x).strip() for x in parsed if str(x).strip()], None


def row_key(row_dir: Path) -> str:
    # row0009_idx9 -> 9
    name = row_dir.name
    if name.startswith("row") and "_idx" in name:
        part = name.split("_idx", 1)[0]
        try:
            return str(int(part.replace("row", "")))
        except ValueError:
            return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair parsed answers and preds ranking for MolBench-VS results.")
    parser.add_argument("results_dir", help="Path like .../results/molbench_vs_qwen_run_YYYYMMDD_HHMMSS")
    parser.add_argument("--re-eval", action="store_true", help="Run evaluate/run_eval_bench.py after repair")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(results_dir)

    row_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("row") and "_idx" in p.name)
    report: dict[str, Any] = {
        "results_dir": str(results_dir),
        "total_rows": len(row_dirs),
        "repaired_rows": 0,
        "failed_rows": 0,
        "rows": [],
    }

    pred_path = results_dir / "preds" / "molbench_vs" / "molbench_vs.json"
    preds: list[dict[str, Any]] = []
    pred_by_row: dict[str, dict[str, Any]] = {}
    if pred_path.is_file():
        with pred_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            preds = [x for x in obj if isinstance(x, dict)]
            for e in preds:
                rk = str(e.get("row_number") or "").strip()
                if rk:
                    pred_by_row[rk] = e

    for d in row_dirs:
        rk = row_key(d)
        parsed_path = d / "parsed_answer.json"
        session_path = d / "complete_session.jsonl"

        old = {}
        if parsed_path.is_file():
            try:
                old = json.loads(parsed_path.read_text(encoding="utf-8"))
                if not isinstance(old, dict):
                    old = {}
            except Exception:
                old = {}

        block = str(old.get("answer_block") or "")
        ranking, err = parse_answer_array(block)
        source = "parsed_answer.answer_block"

        if not ranking:
            txt = extract_text_from_stream_jsonl(session_path)
            block2 = extract_answer_block(txt)
            if not block2 and session_path.is_file():
                raw = session_path.read_text(encoding="utf-8", errors="ignore")
                block2 = extract_answer_block(raw)
            ranking2, err2 = parse_answer_array(block2)
            if ranking2:
                ranking, err, block = ranking2, None, block2
                source = "complete_session"
            else:
                if err2 and (not err or "no <answer>/<solution>" in (err or "")):
                    err = err2

        parsed_payload = {
            "row_number": int(rk) if rk else old.get("row_number"),
            "dataset_index": old.get("dataset_index"),
            "answer_block": block,
            "answer_size": len(ranking or []),
            "answer": ranking or [],
            "parse_error": err,
            "timed_out": bool(old.get("timed_out", False)),
            "timeout_sec": old.get("timeout_sec"),
            "repair_source": source,
        }
        parsed_path.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if rk and rk in pred_by_row:
            pred = pred_by_row[rk]
            jr = pred.get("json_results") if isinstance(pred.get("json_results"), dict) else {}
            jr["ranking"] = ranking or []
            jr["raw_answer"] = block
            pred["json_results"] = jr

        if ranking:
            report["repaired_rows"] += 1
        else:
            report["failed_rows"] += 1

        report["rows"].append(
            {
                "row_dir": d.name,
                "row_number": int(rk) if rk else None,
                "answer_size": len(ranking or []),
                "parse_error": err,
                "source": source,
            }
        )

    if preds:
        pred_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = results_dir / "repair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "results_dir": str(results_dir),
        "total_rows": report["total_rows"],
        "repaired_rows": report["repaired_rows"],
        "failed_rows": report["failed_rows"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))

    if args.re_eval:
        eval_script = results_dir.parent.parent / "evaluate" / "run_eval_bench.py"
        if not eval_script.is_file():
            raise FileNotFoundError(f"eval script not found: {eval_script}")
        import subprocess
        subprocess.run(["python", str(eval_script), str(results_dir)], check=True)


if __name__ == "__main__":
    main()
