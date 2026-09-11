"""Calibrate official NeMo eps on cached full-corpus similarities, not quotas."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from drug_agent.toolrl.v8_dataset import sha256_file


def calibrate(scores, total, low=0.20, high=0.30):
    scores = np.asarray(scores, dtype=np.float64)
    assert 0 < low <= high < 1 and np.isfinite(scores).all()
    assert len(scores) <= total
    # The installed official stage removes exactly score >= 1 - eps.
    # Probe a broad grid and every observed boundary; ties stay together.
    grid = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05,
            0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.95]
    boundaries = [float(np.nextafter(1 - s, 1.0)) for s in np.unique(scores) if 0 < s < 1]
    candidates = sorted({e for e in grid + boundaries if 0 < e < 1})
    ordered = np.sort(scores)
    trials = [{"eps":e, "minimum_similarity":1-e,
               "retained":int(total-len(scores)+np.searchsorted(ordered,1-e,side="left"))}
              for e in candidates]
    minimum, maximum = math.ceil(total*low), math.floor(total*high)
    feasible = [t for t in trials if minimum <= t["retained"] <= maximum]
    target = total*(low+high)/2
    chosen = min(feasible, key=lambda t:(abs(t["retained"]-target),t["eps"])) if feasible else None
    return {"total":total,"target_count_range":[minimum,maximum],"target_fraction_range":[low,high],
            "selected":chosen,"grid_trials":[t for t in trials if t["eps"] in grid],
            "boundary_trials":trials,"minimum_retained_observed":min(t["retained"] for t in trials),
            "status":"target_reached" if chosen else "target_unreachable_without_changing_boundaries_or_protection"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    args = parser.parse_args()
    root = args.run_root
    prep = json.loads((root/"prepared/preparation_manifest.json").read_text())
    official = root/"official_dedup"
    report = json.loads((official/"report.json").read_text())
    assert report["prepared_text_sha256"] == sha256_file(root/"prepared/eligible_text.jsonl")
    files = [p for s in report["scopes"] if s["status"] == "official_workflow_complete"
             for p in sorted((official/s["scope_key"]/"cache/pairwise_results").glob("*.parquet"))]
    pairs = pd.concat([pd.read_parquet(p,columns=["id","max_id","cosine_sim_score"]) for p in files],ignore_index=True)
    assert len(pairs) == pairs["id"].nunique() == report["official_workflow_records"]
    result = calibrate(pairs["cosine_sim_score"],prep["records"])
    out = root/"threshold_experiment"
    out.mkdir(exist_ok=True)
    result.update(source_sha256=prep["input_sha256"],prepared_text_sha256=report["prepared_text_sha256"],
                  protected_records=prep["counts"].get("protected_window_exception",0),
                  tiny_scope_retained_records=report["tiny_scope_retained_records"],
                  pairwise_files_sha256={str(p.relative_to(root)):sha256_file(p) for p in files},
                  quality_review_complete=False,
                  method="Official cached scores; no extra clustering, quota or budget sampling")
    if result["selected"]:
        from nemo_curator.stages.deduplication.semantic import IdentifyDuplicatesStage
        from nemo_curator.tasks import FileGroupTask
        eps = result["selected"]["eps"]
        stage = IdentifyDuplicatesStage(output_path=str(out),eps=eps)
        tasks = stage.process_batch([FileGroupTask(task_id="full_threshold_calibration",dataset_name="toolrl",data=[str(p) for p in files])])
        removed = [str(i) for task in tasks for file in task.data for i in pd.read_parquet(file)["id"]]
        expected = set(pairs.loc[pairs["cosine_sim_score"] >= 1-eps,"id"])
        assert len(removed) == len(set(removed)) and set(removed) == expected
        assert prep["records"]-len(removed) == result["selected"]["retained"]
        (official/f"removed_eps_{eps}.json").write_text(json.dumps(sorted(removed))+"\n")
        report["threshold_removed_counts"][str(eps)] = len(removed)
        (official/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        (out/"selected_eps.txt").write_text(str(eps)+"\n")
        near = pairs.assign(distance=(pairs["cosine_sim_score"]-(1-eps)).abs()).sort_values(["distance","id"]).head(30)
        near.to_json(out/"threshold_near_pairs.jsonl",orient="records",lines=True)
        review_ids = set(near["id"]) | set(near["max_id"])
        source = Path(prep["input"])
        assert sha256_file(source) == prep["input_sha256"]
        coverage = {phase:{k:Counter() for k in ("tool","skill","task","trajectory")}
                    for phase in ("before","after")}
        seen = set()
        with (out/"threshold_near_original_records.jsonl").open("w") as review:
            for line in source.open():
                row = json.loads(line)
                assert row["id"] not in seen
                seen.add(row["id"])
                if row["id"] in review_ids:
                    review.write(line)
                for phase in ("before","after"):
                    if phase == "after" and row["id"] in expected:
                        continue
                    counts = coverage[phase]
                    counts["task"][row["metadata"]["task_type"]] += 1
                    counts["trajectory"][row["metadata"]["source_id"]] += 1
                    for call in row["label"]["target_assistant"].get("tool_calls",[]):
                        function = call["function"]
                        counts["tool"][function["name"]] += 1
                        if function["name"] == "skill":
                            arguments = function["arguments"]
                            if isinstance(arguments,str):
                                arguments = json.loads(arguments)
                            counts["skill"][arguments["name"]] += 1
        assert len(seen) == prep["records"] and expected | review_ids <= seen
        result["coverage"] = coverage
        result["official_identify_duplicates_verified"] = True
    (out/"threshold_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"boundary_trials","pairwise_files_sha256"}},indent=2))
    if not result["selected"]:
        raise SystemExit("No threshold achieves requested range; report saved, no forced subsampling")


if __name__ == "__main__":
    main()
