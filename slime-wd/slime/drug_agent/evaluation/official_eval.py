from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_eval_runner(molbench_root: Path):
    path = molbench_root / "eval/eval_runner.py"
    if not path.is_file():
        raise FileNotFoundError(f"MolBench evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("drug_agent_external_molbench_eval_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MolBench evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_official_evaluation(run_dir: str | Path, molbench_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    bench_root = Path(molbench_root).resolve()
    module = _load_eval_runner(bench_root)
    eval_root = bench_root / "ChemCoTBench/baseline_and_eval"
    if not eval_root.is_dir():
        raise FileNotFoundError(f"ChemCoTBench official evaluator assets not found: {eval_root}")

    jobs = [
        (module.RdkitBenchEval(), run_root / "preds/rdkit_bench"),
        (module.ACNetCuratedEval(), run_root / "preds/acnet_curated"),
        (module.MolbenchVsEval(), run_root / "preds/molbench_vs"),
        (module.ChemCoTBenchMolEditEval(), run_root / "preds/molbench-mo-edit"),
        (module.ChemCoTBenchMolOptPhyschemEval(), run_root / "preds/molbench-mo-opt"),
    ]
    metrics: dict[str, Any] = {}
    for evaluator, preds_dir in jobs:
        if not preds_dir.is_dir():
            raise FileNotFoundError(f"Expected prediction directory missing: {preds_dir}")
        result = evaluator.run(str(preds_dir), str(run_root), str(eval_root))
        if result:
            metrics.update(result)

    path = run_root / "metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
