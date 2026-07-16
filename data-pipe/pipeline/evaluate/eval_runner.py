from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


def _as_smiles_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return [line.strip() for line in s.splitlines() if line.strip()]
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    return []


def _canonicalize_list(smiles_list: List[str], chem_module: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    out: List[str] = []
    errors: List[Dict[str, Any]] = []
    for i, s in enumerate(smiles_list):
        mol = chem_module.MolFromSmiles(s)
        if mol is None:
            errors.append({"index": i, "smiles": s, "reason": "invalid_smiles"})
            continue
        out.append(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return out, errors


def _hit_num(pred_ranking: List[str], gt_answers: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_topk = pred_ranking[:k]
    gt_set = set(gt_answers)
    return float(sum(1 for x in pred_topk if x in gt_set))


def _load_rdkit() -> tuple[bool, Any, str | None]:
    try:
        from rdkit import Chem  # type: ignore

        return True, Chem, None
    except Exception as e:  # pragma: no cover - environment-dependent
        return False, None, str(e)


def eval_molbench_vs_file(pred_json_path: str) -> Dict[str, Any]:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Prediction file must be a JSON list: {pred_json_path}")

    rdkit_available, Chem, rdkit_error = _load_rdkit()

    top3_hits: List[float] = []
    top10_hits: List[float] = []
    quality_hist: Counter[str] = Counter()
    invalid_smiles_counter: Counter[str] = Counter()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        answers_raw = _as_smiles_list(entry.get("answer"))
        jr = entry.get("json_results") if isinstance(entry.get("json_results"), dict) else {}
        ranking_raw = _as_smiles_list(jr.get("ranking"))
        candidates_raw = _as_smiles_list(entry.get("candidates"))

        answers = answers_raw
        ranking = ranking_raw
        candidates = candidates_raw
        canon_meta: Dict[str, Any] = {"enabled": False, "rdkit_available": rdkit_available}

        if rdkit_available:
            cand_canon, cand_err = _canonicalize_list(candidates_raw, Chem)
            ans_canon, ans_err = _canonicalize_list(answers_raw, Chem)
            rank_canon, rank_err = _canonicalize_list(ranking_raw, Chem)
            answers = ans_canon
            ranking = rank_canon
            candidates = cand_canon
            canon_meta = {
                "enabled": True,
                "candidate_invalid_count": len(cand_err),
                "answer_invalid_count": len(ans_err),
                "ranking_invalid_count": len(rank_err),
            }
            if cand_err:
                invalid_smiles_counter["candidate"] += len(cand_err)
            if ans_err:
                invalid_smiles_counter["answer"] += len(ans_err)
            if rank_err:
                invalid_smiles_counter["ranking"] += len(rank_err)
        else:
            canon_meta["rdkit_error"] = rdkit_error

        top3 = _hit_num(ranking, answers, 3)
        top10 = _hit_num(ranking, answers, 10)
        top3_hits.append(top3)
        top10_hits.append(top10)

        if candidates:
            if len(ranking) != len(candidates):
                quality_hist["length_mismatch"] += 1
            outside = sum(1 for x in ranking if x not in set(candidates))
            if outside:
                quality_hist["outside_candidate_set"] += 1
        else:
            quality_hist["empty_candidate_set"] += 1
        if len(set(ranking)) != len(ranking):
            quality_hist["duplicate_predictions"] += 1

        entry["metrics"] = {
            "top3_hit_num": top3,
            "top10_hit_num": top10,
        }
        entry["eval_audit"] = {
            "candidate_size": len(candidates),
            "prediction_size": len(ranking),
            "is_length_match": len(ranking) == len(candidates) if candidates else False,
            "duplicate_prediction_count": len(ranking) - len(set(ranking)),
            "outside_candidate_count": sum(1 for x in ranking if x not in set(candidates)) if candidates else 0,
            "canonicalization": canon_meta,
        }

    n = len(entries)
    result = {
        "molbench_vs_molbench_vs": {
            "top3_avg_hit_num": (sum(top3_hits) / n) if n else 0.0,
            "top10_avg_hit_num": (sum(top10_hits) / n) if n else 0.0,
            "n_samples": n,
        },
        "audit": {
            "rdkit_available": rdkit_available,
            "rdkit_error": rdkit_error,
            "quality_issue_hist": dict(quality_hist),
            "invalid_smiles_hist": dict(invalid_smiles_counter),
        },
    }

    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return result


def eval_molbench_ac_file(pred_json_path: str) -> Dict[str, Any]:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Prediction file must be a JSON list: {pred_json_path}")

    rdkit_available, Chem, rdkit_error = _load_rdkit()

    correct = 0
    valid = 0
    invalid_smiles_counter: Counter[str] = Counter()
    reject_reason_hist: Counter[str] = Counter()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        gt_raw = _as_smiles_list(entry.get("answer"))
        jr = entry.get("json_results") if isinstance(entry.get("json_results"), dict) else {}
        pred_raw = _as_smiles_list(jr.get("prediction"))

        gt = gt_raw[:1]
        pred = pred_raw[:1]

        if not pred_raw:
            reject_reason_hist["empty_pred"] += 1
        if not gt_raw:
            reject_reason_hist["empty_gt"] += 1

        if rdkit_available:
            gt_canon, gt_err = _canonicalize_list(gt, Chem)
            pred_canon, pred_err = _canonicalize_list(pred, Chem)
            if gt_err:
                invalid_smiles_counter["answer"] += len(gt_err)
                reject_reason_hist["invalid_smiles_gt"] += len(gt_err)
            if pred_err:
                invalid_smiles_counter["prediction"] += len(pred_err)
                reject_reason_hist["invalid_smiles_pred"] += len(pred_err)
            gt = gt_canon
            pred = pred_canon

        if gt and pred:
            valid += 1
            if gt[0] == pred[0]:
                correct += 1
        else:
            if gt and not pred:
                reject_reason_hist["no_valid_prediction_after_canonicalize"] += 1
            if pred and not gt:
                reject_reason_hist["no_valid_ground_truth_after_canonicalize"] += 1

        entry["metrics"] = {
            "is_correct": bool(gt and pred and gt[0] == pred[0]),
        }
        entry["eval_audit"] = {
            "ground_truth_size": len(gt),
            "prediction_size": len(pred),
            "rdkit_available": rdkit_available,
            "pred_raw_size": len(pred_raw),
            "gt_raw_size": len(gt_raw),
        }

    n = len(entries)
    result = {
        "molbench_ac_molbench_ac": {
            "accuracy": (correct / valid) if valid else 0.0,
            "n_samples": n,
            "n_valid_scored": valid,
        },
        "audit": {
            "rdkit_available": rdkit_available,
            "rdkit_error": rdkit_error,
            "invalid_smiles_hist": dict(invalid_smiles_counter),
            "reject_reason_hist": dict(reject_reason_hist),
        },
    }

    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return result


def _set_f1(pred: List[str], gt: List[str]) -> tuple[float, float, float]:
    pset = set(pred)
    gset = set(gt)
    if not pset and not gset:
        return 1.0, 1.0, 1.0
    if not pset or not gset:
        return 0.0, 0.0, 0.0
    tp = len(pset & gset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def eval_molbench_pf_file(pred_json_path: str) -> Dict[str, Any]:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError(f"Prediction file must be a JSON list: {pred_json_path}")

    rdkit_available, Chem, rdkit_error = _load_rdkit()

    exact = 0
    total_f1 = 0.0
    total = 0
    single_total = 0
    single_correct = 0
    invalid_smiles_counter: Counter[str] = Counter()

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        gt_raw = _as_smiles_list(entry.get("answer"))
        jr = entry.get("json_results") if isinstance(entry.get("json_results"), dict) else {}
        pred_raw = _as_smiles_list(jr.get("prediction"))

        gt = gt_raw
        pred = pred_raw

        if rdkit_available:
            gt_canon, gt_err = _canonicalize_list(gt, Chem)
            pred_canon, pred_err = _canonicalize_list(pred, Chem)
            if gt_err:
                invalid_smiles_counter["answer"] += len(gt_err)
            if pred_err:
                invalid_smiles_counter["prediction"] += len(pred_err)
            gt = gt_canon
            pred = pred_canon

        precision, recall, f1 = _set_f1(pred, gt)
        is_exact = set(pred) == set(gt)
        exact += int(is_exact)
        total_f1 += f1
        total += 1

        if len(gt) == 1:
            single_total += 1
            if pred and pred[0] == gt[0]:
                single_correct += 1

        entry["metrics"] = {
            "exact_set_match": bool(is_exact),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        entry["eval_audit"] = {
            "ground_truth_size": len(gt),
            "prediction_size": len(pred),
            "rdkit_available": rdkit_available,
        }

    result = {
        "molbench_pf_molbench_pf": {
            "exact_set_match_rate": (exact / total) if total else 0.0,
            "avg_f1": (total_f1 / total) if total else 0.0,
            "single_answer_accuracy": (single_correct / single_total) if single_total else 0.0,
            "n_samples": total,
        },
        "audit": {
            "rdkit_available": rdkit_available,
            "rdkit_error": rdkit_error,
            "invalid_smiles_hist": dict(invalid_smiles_counter),
        },
    }

    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return result


def _infer_task_from_results_dir(results_dir: str) -> str:
    cfg_path = os.path.join(results_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        try:
            cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
            task = str(cfg.get("task") or "").strip().lower()
            if task in {"vs", "ac", "pf"}:
                return task
        except Exception:
            pass

    if os.path.isfile(os.path.join(results_dir, "preds", "molbench_vs", "molbench_vs.json")):
        return "vs"
    if os.path.isfile(os.path.join(results_dir, "preds", "molbench_ac", "molbench_ac.json")):
        return "ac"
    if os.path.isfile(os.path.join(results_dir, "preds", "molbench_pf", "molbench_pf.json")):
        return "pf"

    raise FileNotFoundError(f"Cannot infer task from results_dir: {results_dir}")


def eval_results_dir(results_dir: str, task: str | None = None) -> Dict[str, Any]:
    results_dir = os.path.abspath(results_dir)
    resolved_task = (task or "").strip().lower() or _infer_task_from_results_dir(results_dir)

    if resolved_task == "vs":
        pred_json_path = os.path.join(results_dir, "preds", "molbench_vs", "molbench_vs.json")
        if not os.path.isfile(pred_json_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_json_path}")
        return eval_molbench_vs_file(pred_json_path)

    if resolved_task == "ac":
        pred_json_path = os.path.join(results_dir, "preds", "molbench_ac", "molbench_ac.json")
        if not os.path.isfile(pred_json_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_json_path}")
        return eval_molbench_ac_file(pred_json_path)

    if resolved_task == "pf":
        pred_json_path = os.path.join(results_dir, "preds", "molbench_pf", "molbench_pf.json")
        if not os.path.isfile(pred_json_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_json_path}")
        return eval_molbench_pf_file(pred_json_path)

    raise ValueError(f"Unsupported task: {resolved_task}")
