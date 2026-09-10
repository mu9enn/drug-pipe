"""Official NeMo Curator stages on versioned comparison copies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from drug_agent.toolrl.v8_dataset import sha256_file

REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MODEL = "Qwen/Qwen3-Embedding-8B"
MODEL_CACHE = Path("/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub")
ENGINE = {"max_model_len":32768, "tensor_parallel_size":1, "dtype":"bfloat16",
          "runner":"pooling", "enforce_eager":True, "seed":42,
          "gpu_memory_utilization":0.80, "max_num_seqs":4, "max_num_batched_tokens":32768,
          "pooler_config":{"seq_pooling_type":"LAST", "use_activation":True,
                           "enable_chunked_processing":False}}


def comparison_scope(row):
    scope = json.loads(row["scope"])
    doc = json.loads(row["text"])
    observations = [m for m in doc["history_before_current_decision"] if m["role"] == "tool"]
    if observations and scope["prior_outcome"] == "unknown":
        last = observations[-1]
        content = str(last.get("content") or "").strip()
        if (last.get("is_error") is True or content == "The operation timed out."
                or re.match(r"^\d+ validation errors? for call\[", content)):
            scope["prior_outcome"] = "error"
    scope["explicit_execution_modes"] = [
        {key: call["function"]["arguments"][key]
         for key in ("dry_run", "mode", "operation", "action", "method")
         if key in call["function"]["arguments"]}
        for call in doc["current_teacher_response"].get("tool_calls", [])]
    scope["comparison_boundary_version"] = "native_explicit_modes_errors_v2"
    return json.dumps(scope, sort_keys=True, separators=(",", ":"))


def load_rows(root, smoke):
    rows = list(map(json.loads, (root / "prepared/eligible_text.jsonl").open()))
    for row in rows:
        row["scope"] = comparison_scope(row)
    if smoke:
        groups = defaultdict(list)
        for row in rows:
            groups[row["scope"]].append(row)
        # Real comparable pairs plus skill/error/final and length boundaries.
        ordered = sorted(groups.values(), key=lambda group:(-len(group),group[0]["id"]))
        chosen = ordered[:2]
        for field, value in (("ordered_tool_names", ["skill"]), ("prior_outcome", "error"), ("decision_type", "final_answer")):
            group = next(group for group in ordered if json.loads(group[0]["scope"]).get(field) == value)
            chosen.append(group)
        chosen.append(next(group for group in ordered if len(group) == 3))
        sample = {row["id"]:row for group in chosen for row in group[:8]}
        longest = max(rows, key=lambda row:(row["tokens"],row["id"]))
        sample[longest["id"]] = longest
        changed = {r["id"] for r in map(json.loads, (root / "prepared/encoding_audit.jsonl").open())
                   if r["disposition"] == "eligible" and r["window"]["lossy"]}
        for row in sorted((r for r in rows if r["id"] in changed), key=lambda r:r["id"])[:8]:
            sample[row["id"]] = row
        rows = list(sample.values())
    return rows


def encode(root, smoke):
    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
    from nemo_curator.tasks import DocumentBatch
    from transformers import AutoTokenizer

    rows = load_rows(root, smoke)
    if not rows:
        raise ValueError("No eligible real inputs; cannot claim an embedding run")
    cache = root / "embedding_cache"
    cache.mkdir(exist_ok=True)
    binding = {"model":MODEL, "revision":REVISION, "engine":ENGINE,
               "preparation_sha256":sha256_file(root / "prepared/preparation_manifest.json"),
               "nemo_curator":importlib.metadata.version("nemo-curator"),
               "vllm":importlib.metadata.version("vllm"), "max_chars":None,"pretokenize":False}
    manifest = cache / "binding.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == binding
    else:
        manifest.write_text(json.dumps(binding, indent=2)+"\n")
    model_path = MODEL_CACHE / "models--Qwen--Qwen3-Embedding-8B/snapshots" / REVISION
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # NeMo resolves the cache's default revision. Bind that reference and
    # run with HF_HUB_OFFLINE=1; never download or silently update a revision.
    assert (MODEL_CACHE / "models--Qwen--Qwen3-Embedding-8B/refs/main").read_text().strip() == REVISION
    stage = VLLMEmbeddingModelStage(MODEL, vllm_init_kwargs=ENGINE,
                                    pretokenize=False, max_chars=None, cache_dir=str(MODEL_CACHE))
    stage.setup()
    original_embed = stage.model.embed
    def checked_embed(texts, **kwargs):
        expected = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        assert all(len(ids)<=32768 for ids in expected)
        # NeMo calls truncate_prompt_tokens=-1. Passing only validated inputs
        # and checking the actual returned token IDs proves no input was lost.
        outputs = original_embed(texts, **kwargs)
        assert len(outputs) == len(expected)
        for result, ids in zip(outputs, expected, strict=True):
            assert result.prompt_token_ids == ids, "vLLM changed/truncated encoder input"
        return outputs
    stage.model.embed = checked_embed
    try:
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset+8]
            pending = [r for r in batch if not (cache / (hashlib.sha256(r["id"].encode()).hexdigest()+".npz")).exists()]
            if not pending:
                continue
            for row in pending:
                assert len(tokenizer.encode(row["text"], truncation=False, add_special_tokens=True)) == row["tokens"]
            try:
                output = stage.process(DocumentBatch(task_id=str(offset),dataset_name="toolrl",data=pd.DataFrame(pending))).to_pandas()
            except Exception as error:
                with (root / "encoding_failures.jsonl").open("a") as log:
                    log.write(json.dumps({"ids":[r["id"] for r in pending],
                                          "error_type":type(error).__name__,"error":str(error),
                                          "disposition":"batch_failed_not_duplicates; retry required"})+"\n")
                raise
            assert output["id"].tolist() == [r["id"] for r in pending]
            for row, vector in zip(pending, output["embeddings"], strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                assert vector.shape == (4096,) and np.isfinite(vector).all() and np.linalg.norm(vector)>0
                vector /= np.linalg.norm(vector)
                np.savez(cache / (hashlib.sha256(row["id"].encode()).hexdigest()+".npz"),
                         vector=vector, id=row["id"], text_sha256=hashlib.sha256(row["text"].encode()).hexdigest())
            print(f"embedded {min(offset+8,len(rows))}/{len(rows)}",flush=True)
    finally:
        stage.model.embed = original_embed
        del original_embed
        stage.teardown()
    (root / ("smoke_encoding.json" if smoke else "encoding_complete.json")).write_text(json.dumps({"records":len(rows),"binding":binding},indent=2)+"\n")


def dedup(root, smoke):
    import ray
    import pyarrow as pa
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow, IdentifyDuplicatesStage
    from nemo_curator.tasks import FileGroupTask

    rows = load_rows(root, smoke)
    groups = defaultdict(list)
    for row in rows:
        path = root / "embedding_cache" / (hashlib.sha256(row["id"].encode()).hexdigest()+".npz")
        vector = np.load(path, allow_pickle=False)
        assert str(vector["id"]) == row["id"]
        assert str(vector["text_sha256"]) == hashlib.sha256(row["text"].encode()).hexdigest()
        assert abs(float(np.linalg.norm(vector["vector"]))-1)<1e-5
        groups[row["scope"]].append({"id":row["id"], "embeddings":vector["vector"].tolist()})
    out = root / ("smoke_dedup" if smoke else "official_dedup")
    out.mkdir(exist_ok=True)
    ray.init(num_gpus=1, num_cpus=8, include_dashboard=False, object_store_memory=512*1024**2)
    summaries = []
    thresholds = (0.0001, 0.001, 0.005, 0.01)
    removed = {str(eps):[] for eps in thresholds}
    try:
        for scope, members in sorted(groups.items()):
            key = hashlib.sha256(scope.encode()).hexdigest()[:20]
            work = out / key
            work.mkdir(exist_ok=True)
            summary = {"scope":json.loads(scope),"scope_key":key,"records":len(members)}
            if len(members)<3:
                summary["status"] = "tiny_scope_retained"
                summaries.append(summary)
                continue
            pd.DataFrame(members).to_parquet(work / "input.parquet",index=False,
                schema=pa.schema([("id",pa.string()),("embeddings",pa.list_(pa.float32()))]))
            clusters = max(1, math.ceil(len(members)/256))
            workflow = SemanticDeduplicationWorkflow(input_path=[str(work / "input.parquet")],
                output_path=str(work / "result"),cache_path=str(work / "cache"),n_clusters=clusters,
                embedding_dim=4096,which_to_keep="hard",random_state=42,eps=None,
                pairwise_batch_size=256,verbose=False)
            done = work / "similarities_complete.json"
            if not done.exists():
                workflow.run(pairwise_executor=RayActorPoolExecutor(show_progress=False))
            files = sorted((work / "cache/pairwise_results").glob("*.parquet"))
            assert files
            pairs = pd.concat([pd.read_parquet(file, columns=["id","max_id","cosine_sim_score"]) for file in files],ignore_index=True)
            member_ids = {m["id"] for m in members}
            assert len(pairs) == len(member_ids) and set(pairs["id"]) == member_ids, "Incomplete official pairwise results"
            assert set(pairs["max_id"]) <= member_ids and np.isfinite(pairs["cosine_sim_score"]).all()
            done.write_text(json.dumps({"clusters":clusters,"records":len(members),"all_pairwise_ids_verified":True})+"\n")
            for eps in thresholds:
                destination = work / f"eps_{eps}"
                destination.mkdir(exist_ok=True)
                stage = IdentifyDuplicatesStage(output_path=str(destination),eps=eps)
                tasks = stage.process_batch([FileGroupTask(task_id=key,dataset_name=key,data=[str(p) for p in files])])
                ids = [str(ident) for task in tasks for file in task.data for ident in pd.read_parquet(file)["id"]]
                assert set(ids) == set(pairs.loc[pairs["cosine_sim_score"] >= 1-eps,"id"])
                removed[str(eps)].extend(ids)
            summary.update(status="official_workflow_complete",n_clusters=clusters)
            summaries.append(summary)
            print("scopes",len(summaries),"/",len(groups),flush=True)
    finally:
        ray.shutdown()
    for eps, ids in removed.items():
        assert len(ids)==len(set(ids))
        (out / f"removed_eps_{eps}.json").write_text(json.dumps(sorted(ids),indent=2)+"\n")
    report = {"records":len(rows),"scopes":summaries,
              "comparison_boundary_version":"native_explicit_modes_errors_v2",
              "official_workflow_records":sum(s["records"] for s in summaries if s["status"] == "official_workflow_complete"),
              "tiny_scope_retained_records":sum(s["records"] for s in summaries if s["status"] == "tiny_scope_retained"),
              "threshold_removed_counts":{e:len(ids) for e,ids in removed.items()},
              "eps_meaning":"remove if official cosine_sim_score >= 1 - eps", "which_to_keep":"hard",
              "keep_rule":"official ranking: farthest from centroid first; stable ID tie-breaker",
              "random_state":42,"embedding_dtype":"float32","pairwise_relationship":"id -> max_id (direct comparison, not necessarily a finally retained representative)",
              "versions":{name:importlib.metadata.version(name) for name in ("nemo-curator","vllm","cudf-cu12","cuml-cu12","torch","ray")}}
    (out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="scopes"},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("encode","dedup"))
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    {"encode":encode,"dedup":dedup}[args.operation](args.run_root.resolve(),args.smoke)
