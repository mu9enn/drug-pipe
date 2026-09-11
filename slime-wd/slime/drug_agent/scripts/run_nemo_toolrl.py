"""Official NeMo Curator stages on versioned comparison copies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from drug_agent.toolrl.v8_dataset import sha256_file
from drug_agent.toolrl.nemo_scope import VERSION as SCOPE_VERSION

from drug_agent.toolrl.nemo_config import MODEL, MODEL_CACHE, MODEL_PATH, REVISION, ENGINE


def embedding_binding():
    return {"model":MODEL, "revision":REVISION, "engine":ENGINE,
            "nemo_curator":importlib.metadata.version("nemo-curator"),
            "vllm":importlib.metadata.version("vllm"), "max_chars":None,"pretokenize":False}


def read_embedding(path, row):
    with np.load(path, allow_pickle=False) as saved:
        assert str(saved["id"]) == row["id"]
        assert str(saved["text_sha256"]) == hashlib.sha256(row["text"].encode()).hexdigest(), "Cached text changed"
        vector = saved["vector"]
        assert vector.shape == (4096,) and np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector))-1)<1e-5
        return vector


def embed_checked(embed, texts, expected, **kwargs):
    assert len(texts) == len(expected) and all(len(ids)<=32768 for ids in expected)
    # vLLM 0.14.1 caps RUNNING requests at max_model_len - 1. A full-limit
    # pooling request split across scheduling steps can therefore stall. Run
    # these rare batches one request at a time, so each complete prefill fits
    # the 32768-token budget. Text, model, pooling and output order are unchanged.
    batches = [[text] for text in texts] if any(len(ids)==32768 for ids in expected) else [texts]
    outputs = [result for batch in batches for result in embed(batch, **kwargs)]
    assert len(outputs) == len(expected)
    for result, ids in zip(outputs, expected, strict=True):
        assert result.prompt_token_ids == ids, "vLLM changed/truncated encoder input"
    return outputs


def load_rows(root, smoke):
    prep = json.loads((root / "prepared/preparation_manifest.json").read_text())
    assert prep["comparison_boundary_version"] == SCOPE_VERSION, "Prepare native comparison scopes first"
    rows = list(map(json.loads, (root / "prepared/eligible_text.jsonl").open()))
    if smoke:
        groups = defaultdict(list)
        for row in rows:
            groups[row["scope"]].append(row)
        # Real comparable pairs plus skill/error/final and length boundaries.
        ordered = sorted(groups.values(), key=lambda group:(-len(group),group[0]["id"]))
        chosen = ordered[:2]
        for predicate in (lambda s: any(c["name"] == "skill" for c in s.get("calls", [])),
                          lambda s: s["prior_outcome"] == "error",
                          lambda s: s["decision_type"] == "final_answer"):
            group = next((g for g in ordered if predicate(json.loads(g[0]["scope"]))), None)
            if group:
                chosen.append(group)
        # Include both ends of each native scope so the pilot does not only
        # inspect the first task family in a canonically ordered source.
        sample = {row["id"]:row for group in chosen for row in group[:4]+group[-4:]}
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
    binding = embedding_binding()
    manifest = cache / "binding.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == binding
    else:
        manifest.write_text(json.dumps(binding, indent=2)+"\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    # NeMo resolves the cache's default revision. Bind that reference and
    # run with HF_HUB_OFFLINE=1; never download or silently update a revision.
    assert (MODEL_CACHE / "models--Qwen--Qwen3-Embedding-8B/refs/main").read_text().strip() == REVISION
    stage = VLLMEmbeddingModelStage(MODEL, vllm_init_kwargs=ENGINE,
                                    pretokenize=False, max_chars=None, cache_dir=str(MODEL_CACHE))
    stage.setup()
    original_embed = stage.model.embed
    def checked_embed(texts, **kwargs):
        expected = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        # NeMo calls truncate_prompt_tokens=-1. Passing only validated inputs
        # and checking the actual returned token IDs proves no input was lost.
        return embed_checked(original_embed, texts, expected, **kwargs)
    stage.model.embed = checked_embed
    cached_records = new_records = 0
    try:
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset+8]
            pending = []
            for row in batch:
                path = cache / (hashlib.sha256(row["id"].encode()).hexdigest()+".npz")
                if path.exists():
                    read_embedding(path, row)
                    cached_records += 1
                else:
                    pending.append(row)
            if not pending:
                continue
            for row in pending:
                assert len(tokenizer.encode(row["text"], truncation=False, add_special_tokens=True)) == row["tokens"]
            print("encoding batch", offset, [(r["id"],r["tokens"]) for r in pending], flush=True)
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
            new_records += len(pending)
            print(f"embedded {min(offset+8,len(rows))}/{len(rows)}",flush=True)
    finally:
        stage.model.embed = original_embed
        del original_embed
        stage.teardown()
    (root / ("smoke_encoding.json" if smoke else "encoding_complete.json")).write_text(json.dumps(
        {"records":len(rows),"newly_encoded_records":new_records,"verified_cached_records":cached_records,
         "full_limit_requests_isolated":True,"binding":binding},indent=2)+"\n")


def dedup(root, smoke, eps):
    import ray
    import pyarrow as pa
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow, IdentifyDuplicatesStage
    from nemo_curator.tasks import FileGroupTask

    rows = load_rows(root, smoke)
    assert json.loads((root / "embedding_cache/binding.json").read_text()) == embedding_binding()
    groups = defaultdict(list)
    for row in rows:
        path = root / "embedding_cache" / (hashlib.sha256(row["id"].encode()).hexdigest()+".npz")
        vector = read_embedding(path, row)
        groups[row["scope"]].append({"id":row["id"], "embeddings":vector.tolist()})
    out = root / ("smoke_dedup" if smoke else "official_dedup")
    out.mkdir(exist_ok=True)
    ray.init(num_gpus=1, num_cpus=8, include_dashboard=False, object_store_memory=512*1024**2)
    summaries = []
    removed = []
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
            input_sha = sha256_file(work / "input.parquet")
            if done.exists():
                assert json.loads(done.read_text())["input_sha256"] == input_sha, "Similarity cache input changed; use a new run directory"
            if not done.exists():
                workflow.run(pairwise_executor=RayActorPoolExecutor(show_progress=False))
            files = sorted((work / "cache/pairwise_results").glob("*.parquet"))
            assert files
            pairs = pd.concat([pd.read_parquet(file, columns=["id","max_id","cosine_sim_score"]) for file in files],ignore_index=True)
            member_ids = {m["id"] for m in members}
            assert len(pairs) == len(member_ids) and set(pairs["id"]) == member_ids, "Incomplete official pairwise results"
            assert set(pairs["max_id"]) <= member_ids and np.isfinite(pairs["cosine_sim_score"]).all()
            done.write_text(json.dumps({"input_sha256":input_sha,"clusters":clusters,"records":len(members),"all_pairwise_ids_verified":True})+"\n")
            destination = work / f"eps_{eps}"
            destination.mkdir(exist_ok=True)
            stage = IdentifyDuplicatesStage(output_path=str(destination),eps=eps)
            tasks = stage.process_batch([FileGroupTask(task_id=key,dataset_name=key,data=[str(p) for p in files])])
            ids = [str(ident) for task in tasks for file in task.data for ident in pd.read_parquet(file)["id"]]
            assert set(ids) == set(pairs.loc[pairs["cosine_sim_score"] >= 1-eps,"id"])
            removed.extend(ids)
            summary.update(status="official_workflow_complete",n_clusters=clusters)
            summaries.append(summary)
            print("scopes",len(summaries),"/",len(groups),flush=True)
    finally:
        ray.shutdown()
    assert len(removed)==len(set(removed))
    (out / f"removed_eps_{eps}.json").write_text(json.dumps(sorted(removed),indent=2)+"\n")
    report = {"records":len(rows),"scopes":summaries,
              "comparison_boundary_version":SCOPE_VERSION,
              "prepared_text_sha256":sha256_file(root / "prepared/eligible_text.jsonl"),
              "official_workflow_records":sum(s["records"] for s in summaries if s["status"] == "official_workflow_complete"),
              "tiny_scope_retained_records":sum(s["records"] for s in summaries if s["status"] == "tiny_scope_retained"),
              "threshold_removed_counts":{str(eps):len(removed)},
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
    parser.add_argument("--eps",type=float,help="One explicit official cosine-distance threshold for dedup")
    args=parser.parse_args()
    if args.operation == "dedup":
        if args.eps is None or not 0 < args.eps < 1:
            parser.error("dedup requires one --eps between 0 and 1")
        dedup(args.run_root.resolve(),args.smoke,args.eps)
    else:
        encode(args.run_root.resolve(),args.smoke)
