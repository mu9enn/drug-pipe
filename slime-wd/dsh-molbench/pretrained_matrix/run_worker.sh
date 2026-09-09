#!/usr/bin/env bash
set -Eeuo pipefail
# User scope: SFT evaluation is L1-only. Also applies to already-running serial parents.
if [[ "${MODEL_ID:-}" == qwen3.5-9b-sft-local && "${WORKSPACE_VARIANT:-}" != l1-flat ]]; then
  echo "Skipping SFT hierarchy evaluation: cancelled by user."
  exit 0
fi
umask 002

: "${MODEL_DIR:?}" "${MODEL_ID:?}" "${MODEL_NAME:?}" "${GPU_COUNT:?}"
: "${TP_SIZE:?}" "${WORKSPACE_VARIANT:?}" "${RUN_NAME:?}" "${INFRA_NAME:?}"
: "${LIMIT_PER_SUITE:=0}"
: "${SUITES:=ms1 ms2}"
: "${EVAL_SEED:=42}"
read -r -a selected_suites <<< "$SUITES"
suite_args=()
for suite in "${selected_suites[@]}"; do suite_args+=(--suite "$suite"); done

shared_user_root=/root/slime_sxy/group-space/sunxiangyu
worker_root=$shared_user_root/drug-pipe/slime-wd
project_root=$shared_user_root/drug-pipe
matrix_root=$worker_root/dsh-molbench/pretrained_matrix
infra_dir=$worker_root/outputs/$INFRA_NAME
run_dir=$worker_root/outputs/dsh_molbench_evals/$RUN_NAME
dsh_dir=$worker_root/deepseek-harness
runtime_dir=$worker_root/outputs/dsh_eval_runtime
agent_preset=molclaw-v8-eval
system_prompt_file=$project_root/data-pipe/pipeline/cleaning/prompts/qwen35_system.md

status_log=$infra_dir/status.log
sglang_log=$infra_dir/sglang.log
dsh_log=$infra_dir/dsh.log
rollout_log=$infra_dir/rollout.log
sglang_pid=
dsh_pid=

mkdir -p "$infra_dir" "$run_dir"
log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$status_log"; }
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for process_pid in "${dsh_pid:-}" "${sglang_pid:-}"; do
    if [[ -n "$process_pid" ]] && kill -0 "$process_pid" 2>/dev/null; then
      kill "$process_pid" 2>/dev/null || true
      wait "$process_pid" 2>/dev/null || true
    fi
  done
  chmod -R a+rwX "$infra_dir" "$run_dir" 2>/dev/null || true
  printf '%s\n' "$rc" > "$infra_dir/worker.exit"
  log "worker_exit=$rc"
  exit "$rc"
}
trap cleanup EXIT INT TERM

wait_http() {
  local url=$1 process_pid=$2 attempts=$3 label=$4 attempt
  for ((attempt=1; attempt<=attempts; attempt++)); do
    kill -0 "$process_pid" 2>/dev/null || { log "$label exited before readiness"; return 1; }
    if /usr/bin/python3 - "$url" <<'PY' >/dev/null 2>&1
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    assert response.status == 200
PY
    then return 0; fi
    sleep 5
  done
  log "$label readiness timed out"
  return 1
}

for required in \
  "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors.index.json" \
  "$MODEL_DIR/tokenizer_config.json" "$dsh_dir/apps/cli/src/bin.ts" \
  "$worker_root/dsh-molbench/run_dsh_molbench.py" \
  "$worker_root/dsh-molbench/install_eval_preset.py" \
  "$worker_root/dsh-molbench/audit_protocol.py" \
  "$matrix_root/validate_tokenizer_protocol.py" \
  "$system_prompt_file" \
  "$runtime_dir/node-v24.19.0/bin/node" \
  "$matrix_root/settings.template.yaml" "$matrix_root/molclaw.cordis.patch.yml"; do
  [[ -s "$required" ]] || { log "missing_required=$required"; exit 1; }
done

/usr/bin/python3 - "$MODEL_DIR" <<'PY' | tee "$infra_dir/model_validation.json"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map") or {}
assert weight_map, "empty safetensors weight map"
shards = sorted(set(weight_map.values()))
for shard in shards:
    path = root / shard
    assert path.is_file() and path.stat().st_size > 0, f"missing or empty shard: {shard}"
config = json.loads((root / "config.json").read_text())
assert str(config.get("model_type", "")).startswith("qwen3_5"), config.get("model_type")
print(json.dumps({"weights": len(weight_map), "shards": len(shards), "model_type": config["model_type"]}))
PY

if [[ "$MODEL_ID" == qwen3.5-9b-sft-local ]]; then
  training_root=$(dirname "$MODEL_DIR")
  [[ -f "$training_root/training.complete" && -s "$training_root/probe_reload_validation.json" ]]
  expected_steps=$(/usr/bin/python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["count"]//m["global_batch_size"])' "$training_root/resolved_config.json")
  /usr/bin/python3 "$worker_root/experiments/aligned_sft/verify_metrics.py" "$training_root/train_metrics" "$expected_steps"
fi

actual_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$actual_gpus" -eq "$GPU_COUNT" ]] || { log "gpu_count expected=$GPU_COUNT actual=$actual_gpus"; exit 1; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc H200 || true)" -eq 0 ]]
log "pipeline_start host=$(hostname) model=$MODEL_DIR variant=$WORKSPACE_VARIANT"
log "gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
/usr/bin/python3 "$worker_root/experiments/aligned_sft/capture_runtime.py" \
  --model "$MODEL_DIR" --output "$infra_dir/runtime_snapshot.json"

case "$WORKSPACE_VARIANT" in
  l1-flat)
    skill_source=$project_root/workdir-skills/molclaw-l1-workspace
    expected_skill_count=52
    ;;
  legacy-hierarchy)
    legacy_source=$project_root/workdir-skills/molclaw-trajectory-execution
    flat_source=$project_root/workdir-skills/molclaw-l1-workspace
    skill_source=$infra_dir/workspace_template
    [[ ! -e "$skill_source" ]] || { log "existing_workspace_template=$skill_source"; exit 1; }
    diff -qr "$flat_source/.agents/skills" "$legacy_source/.claude/skills/L1_tools" >/dev/null
    mkdir -p "$skill_source/.agents"
    cp -a "$legacy_source/.claude/skills" "$skill_source/.agents/skills"
    cp "$flat_source/prompt_prefix.md" "$skill_source/prompt_prefix.md"
    while IFS= read -r -d '' path; do
      sed -i 's#\.claude/skills#\.agents/skills#g' "$path"
    done < <(find "$skill_source" -type f \( -name '*.md' -o -name '*.json' -o -name '*.sh' -o -name '*.py' \) -print0)
    skill_root=$skill_source/.agents/skills
    for path in "$skill_root/L1_tools"/*; do
      name=$(basename "$path")
      [[ ! -e "$skill_root/$name" && ! -L "$skill_root/$name" ]]
      ln -s "L1_tools/$name" "$skill_root/$name"
    done
    for tier in L2_workflows L3_methodology; do
      for path in "$skill_root/$tier"/*.md; do
        name=$(basename "$path")
        [[ ! -e "$skill_root/$name" && ! -L "$skill_root/$name" ]]
        ln -s "$tier/$name" "$skill_root/$name"
      done
    done
    expected_skill_count=68
    [[ "$(find "$skill_root" -mindepth 1 -maxdepth 1 -type l | wc -l)" -eq "$expected_skill_count" ]]
    ;;
  *) log "unknown_workspace_variant=$WORKSPACE_VARIANT"; exit 1 ;;
esac
[[ -s "$skill_source/prompt_prefix.md" && -d "$skill_source/.agents/skills" ]]

export PATH="$runtime_dir/node-v24.19.0/bin:$PATH"
export PYTHONPATH="$runtime_dir/python${PYTHONPATH:+:$PYTHONPATH}"
export SLIME_LOCAL_API_KEY=local NODE_USE_ENV_PROXY=1
export HTTP_PROXY=http://127.0.0.1:13208
export HTTPS_PROXY=$HTTP_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY

mkdir -p /root/.dsh
sed -e "s/__MODEL_ID__/$MODEL_ID/g" -e "s/__MODEL_NAME__/$MODEL_NAME/g" \
  "$matrix_root/settings.template.yaml" > /root/.dsh/settings.yaml
install -m 600 "$matrix_root/molclaw.cordis.patch.yml" /root/.dsh/cordis.patch.yml

log waiting_for_molclaw_secret
for attempt in $(seq 1 720); do
  [[ -s /root/.dsh/molclaw.env ]] && break
  [[ "$attempt" != 720 ]] || { log molclaw_secret_timeout; exit 1; }
  sleep 5
done
source /root/.dsh/molclaw.env
[[ -n "${MOLCLAW_SCP_API_KEY:-}" ]] || { log MOLCLAW_SCP_API_KEY_is_not_set; exit 1; }

log waiting_for_mcp_reverse_tunnel
for attempt in $(seq 1 720); do
  if /usr/bin/python3 - <<'PY' >/dev/null 2>&1
import socket
with socket.create_connection(("127.0.0.1", 13208), timeout=2): pass
PY
  then break; fi
  [[ "$attempt" != 720 ]] || { log mcp_reverse_tunnel_timeout; exit 1; }
  sleep 5
done

source "$shared_user_root/slime_env/slime_env.sh"
visible_devices=$(seq -s, 0 $((GPU_COUNT - 1)))
python "$matrix_root/validate_tokenizer_protocol.py" \
  --model-dir "$MODEL_DIR" --output "$infra_dir/tokenizer_protocol.json"
log "sglang_start tp=$TP_SIZE context=262144"
CUDA_VISIBLE_DEVICES="$visible_devices" python -u -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port 30000 --tp-size "$TP_SIZE" --context-length 262144 \
  --mem-fraction-static 0.70 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --preferred-sampling-params '{"temperature":0}' --random-seed "$EVAL_SEED" \
  --disable-cuda-graph --disable-custom-all-reduce > "$sglang_log" 2>&1 &
sglang_pid=$!
wait_http http://127.0.0.1:30000/v1/models "$sglang_pid" 720 sglang
log "sglang_ready pid=$sglang_pid"

/usr/bin/python3 - "$MODEL_ID" "$infra_dir" "$GPU_COUNT" "$TP_SIZE" "$EVAL_SEED" <<'PY'
import json, pathlib, sys, urllib.request
model_id, out = sys.argv[1], pathlib.Path(sys.argv[2])
def request(path, payload=None):
    req = urllib.request.Request("http://127.0.0.1:30000" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=600) as response: return json.loads(response.read())
models = request("/v1/models")
assert model_id in {item.get("id") for item in models.get("data", [])}
model_info = request("/get_model_info")
preferred = model_info.get("preferred_sampling_params")
if isinstance(preferred, str): preferred = json.loads(preferred)
assert isinstance(preferred, dict) and preferred.get("temperature") == 0, model_info
tool = request("/v1/chat/completions", {"model": model_id,
    "messages": [{"role": "user", "content": "Call get_weather for Shanghai now. You must use the tool."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather by city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "temperature": 0})
calls = tool.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
assert calls and calls[0].get("function", {}).get("name") == "get_weather", tool
json.loads(calls[0]["function"]["arguments"])
(out / "models_test.json").write_text(json.dumps(models, indent=2))
(out / "model_info.json").write_text(json.dumps(model_info, indent=2))
(out / "tool_test.json").write_text(json.dumps(tool, indent=2))
import ast
argument_line = next(line for line in (out / 'sglang.log').read_text().splitlines() if 'ServerArgs(' in line)
call = ast.parse(argument_line[argument_line.index('ServerArgs('):], mode='eval').body
actual_args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
assert actual_args['tp_size'] == int(sys.argv[4])
assert actual_args['random_seed'] == int(sys.argv[5])
(out / 'actual_server_args.json').write_text(json.dumps(actual_args, indent=2))
(out / "runtime_audit.json").write_text(json.dumps({
    "actual_server_args": actual_args,
    "runtime_snapshot": json.loads((out / "runtime_snapshot.json").read_text()),
    "server_args": {
        "gpu_count": int(sys.argv[3]), "tp_size": int(sys.argv[4]), "random_seed": int(sys.argv[5]),
        "context_length": 262144, "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder", "preferred_sampling_params": {"temperature": 0},
        "disable_cuda_graph": True, "disable_custom_all_reduce": True,
    },
    "server_model_info": model_info,
    "effective_temperature": 0,
    "temperature_source": "sglang_preferred_sampling_params",
}, indent=2))
PY
log sglang_acceptance_complete

cd "$dsh_dir"
for bootstrap_attempt in 0 1 2; do
  node --import tsx/esm apps/cli/src/bin.ts web --no-open > "$infra_dir/dsh-start-${bootstrap_attempt}.log" 2>&1 &
  dsh_pid=$!
  if wait_http http://127.0.0.1:3080 "$dsh_pid" 120 dsh; then
    ln -sf "dsh-start-${bootstrap_attempt}.log" "$dsh_log"
    printf '%s\n' "$bootstrap_attempt" > "$infra_dir/bootstrap_infrastructure_retries.txt"
    break
  fi
  kill "$dsh_pid" 2>/dev/null || true
  wait "$dsh_pid" 2>/dev/null || true
  [[ "$bootstrap_attempt" != 2 ]] || exit 1
  # Retry only transport failure before any question has been submitted.
  rg -q 'fetch failed|ECONNRESET|ETIMEDOUT|Request was cancelled' "$infra_dir/dsh-start-${bootstrap_attempt}.log" || exit 1
  sleep 30
done
log "dsh_ready pid=$dsh_pid"

cd "$worker_root"
/usr/bin/python3 -u dsh-molbench/run_dsh_molbench.py "${suite_args[@]}" \
  --rollout-only --run-dir "$run_dir" --task-timeout-sec 14400 --max-workers 2 \
  --model-provider slime-local --model-id "$MODEL_ID" --agent-preset "$agent_preset" \
  --skill-source "$skill_source" --limit-per-suite "$LIMIT_PER_SUITE" \
  --required-mcp-tools 81 --required-skill-count "$expected_skill_count" \
  --system-prompt-file "$system_prompt_file" --tokenizer-config "$MODEL_DIR/tokenizer_config.json" \
  --runtime-audit-file "$infra_dir/runtime_audit.json" 2>&1 | tee "$rollout_log"

/usr/bin/python3 dsh-molbench/audit_protocol.py --run-dir "$run_dir" \
  --system-prompt-file "$system_prompt_file" --expected-mcp-tools 81 \
  --expected-skill-count "$expected_skill_count" \
  | tee "$infra_dir/protocol_audit.log"

for retry in 1 2; do
  /usr/bin/python3 -u dsh-molbench/run_dsh_molbench.py "${suite_args[@]}" \
    --rollout-only --resume --retry-infra-failed --run-dir "$run_dir" \
    --task-timeout-sec 14400 --max-workers 2 --model-provider slime-local \
    --model-id "$MODEL_ID" --agent-preset "$agent_preset" --skill-source "$skill_source" \
    --limit-per-suite "$LIMIT_PER_SUITE" --required-mcp-tools 81 \
    --required-skill-count "$expected_skill_count" --system-prompt-file "$system_prompt_file" \
    --tokenizer-config "$MODEL_DIR/tokenizer_config.json" \
    --runtime-audit-file "$infra_dir/runtime_audit.json" \
    2>&1 | tee -a "$rollout_log"
done

/usr/bin/python3 - "$run_dir" "$WORKSPACE_VARIANT" "$LIMIT_PER_SUITE" <<'PY' | tee "$infra_dir/rollout_validation.json"
import json, pathlib, sys
run, variant, limit = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = json.loads((run / "run_manifest.json").read_text())
assert manifest["agent_preset"] == "molclaw-v8-eval"
assert manifest["evaluation_contract"]["native_thinking"] is True
assert manifest["evaluation_contract"]["temperature"] == 0
assert "top_p" not in manifest["evaluation_contract"]
assert manifest["startup_protocol_probe"]["passed"] is True
assert manifest["startup_protocol_probes"]
assert all(probe["passed"] is True for probe in manifest["startup_protocol_probes"])
assert manifest["actual_first_request"]["passed"] is True
records = sorted(run.glob("results/*/record.json"))
workspaces = sorted(run.glob("workspaces/*"))
expected = manifest["sample_count"]
assert expected == len(manifest["sample_ids"])
assert len(records) == len(workspaces) == expected
assert all((path / ".agents/skills").is_dir() for path in workspaces)
assert all(not (path / ".dsh/skills").exists() for path in workspaces)
if variant == "legacy-hierarchy":
    assert all((path / ".agents/skills/L1_tools").is_dir() for path in workspaces)
summary = json.loads((run / "rollout_summary.json").read_text())
assert summary["publishable"] is True, summary
print(json.dumps({"records": len(records), "workspaces": len(workspaces), "preset": manifest["agent_preset"], "summary": summary}))
PY

touch "$infra_dir/rollout.complete"
log PIPELINE_COMPLETE
