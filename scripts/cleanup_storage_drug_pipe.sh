#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DEFAULT="/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe"
ROOT="${DRUG_PIPE_ROOT:-$ROOT_DEFAULT}"
APPLY=0; CONFIRM=0; KEEP=0; GATES=0; DEBUG_DAYS=0; PYCACHE=0
usage(){ echo "Usage: $0 [--dry-run] [--apply --confirm] [--remove-pycache] [--prune-toolrl N] [--remove-gates] [--remove-debug-days N]"; }
while (($#)); do
  case "$1" in
    --dry-run) APPLY=0 ;;
    --apply) APPLY=1 ;;
    --confirm) CONFIRM=1 ;;
    --remove-pycache) PYCACHE=1 ;;
    --prune-toolrl) KEEP="$2"; shift; [[ "$KEEP" =~ ^[0-9]+$ ]] || exit 2 ;;
    --remove-gates) GATES=1 ;;
    --remove-debug-days) DEBUG_DAYS="$2"; shift; [[ "$DEBUG_DAYS" =~ ^[0-9]+$ ]] || exit 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
[[ "$ROOT" == "$ROOT_DEFAULT" || "$ROOT" == "$ROOT_DEFAULT"/* ]] || { echo "refusing root $ROOT" >&2; exit 3; }
[[ -d "$ROOT" ]] || { echo "missing root $ROOT" >&2; exit 3; }
(( ! APPLY || CONFIRM )) || { echo "--apply requires --confirm" >&2; exit 4; }
size(){ du -sh -- "$1" 2>/dev/null | awk '{print $1}' || echo '?'; }
remove(){ local p="$1"; printf '%s\t%s\n' "$(size "$p")" "$p"; if (( APPLY )); then [[ ! -L "$p" && "$p" == "$ROOT"/* ]] || exit 5; rm -rf -- "$p"; fi; }
echo "Root=$ROOT Mode=$([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"; df -h "$ROOT" | tail -1
if (( PYCACHE )); then
  find "$ROOT" -xdev \( -type d -name __pycache__ -o -type f -name '*.pyc' \) -print0 | while IFS= read -r -d '' p; do remove "$p"; done
fi
if (( KEEP > 0 )); then
  for run in "$ROOT"/slime-wd/outputs/slime_drug_agent_runs/*production_*/toolrl; do
    [[ -d "$run" ]] || continue
    mapfile -t dirs < <(for d in "$run"/iter_*; do [[ -d "$d" ]] || continue; b=$(find "$d" -xdev -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'); (( b > 100000000000 )) && echo "$d"; done | sort -V)
    for ((i=0; i<${#dirs[@]}-KEEP; i++)); do (( i >= 0 )) && remove "${dirs[$i]}"; done
  done
fi
if (( GATES )); then
  for p in "$ROOT"/slime-wd/outputs/slime_drug_agent_runs/*production_*/gates; do [[ -d "$p" ]] && remove "$p"; done
fi
if (( DEBUG_DAYS > 0 )); then
  find "$ROOT/.runtime" -xdev -type d -path '*/llm_clean_chunks/*/debug' -mtime "+$DEBUG_DAYS" -print0 2>/dev/null | while IFS= read -r -d '' p; do remove "$p"; done
fi
echo "No deletion unless --apply --confirm are both supplied."
