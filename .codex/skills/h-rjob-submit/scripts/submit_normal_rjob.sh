#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    --) break ;;
    --priority|--priority=*|--task-type|--task-type=*)
      echo "normal rjob helper fixes priority at 9; omit --priority and --task-type" >&2
      exit 2
      ;;
  esac
done

exec rjob submit --priority=9 "$@"
