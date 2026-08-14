#!/usr/bin/env bash

# Resolve slime_env.sh from the checkout layout before trying legacy absolute
# paths.  This file is sourced by launchers, so keep it free of shell options
# and side effects other than exporting SLIME_ENV.
_SLIME_ENV_RESOLVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLIME_ENV_CANDIDATES=(
  "${SLIME_ENV:-}"
  "$_SLIME_ENV_RESOLVER_DIR/../../../slime_env/slime_env.sh"
  "/root/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime_env/slime_env.sh"
  "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/drug-pipe/slime-wd/slime_env/slime_env.sh"
  "/root/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh"
  "/home/sunxiangyu/slime_sxy/group-space/sunxiangyu/slime_env/slime_env.sh"
)

SLIME_ENV=""
for _slime_env_candidate in "${_SLIME_ENV_CANDIDATES[@]}"; do
  if [[ -n "$_slime_env_candidate" && -f "$_slime_env_candidate" ]]; then
    SLIME_ENV="$(cd "$(dirname "$_slime_env_candidate")" && pwd)/$(basename "$_slime_env_candidate")"
    break
  fi
done

if [[ -z "$SLIME_ENV" ]]; then
  echo "SLIME environment file not found; checked repository-relative and cluster paths" >&2
  return 2 2>/dev/null || exit 2
fi
export SLIME_ENV

unset _slime_env_candidate _SLIME_ENV_CANDIDATES _SLIME_ENV_RESOLVER_DIR
