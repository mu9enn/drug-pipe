"""CLI used by the Claude gate to clean invocation-scoped descendants."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


INVOCATION_ID_ENV = "DRUG_PIPE_CLAUDE_INVOCATION_ID"
DESCENDANT_TERMINATION_GRACE_SEC = 10.0


def _invocation_processes(
    invocation_id: str,
    *,
    exclude_pids: set[int] | None = None,
) -> list[int]:
    expected = f"{INVOCATION_ID_ENV}={invocation_id}".encode("utf-8")
    excluded = exclude_pids or set()
    matches: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid() or pid in excluded:
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if expected in environment:
            matches.append(pid)
    return sorted(matches)


def _signal_invocation_processes(pids: list[int], sig: signal.Signals) -> int:
    own_group = os.getpgrp()
    groups: set[int] = set()
    individual_pids: set[int] = set()
    for pid in pids:
        try:
            process_group = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            continue
        if process_group > 0 and process_group != own_group:
            groups.add(process_group)
        else:
            individual_pids.add(pid)

    signalled = 0
    for process_group in groups:
        try:
            os.killpg(process_group, sig)
            signalled += 1
        except (ProcessLookupError, PermissionError):
            continue
    for pid in individual_pids:
        try:
            os.kill(pid, sig)
            signalled += 1
        except (ProcessLookupError, PermissionError):
            continue
    return signalled


def _wait_for_invocation_processes(
    invocation_id: str,
    timeout_sec: float,
    *,
    exclude_pids: set[int] | None = None,
) -> list[int]:
    deadline = time.monotonic() + timeout_sec
    while True:
        remaining = _invocation_processes(invocation_id, exclude_pids=exclude_pids)
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.05)


def _cleanup_detached_descendants(
    invocation_id: str,
    *,
    grace_sec: float | None = None,
    exclude_pids: set[int] | None = None,
) -> dict[str, object]:
    grace_sec = DESCENDANT_TERMINATION_GRACE_SEC if grace_sec is None else grace_sec
    candidates = _invocation_processes(invocation_id, exclude_pids=exclude_pids)
    term_targets = 0
    kill_targets = 0
    forced_kill = False
    if candidates:
        term_targets = _signal_invocation_processes(candidates, signal.SIGTERM)
    remaining = _wait_for_invocation_processes(
        invocation_id, grace_sec, exclude_pids=exclude_pids
    )
    if remaining:
        forced_kill = True
        kill_targets = _signal_invocation_processes(remaining, signal.SIGKILL)
        remaining = _wait_for_invocation_processes(
            invocation_id, 5.0, exclude_pids=exclude_pids
        )
    return {
        "detached_descendant_candidate_pids": candidates,
        "detached_descendant_term_targets": term_targets,
        "detached_descendant_forced_kill": forced_kill,
        "detached_descendant_kill_targets": kill_targets,
        "detached_descendant_remaining_pids": remaining,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--exclude-pid", action="append", type=int, default=[])
    parser.add_argument("--grace-sec", type=float, default=None)
    args = parser.parse_args()
    result = _cleanup_detached_descendants(
        args.invocation_id,
        grace_sec=args.grace_sec,
        exclude_pids=set(args.exclude_pid),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
