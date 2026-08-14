"""Validate finite RL learning metrics without scanning logged payload/config text."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FLOAT = r"[-+0-9.eE]+"


def _metric_values(text: str, metric: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(rf"['\"]{re.escape(metric)}['\"]:\s*({_FLOAT})", text)
    ]


def _metric_suffix_values(text: str, suffix: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(rf"['\"][^'\"]*{re.escape(suffix)}['\"]:\s*({_FLOAT})", text)
    ]


def _runtime_failure_lines(text: str) -> list[str]:
    """Return genuine runtime failures, excluding commands, configs, and samples.

    SGLang prints all server arguments on one very long line.  Fields such as
    ``enable_nccl_nvls=False`` and ``watchdog_timeout`` made the former broad
    ``NCCL.*(?:error|timeout)`` expression reject successful jobs.
    """

    failures: list[str] = []
    payload_markers = (
        "First rollout sample:",
        "Finish rollout:",
        "ServerArgs(",
        "Running entrypoint for job",
    )
    patterns = (
        re.compile(r"Traceback \(most recent call last\)"),
        re.compile(r"CUDA out of memory", re.I),
        re.compile(r"\bSIGKILL\b", re.I),
        re.compile(r"\bNCCL\s+(?:WARN|ERROR)\b", re.I),
        re.compile(r"\bnccl(?:System|Internal|UnhandledCuda|Remote)Error\b", re.I),
        re.compile(r"\bNCCL\b.{0,240}\b(?:timed out|timeout)\b", re.I),
    )
    for raw_line in text.splitlines():
        line = _ANSI.sub("", raw_line)
        stripped = line.lstrip()
        if stripped.startswith("+") or any(marker in line for marker in payload_markers):
            continue
        if any(pattern.search(line) for pattern in patterns):
            failures.append(line[-1000:])
    return failures


def check_log(
    path: str | Path,
    minimum_updates: int,
    *,
    minimum_nonzero_group_ratio: float = 0.0,
) -> dict[str, object]:
    if not 0.0 <= minimum_nonzero_group_ratio <= 1.0:
        raise ValueError("minimum_nonzero_group_ratio must be within [0, 1]")
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    gradients = _metric_values(text, "train/grad_norm")
    if len(gradients) < minimum_updates:
        raise ValueError(f"only {len(gradients)} grad_norm updates; expected at least {minimum_updates}")
    if not all(math.isfinite(value) for value in gradients) or not any(abs(value) > 1e-12 for value in gradients):
        raise ValueError(f"invalid or all-zero gradients: {gradients}")

    zero_run = 0
    for value in gradients:
        zero_run = zero_run + 1 if abs(value) <= 1e-12 else 0
        if zero_run >= 6:
            raise ValueError(f"six consecutive zero-gradient updates: {gradients}")

    metrics: dict[str, list[float]] = {}
    for metric in ("train/loss", "rollout/raw_reward", "rollout/truncated"):
        values = _metric_values(text, metric)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"missing or non-finite {metric}: {values}")
        metrics[metric] = values

    failures = _runtime_failure_lines(text)
    if failures:
        raise ValueError(f"runtime failure signature found: {failures[0]}")

    nonzero_group_ratios = _metric_suffix_values(text, "/nonzero_std_group_ratio")
    if minimum_nonzero_group_ratio > 0.0:
        if len(nonzero_group_ratios) < minimum_updates:
            raise ValueError(
                f"only {len(nonzero_group_ratios)} reward-variance groups; expected at least {minimum_updates}"
            )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in nonzero_group_ratios):
            raise ValueError(f"invalid nonzero reward-group ratios: {nonzero_group_ratios}")
        observed_ratio = sum(nonzero_group_ratios) / len(nonzero_group_ratios)
        if observed_ratio + 1e-12 < minimum_nonzero_group_ratio:
            raise ValueError(
                f"nonzero reward-group ratio {observed_ratio} is below {minimum_nonzero_group_ratio}: "
                f"{nonzero_group_ratios}"
            )
    else:
        observed_ratio = None

    return {
        "updates": len(gradients),
        "grad_norm": gradients,
        "raw_reward": metrics["rollout/raw_reward"],
        "nonzero_reward_group_ratio": observed_ratio,
        "gate": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("minimum_updates", type=int)
    parser.add_argument("--minimum-nonzero-group-ratio", default=0.0, type=float)
    args = parser.parse_args()
    print(
        check_log(
            args.log,
            args.minimum_updates,
            minimum_nonzero_group_ratio=args.minimum_nonzero_group_ratio,
        )
    )


if __name__ == "__main__":
    main()
