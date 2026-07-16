from __future__ import annotations

import argparse
import os

MOLCLAW_ENV_KEYS = (
    "MOLCLAW_SCP_SERVER_URL",
    "MOLCLAW_SCP_API_KEY",
    "MOLCLAW_CONNECT_TIMEOUT_SEC",
    "MOLCLAW_LIST_TOOLS_TIMEOUT_SEC",
    "MOLCLAW_TOOL_TIMEOUT_SEC",
    "MOLCLAW_TOOL_HEARTBEAT_SEC",
)


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def assert_tool_environment_allowed(operation: str = "tool environment access") -> None:
    """Fail closed unless an explicitly online eval/debug process opts in."""
    if _enabled("DRUG_AGENT_TRAINING_OFFLINE", default=False):
        raise RuntimeError(
            f"{operation} is forbidden while DRUG_AGENT_TRAINING_OFFLINE=1. "
            "Training uses fixed offline decision states and must not execute actions."
        )
    if not _enabled("DRUG_AGENT_ALLOW_TOOL_ENV", default=False):
        raise RuntimeError(
            f"{operation} is disabled by default. Only an explicitly named online evaluation/debug "
            "entry may set DRUG_AGENT_ALLOW_TOOL_ENV=1."
        )


def assert_offline_training_environment() -> None:
    if not _enabled("DRUG_AGENT_TRAINING_OFFLINE", default=False):
        raise RuntimeError("Formal offline training must set DRUG_AGENT_TRAINING_OFFLINE=1")
    if _enabled("DRUG_AGENT_ALLOW_TOOL_ENV", default=False):
        raise RuntimeError("Offline training cannot set DRUG_AGENT_ALLOW_TOOL_ENV=1")
    leaked = [key for key in MOLCLAW_ENV_KEYS if os.environ.get(key)]
    if leaked:
        raise RuntimeError(f"Offline training inherited forbidden MolClaw environment variables: {leaked}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Drug Agent offline-training/tool-environment guard")
    parser.add_argument("--check-offline-training", action="store_true")
    parser.add_argument("--check-tool-access", action="store_true")
    args = parser.parse_args()
    if args.check_offline_training:
        assert_offline_training_environment()
    if args.check_tool_access:
        assert_tool_environment_allowed()
    print("drug_agent offline guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
