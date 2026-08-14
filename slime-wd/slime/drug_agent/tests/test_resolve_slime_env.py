from __future__ import annotations

import subprocess
from pathlib import Path


RESOLVER = Path(__file__).parents[1] / "scripts" / "resolve_slime_env.sh"


def test_resolves_repo_adjacent_slime_env(tmp_path: Path) -> None:
    checkout = tmp_path / "slime-wd"
    resolver_dir = checkout / "slime" / "drug_agent" / "scripts"
    resolver_dir.mkdir(parents=True)
    copied_resolver = resolver_dir / RESOLVER.name
    copied_resolver.write_text(RESOLVER.read_text(encoding="utf-8"), encoding="utf-8")

    env_file = checkout / "slime_env" / "slime_env.sh"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("export SLIME_ENV_TEST_SENTINEL=resolved\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'unset SLIME_ENV; source "$1"; source "$SLIME_ENV"; '
            'printf "%s\\n%s\\n" "$SLIME_ENV" "$SLIME_ENV_TEST_SENTINEL"',
            "_",
            str(copied_resolver),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [str(env_file), "resolved"]
