from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_ALIASES = {
    "MOLCLAW_SCP_SERVER_URL": ("MOLCLAW_SCP_SERVER_URL", "MOLCLAW_SCP_MCP_URL"),
    "MOLCLAW_SCP_API_KEY": ("MOLCLAW_SCP_API_KEY", "MOLCLAW_SCP_MCP_AUTH"),
    "MOLCLAW_SCP_AUTH_HEADER": ("MOLCLAW_SCP_AUTH_HEADER", "MOLCLAW_SCP_MCP_AUTH_HEADER"),
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_molclaw_environment(
    env_files: Iterable[str | Path] = (),
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load MolClaw runtime variables without exposing secret values.

    Existing process variables win by default. Drug-Pipe's historical `.env`
    spellings are normalized to the single names consumed by the online runtime.
    """

    loaded: dict[str, str] = {}
    for raw_path in env_files:
        loaded.update(_parse_dotenv(Path(raw_path).expanduser()))

    source = {**loaded, **os.environ} if not override else {**os.environ, **loaded}
    resolved: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        value = next((source.get(name, "").strip() for name in aliases if source.get(name, "").strip()), "")
        if value:
            if override or not os.environ.get(canonical):
                os.environ[canonical] = value
            resolved[canonical] = value

    header = resolved.get("MOLCLAW_SCP_AUTH_HEADER") or os.environ.get("MOLCLAW_SCP_AUTH_HEADER")
    if not header:
        os.environ.setdefault("MOLCLAW_SCP_AUTH_HEADER", "SCP-HUB-API-KEY")
        resolved["MOLCLAW_SCP_AUTH_HEADER"] = "SCP-HUB-API-KEY"
    return resolved


def missing_molclaw_environment() -> list[str]:
    return [
        key
        for key in ("MOLCLAW_SCP_SERVER_URL", "MOLCLAW_SCP_API_KEY")
        if not os.environ.get(key, "").strip()
    ]


def redacted_environment_summary() -> dict[str, object]:
    return {
        "server_url_configured": bool(os.environ.get("MOLCLAW_SCP_SERVER_URL", "").strip()),
        "api_key_configured": bool(os.environ.get("MOLCLAW_SCP_API_KEY", "").strip()),
        "auth_header": os.environ.get("MOLCLAW_SCP_AUTH_HEADER", "SCP-HUB-API-KEY"),
        "proxy_configured": bool(os.environ.get("MOLCLAW_PROXY_URL", "").strip()),
    }
