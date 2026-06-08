from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .io_utils import sha256_file, write_json
from .settings import ProjectConfig


def write_repro_manifest(config: ProjectConfig) -> dict[str, object]:
    files = []
    for p in sorted(config.paths.run_dir.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "path": str(p),
                    "size": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.paths.run_dir.name,
        "file_count": len(files),
        "files": files,
    }
    out = config.paths.run_dir / "repro_manifest.json"
    write_json(out, manifest)
    return {"file_count": len(files), "output": str(out)}
