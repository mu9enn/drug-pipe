from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from .io_utils import write_json, write_jsonl
from .models import DocChunk
from .settings import ProjectConfig


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _resolve_skills_base(skills_root: Path) -> Path:
    direct = skills_root / "L1_tools"
    nested = skills_root / ".claude" / "skills" / "L1_tools"
    if direct.exists():
        return skills_root
    if nested.exists():
        return skills_root / ".claude" / "skills"
    raise FileNotFoundError(
        f"skills root must contain L1/L2/L3 directories either directly or under .claude/skills: {skills_root}"
    )


def _skill_level(path: Path, skills_base: Path) -> str:
    rel = path.relative_to(skills_base)
    parts = rel.parts
    if parts and parts[0] == "L1_tools":
        return "L1"
    if parts and parts[0] == "L2_workflows":
        return "L2"
    if parts and parts[0] == "L3_methodology":
        return "L3"
    return "UNK"


def _iter_docs(skills_base: Path) -> Iterator[Path]:
    for p in sorted((skills_base / "L1_tools").glob("*/SKILL.md")):
        yield p
    for p in sorted((skills_base / "L2_workflows").glob("*.md")):
        yield p
    for p in sorted((skills_base / "L3_methodology").glob("*.md")):
        yield p


def _make_chunk_id(doc_id: str, heading_path: list[str], idx: int) -> str:
    raw = f"{doc_id}|{'/'.join(heading_path)}|{idx}".encode("utf-8")
    return "chunk::" + hashlib.md5(raw).hexdigest()[:16]


def _block_type(text: str) -> str:
    t = text.strip()
    if not t:
        return "paragraph"
    has_code = "```" in t
    has_list = any(line.strip().startswith(("- ", "* ", "1.", "2.", "3.")) for line in t.splitlines())
    if has_code and has_list:
        return "mixed"
    if has_code:
        return "code"
    if has_list:
        return "list"
    return "paragraph"


def chunk_skills(config: ProjectConfig) -> dict[str, int]:
    skills_root = config.runtime.skills_root
    skills_base = _resolve_skills_base(skills_root)
    chunks: list[DocChunk] = []

    for doc_path in _iter_docs(skills_base):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = f"doc::{doc_path.relative_to(skills_base)}"
        level = _skill_level(doc_path, skills_base)

        lines = text.splitlines(keepends=True)
        heading_path: list[str] = [doc_path.stem]
        cur_block: list[str] = []
        cur_start = 0
        offset = 0
        section_idx = 0

        def flush_block(end_offset: int):
            nonlocal cur_block, cur_start, section_idx
            if not cur_block:
                return
            raw = "".join(cur_block).strip()
            if not raw:
                cur_block = []
                return
            chunk = DocChunk(
                doc_id=doc_id,
                path=str(doc_path),
                skill_level=level,
                section_id=f"{doc_id}::sec{section_idx:04d}",
                heading_path=list(heading_path),
                block_type=_block_type(raw),
                chunk_id=_make_chunk_id(doc_id, heading_path, section_idx),
                char_start=cur_start,
                char_end=end_offset,
                text=raw,
            )
            chunks.append(chunk)
            section_idx += 1
            cur_block = []

        for line in lines:
            hm = HEADING_RE.match(line.strip("\n"))
            if hm:
                flush_block(offset)
                level_num = len(hm.group(1))
                title = hm.group(2).strip()
                if len(heading_path) >= level_num:
                    heading_path = heading_path[:level_num - 1]
                heading_path.append(title)
                cur_start = offset + len(line)
            else:
                if not cur_block:
                    cur_start = offset
                cur_block.append(line)
            offset += len(line)

        flush_block(offset)

    out = config.paths.run_dir / "doc_chunks.jsonl"
    write_jsonl(out, [c.model_dump() for c in chunks])
    write_json(
        config.paths.run_dir / "doc_chunks_meta.json",
        {
            "chunk_count": len(chunks),
            "doc_count": len({c.doc_id for c in chunks}),
            "path": str(out),
        },
    )
    return {"chunk_count": len(chunks), "doc_count": len({c.doc_id for c in chunks})}
