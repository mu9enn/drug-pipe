from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .pipeline import run_all
from .settings import build_config
from .mcp_snapshot import run_snapshot
from .tool_card_builder import build_tool_cards
from .doc_chunker import chunk_skills
from .candidate_generation import generate_candidates
from .pairwise_runner import run_pairwise_adjudication
from .confidence import score_edges
from .graph_views import build_graph_views
from .provenance import build_provenance_sidecar
from .exporters import export_artifacts
from .audit_sampler import sample_for_audit
from .evaluate_logs import evaluate_against_logs
from .manifest import write_repro_manifest
from .question_sampling import sample_questions, sample_simple_questions


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("molclaw-kg")
    p.add_argument("--project-root", default=".")
    p.add_argument("--run-id", default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--api-key", default="")
    p.add_argument("--server-url", default=None)
    p.add_argument("--skills-root", default=None)
    p.add_argument("--logs-root", default=None)
    p.add_argument("--mode", default="claude_cc", choices=["claude_cc"])
    p.add_argument("--max-workers", type=int, default=1, help="Parallel worker count for tool-card/adjudication stages.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run_dir instead of starting from scratch.")
    return p


def main() -> None:
    parser = _base_parser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    subparsers: dict[str, argparse.ArgumentParser] = {}
    for name in [
        "snapshot",
        "tool-cards",
        "doc-chunks",
        "candidates",
        "adjudicate",
        "score",
        "views",
        "provenance",
        "export",
        "audit",
        "eval-logs",
        "manifest",
        "sample-questions",
        "run-all",
    ]:
        subparsers[name] = sub.add_parser(name)

    subparsers["tool-cards"].add_argument(
        "--tool-ids-file",
        default=None,
        help="Optional file path with one tool_id per line for subset rerun.",
    )
    subparsers["tool-cards"].add_argument(
        "--merge-into-existing",
        action="store_true",
        help="When subset rerun, merge updated tool cards into existing full tool_cards.jsonl.",
    )
    subparsers["tool-cards"].add_argument(
        "--rerun-round",
        type=int,
        default=0,
        help="Optional rerun round index for metadata.",
    )
    subparsers["adjudicate"].add_argument(
        "--pair-ids-file",
        default=None,
        help="Optional file path with one pair_id per line for subset adjudication rerun.",
    )
    subparsers["adjudicate"].add_argument(
        "--merge-into-existing",
        action="store_true",
        help="When subset rerun, merge updated adjudications into existing pair_adjudications.jsonl.",
    )
    subparsers["adjudicate"].add_argument(
        "--bypass-cache-for-targets",
        action="store_true",
        help="Bypass existing cache for targeted rerun pairs/groups.",
    )
    subparsers["adjudicate"].add_argument(
        "--rerun-round",
        type=int,
        default=0,
        help="Optional rerun round index for metadata.",
    )
    subparsers["sample-questions"].add_argument("--sample-size", type=int, default=None)
    subparsers["sample-questions"].add_argument("--target-successes", type=int, default=None)
    subparsers["sample-questions"].add_argument("--max-attempts", type=int, default=None)
    subparsers["sample-questions"].add_argument("--json-repair-rounds", type=int, default=1)
    subparsers["sample-questions"].add_argument("--science-kb-topk", type=int, default=3)
    subparsers["sample-questions"].add_argument(
        "--grounding-selection",
        default="random_seeded",
        choices=["random_seeded"],
    )
    subparsers["sample-questions"].add_argument("--max-repeat-target", type=int, default=2)
    subparsers["sample-questions"].add_argument("--max-repeat-compound", type=int, default=2)
    subparsers["sample-questions"].add_argument("--min-hops", type=int, default=2)
    subparsers["sample-questions"].add_argument("--max-hops", type=int, default=4)
    subparsers["sample-questions"].add_argument("--seed", type=int, default=None)
    subparsers["sample-questions"].add_argument(
        "--sampling-mode",
        type=str,
        default="simple_toolchain_question",
        choices=["dag_closure", "linear_debug", "simple_toolchain_question"],
    )
    subparsers["sample-questions"].add_argument(
        "--partial-policy",
        default="closure_required",
        choices=["closure_required", "exclude"],
    )
    subparsers["sample-questions"].add_argument(
        "--edge-profile",
        default="core_strict",
        choices=["core_strict", "core_expanded"],
    )
    subparsers["sample-questions"].add_argument("--max-repair-rounds", type=int, default=2)

    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    if args.cmd == "run-all":
        out = run_all(
            project_root=root,
            run_id=args.run_id,
            api_key=args.api_key,
            server_url=args.server_url,
            skills_root=args.skills_root,
            logs_root=args.logs_root,
            adjudication_mode=args.mode,
            max_workers=max(1, int(args.max_workers or 1)),
            resume=bool(args.resume),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    config = build_config(
        project_root=root,
        run_id=args.run_id,
        server_url=args.server_url,
        api_key=args.api_key,
        skills_root=args.skills_root,
        logs_root=args.logs_root,
        model_name=args.mode,
    )

    if args.cmd == "snapshot":
        out = run_snapshot(config)
    elif args.cmd == "tool-cards":
        out = build_tool_cards(
            config,
            tool_ids_file=(Path(args.tool_ids_file).resolve() if args.tool_ids_file else None),
            merge_into_existing=bool(args.merge_into_existing),
            max_workers=max(1, int(args.max_workers or 1)),
            resume=bool(args.resume),
            rerun_round=int(args.rerun_round or 0),
        )
    elif args.cmd == "doc-chunks":
        out = chunk_skills(config)
    elif args.cmd == "candidates":
        out = generate_candidates(config)
    elif args.cmd == "adjudicate":
        out = run_pairwise_adjudication(
            config,
            mode=args.mode,
            pair_ids_file=(Path(args.pair_ids_file).resolve() if args.pair_ids_file else None),
            merge_into_existing=bool(args.merge_into_existing),
            bypass_cache_for_targets=bool(args.bypass_cache_for_targets),
            max_workers=max(1, int(args.max_workers or 1)),
            resume=bool(args.resume),
            rerun_round=int(args.rerun_round or 0),
        )
    elif args.cmd == "score":
        out = score_edges(config)
    elif args.cmd == "views":
        out = build_graph_views(config)
    elif args.cmd == "provenance":
        out = build_provenance_sidecar(config)
    elif args.cmd == "export":
        out = export_artifacts(config)
    elif args.cmd == "audit":
        out = sample_for_audit(config)
    elif args.cmd == "eval-logs":
        out = evaluate_against_logs(config)
    elif args.cmd == "manifest":
        out = write_repro_manifest(config)
    elif args.cmd == "sample-questions":
        if args.sampling_mode == "simple_toolchain_question":
            if args.target_successes is None or args.max_attempts is None:
                parser.error("simple_toolchain_question requires --target-successes and --max-attempts")
            out = sample_simple_questions(
                config,
                target_successes=int(args.target_successes),
                max_attempts=int(args.max_attempts),
                min_hops=int(args.min_hops),
                max_hops=int(args.max_hops),
                json_repair_rounds=max(0, int(args.json_repair_rounds)),
                science_kb_topk=max(1, int(args.science_kb_topk)),
                grounding_selection=str(args.grounding_selection),
                max_repeat_target=max(1, int(args.max_repeat_target)),
                max_repeat_compound=max(1, int(args.max_repeat_compound)),
                seed=args.seed if args.seed is None else int(args.seed),
            )
        else:
            if args.sample_size is None:
                parser.error("dag_closure/linear_debug require --sample-size")
            out = sample_questions(
                config,
                sample_size=int(args.sample_size),
                min_hops=int(args.min_hops),
                max_hops=int(args.max_hops),
                seed=args.seed if args.seed is None else int(args.seed),
                sampling_mode=str(args.sampling_mode),
                partial_policy=str(args.partial_policy),
                edge_profile=str(args.edge_profile),
                max_repair_rounds=max(0, int(args.max_repair_rounds)),
            )
    else:
        raise ValueError(f"Unknown command {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
