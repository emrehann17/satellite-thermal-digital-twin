#!/usr/bin/env python3
"""CLI/runner for the few-shot recovery curve analysis
(src/few_shot_recovery.py).

Thin dispatcher only -- no scientific logic lives here. The stage contract, the
block tiering, the nested selection ordering, the fit registry, the recovery
arithmetic and every numeric rule are implemented in the module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.few_shot_recovery import STAGES, run_analysis


def main(
    experiments: list[str] | None = None,
    from_stage: str = "plan",
    to_stage: str = "summarize",
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    output_root: str | Path | None = None,
    experiments_root: str | Path | None = None,
) -> dict:
    # `output_root` / `experiments_root` are programmatic dependency-injection
    # points (tests, alternative namespaces); None means the canonical roots.
    return run_analysis(
        experiments=experiments,
        from_stage=from_stage,
        to_stage=to_stage,
        dry_run=dry_run,
        resume=resume,
        force=force,
        output_root=Path(output_root) if output_root is not None else None,
        experiments_root=Path(experiments_root) if experiments_root is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot recovery curve: how much of the gap between zero-shot raw "
            "transfer and the target-only within-region ceiling is recovered when k "
            "labeled target spatial blocks are supplied, over 6 directed pairs. "
            "Target labels ARE used -- this is a supervised adaptation sensitivity "
            "analysis, not an operational deployment claim, not active learning and "
            "not a causal decomposition."
        )
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs; defaults to the three canonical AOIs of this frozen analysis.",
    )
    parser.add_argument(
        "--from-stage", default="plan", choices=list(STAGES),
        help=f"First stage to run. Stage order: {list(STAGES)}.",
    )
    parser.add_argument(
        "--to-stage", default="summarize", choices=list(STAGES),
        help="Last stage to run (inclusive).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the exact plan, the resolved input hashes and every planned output "
            "path. Creates no directory, writes no file and fits no model."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Verify and reuse complete, hash-bound stages and direction partitions "
            "instead of refusing to overwrite an existing namespace. A partially "
            "written direction is never accepted."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Quarantine an existing namespace under _quarantine/ and start fresh. Never deletes.",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Alternative outputs/ root (dependency injection).",
    )
    parser.add_argument(
        "--experiments-root", default=None,
        help="Alternative outputs/experiments/ root (dependency injection).",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = main(
        experiments=args.experiments,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
        output_root=args.output_root,
        experiments_root=args.experiments_root,
    )
    print(json.dumps(result, indent=2, default=str))
