#!/usr/bin/env python3
"""CLI/runner for the Marginal Area-of-Applicability COMPLETION analysis
(src/marginal_aoa_completion.py).

Thin dispatcher only -- no scientific logic lives here. The stage contract,
the label firewall, the pairwise normaliser, the upper-whisker threshold and
every numeric rule are implemented in the module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.marginal_aoa_completion import STAGES, run_analysis


def main(
    experiments: list[str] | None = None,
    from_stage: str = "plan",
    to_stage: str = "compare",
    dry_run: bool = False,
    resume: bool = False,
    allow_earth_engine: bool = False,
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
        allow_earth_engine=allow_earth_engine,
        output_root=Path(output_root) if output_root is not None else None,
        experiments_root=Path(experiments_root) if experiments_root is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marginal AoA completion: importance-weighted predictor-space "
            "dissimilarity (directed), climatic distance (symmetric) and "
            "geographic distance (symmetric), over 12 directed pairs. "
            "target-label-blind, source-model-informed."
        )
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs; defaults to the four canonical AOIs of this frozen analysis.",
    )
    parser.add_argument(
        "--from-stage", default="plan", choices=list(STAGES),
        help=f"First stage to run. Stage order: {list(STAGES)}.",
    )
    parser.add_argument(
        "--to-stage", default="compare", choices=list(STAGES),
        help="Last stage to run (inclusive).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve the plan and prerequisites and print them. Creates no "
            "directory, writes no file, contacts no Earth Engine, fits no "
            "model and computes no distance."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Verify and reuse an existing namespace instead of refusing to overwrite it.",
    )
    parser.add_argument(
        "--allow-earth-engine", action="store_true",
        help=(
            "Authorise LIVE Earth Engine queries and the climate export. "
            "Required by the 'climate-export' stage, which is never started "
            "implicitly."
        ),
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Alternative outputs/ root (dependency injection; default is the canonical root).",
    )
    parser.add_argument(
        "--experiments-root", default=None,
        help="Alternative outputs/experiments/ root (dependency injection).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(
            experiments=args.experiments,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
            resume=args.resume,
            allow_earth_engine=args.allow_earth_engine,
            output_root=args.output_root,
            experiments_root=args.experiments_root,
        ),
        indent=2, default=str,
    ))
