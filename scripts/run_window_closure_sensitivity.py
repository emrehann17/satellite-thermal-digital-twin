#!/usr/bin/env python3
"""CLI/runner for the window-closure sensitivity analysis
(src/window_closure_sensitivity.py).

Thin dispatcher only -- no scientific logic. Window arithmetic, the censoring
contract, the common-cohort rule, the model/fold/bootstrap configuration and
every interval classification live in the module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.window_closure_sensitivity import DEFAULT_SHIFTS, STAGES, run_analysis


def main(
    experiment_id: str,
    shifts: list[int] | None = None,
    from_stage: str = "plan",
    to_stage: str = "compare",
    dry_run: bool = False,
    force: bool = False,
    resume: bool = False,
    output_root: str | Path | None = None,
    experiments_root: str | Path | None = None,
) -> dict:
    # `output_root` / `experiments_root` are programmatic dependency-injection
    # points (tests, alternative namespaces); None means the canonical roots.
    return run_analysis(
        experiment_id=experiment_id,
        shifts=tuple(shifts) if shifts is not None else DEFAULT_SHIFTS,
        from_stage=from_stage,
        to_stage=to_stage,
        dry_run=dry_run,
        force=force,
        resume=resume,
        output_root=Path(output_root) if output_root is not None else None,
        experiments_root=Path(experiments_root) if experiments_root is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Window-closure sensitivity: shift the canonical predictor window "
            "earlier by fixed day counts while preserving its length, and "
            "measure how within-AOI baseline/thermal results and the thermal "
            "contribution respond. The label window never moves."
        )
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID (core/regions.py registry).")
    parser.add_argument(
        "--shifts", nargs="+", type=int, default=list(DEFAULT_SHIFTS),
        help="Preregistered closure shifts in days (default: 0 7 14).",
    )
    parser.add_argument("--from-stage", default="plan", choices=list(STAGES))
    parser.add_argument("--to-stage", default="compare", choices=list(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="Print the full plan; write no files, run no export/model/bootstrap.")
    parser.add_argument("--force", action="store_true", help="Overwrite this window-closure namespace only.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages whose recorded hashes still match.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(
            experiment_id=args.experiment, shifts=args.shifts,
            from_stage=args.from_stage, to_stage=args.to_stage,
            dry_run=args.dry_run, force=args.force, resume=args.resume,
        ),
        indent=2, default=str,
    ))
