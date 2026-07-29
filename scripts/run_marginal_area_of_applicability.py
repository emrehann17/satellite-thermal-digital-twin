#!/usr/bin/env python3
"""CLI/runner for the generic, directed, label-blind marginal
Area-of-Applicability analysis (src/marginal_area_of_applicability.py).

Thin dispatcher only -- no scientific logic is implemented here. Support
definitions, population resolution, the label firewall and every numeric rule
live in the module.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.marginal_area_of_applicability import run_analysis


def main(
    experiments: list[str] | None = None,
    all_enabled: bool = False,
    dry_run: bool = False,
    force: bool = False,
    output_root: str | Path | None = None,
    experiments_root: str | Path | None = None,
) -> dict:
    # `output_root` / `experiments_root` are programmatic dependency-injection
    # points (tests, alternative namespaces); None means the canonical roots.
    return run_analysis(
        experiments=experiments,
        all_enabled=all_enabled,
        dry_run=dry_run,
        force=force,
        output_root=Path(output_root) if output_root is not None else None,
        experiments_root=Path(experiments_root) if experiments_root is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generic, directed, label-blind marginal Area-of-Applicability "
            "analysis: for every ordered source->target pair, whether target "
            "predictor values fall inside the marginal predictor support "
            "observed in the source AOI. Fits no model, reads no label."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit list of experiment IDs (core/regions.py registry).",
    )
    selection.add_argument(
        "--all-enabled", action="store_true",
        help="Resolve every enabled, non-legacy registry experiment with a canonical Step8A dataset.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve inputs/schema and print the plan; no analysis, no files written.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing marginal AoA outputs produced by a different analysis_id.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(
            experiments=args.experiments, all_enabled=args.all_enabled,
            dry_run=args.dry_run, force=args.force,
        ),
        indent=2, default=str,
    ))
