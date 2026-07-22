#!/usr/bin/env python3
"""CLI/runner for the single-experiment Step8 big-spatial-block robustness
analysis (src/step8_big_block_robustness.py). Thin dispatcher only -- no
scientific logic is implemented here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step8_big_block_robustness import (
    DEFAULT_BLOCK_SIZES, regenerate_reports_from_frozen_artifacts, run_analysis,
)


def main(
    experiment: str, block_sizes: list[int] | None = None,
    dry_run: bool = False, force: bool = False, regenerate_reports_only: bool = False,
) -> dict:
    # Dispatched BEFORE anything in run_analysis: no preregistration
    # creation, no runtime scientific-config comparison, no fold
    # construction, no model fitting, no bootstrap setup. Block sizes,
    # population, model settings, and analysis_id all come from the
    # existing frozen preregistration/manifests, never from block_sizes/
    # CLI defaults -- --regenerate-reports-only ignores --block-sizes.
    if regenerate_reports_only:
        return regenerate_reports_from_frozen_artifacts(experiment_id=experiment, dry_run=dry_run)

    # Normal path: preserved exactly as before.
    return run_analysis(
        experiment_id=experiment,
        block_sizes=list(block_sizes) if block_sizes else list(DEFAULT_BLOCK_SIZES),
        dry_run=dry_run, force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Runs the existing Step8B baseline-vs-thermal spatial-CV and "
            "Step8C paired spatial-block bootstrap for ONE experiment at "
            "one or more large block sizes (default: 10 and 20 cells, "
            "approximately 5 km / 10 km), and compares against that "
            "experiment's existing small-block (2-cell) results. Read-only "
            "with respect to Step8A/B/C/E; writes only under "
            "outputs/experiments/<experiment>/robustness/step8_big_blocks/."
        ),
    )
    parser.add_argument("--experiment", required=True, help="core/regions.py EXPERIMENTS registry experiment_id.")
    parser.add_argument(
        "--block-sizes", nargs="+", type=int, default=list(DEFAULT_BLOCK_SIZES),
        help=(
            "Large block sizes in 500m cells (default: 10 20). Ignored "
            "when --regenerate-reports-only is set -- that mode always "
            "reads block sizes from the existing frozen preregistration."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan; no fit/bootstrap, no files written.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downstream (non-preregistration) outputs.")
    parser.add_argument(
        "--regenerate-reports-only", action="store_true",
        help=(
            "Report-only mode: requires an existing COMPLETED analysis "
            "(the immutable comparison/manifest.json preregistration plus "
            "per-block-size step8b_metrics.json/bootstrap_summary.json/"
            "fold_assignments.parquet from a prior full run) and "
            "regenerates ONLY the JSON/Markdown/CSV/manifest reporting "
            "artifacts. Reads block sizes, population, model settings, and "
            "analysis_id from that frozen preregistration -- never "
            "constructs a new runtime scientific configuration, never "
            "compares against it, and never requires --force. NEVER "
            "invokes preregistration creation, fold construction, model "
            "fitting, prediction generation, or bootstrap sampling, and "
            "never writes oof_predictions.parquet, fold_assignments.parquet, "
            "or bootstrap_replicates.parquet. Fails clearly (never falls "
            "back to a new analysis) if the prior run's artifacts are "
            "missing."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(args.experiment, args.block_sizes, args.dry_run, args.force, args.regenerate_reports_only),
        indent=2, default=str,
    ))
