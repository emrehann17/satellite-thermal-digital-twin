#!/usr/bin/env python3
"""CLI/runner for the generic multi-experiment pairwise domain-classifier
(covariate-separability) diagnostic (src/domain_classifier_audit.py). Thin
dispatcher only -- no scientific logic is implemented here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_classifier_audit import run_analysis


def main(
    experiments: list[str] | None = None,
    all_enabled: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    return run_analysis(experiments=experiments, all_enabled=all_enabled, dry_run=dry_run, force=force)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generic multi-experiment pairwise domain-classifier "
            "(covariate-separability) diagnostic. Measures how well two "
            "regions can be distinguished using predictor features alone "
            "(never burned labels, coordinates, or experiment identity). "
            "Not a burned-area prediction model; does not establish "
            "causality."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit list of experiment IDs (core/regions.py registry). All unordered pairs are generated automatically.",
    )
    selection.add_argument(
        "--all-enabled", action="store_true",
        help="Resolve every enabled, non-legacy registry experiment with a canonical Step8A dataset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve experiments/pairs and print the plan; no model fitting, no files written.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs produced by a different analysis_id.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(experiments=args.experiments, all_enabled=args.all_enabled, dry_run=args.dry_run, force=args.force),
        indent=2, default=str,
    ))
