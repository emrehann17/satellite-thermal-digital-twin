#!/usr/bin/env python3
"""Independent reproduction check of the FROZEN FIVE-REGION analysis.

Re-executes the canonical within-region Step8B spatial-block CV and the
canonical Step10B/Step10C label-blind adaptation + evaluation for all 5*4 = 20
directed cross-region transfers, and compares the freshly computed ROC-AUC /
PR-AUC against the frozen artefacts.

This entry point defines NO scientific method: every model, feature list,
population rule, spatial-block rule, seed and transform comes from the
canonical pipeline modules. It writes ONLY into
`outputs/diagnostics/reproduction_check/` and never touches
`outputs/experiments/` or `outputs/cross_region/`.

    python scripts/run_reproduction_check.py
    python scripts/run_reproduction_check.py --cohort-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reproduction_validation.common import resolve_frozen_five_region_cohort
from src.reproduction_validation.five_region import run_five_region_reproduction_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-only", action="store_true",
        help="Resolve and print the frozen five-region cohort, then stop. "
             "Runs no model.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Override the validation output namespace (default: "
             "outputs/diagnostics/reproduction_check/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cohort_only:
        print(json.dumps(resolve_frozen_five_region_cohort(), indent=2))
        return 0

    payload = run_five_region_reproduction_check(output_root=args.output_root)

    within = payload["within_region"]
    coral = payload["coral_transfer"]
    print("=" * 72)
    print(f"status                              : {payload['status']}")
    print(f"cohort                              : {', '.join(payload['cohort'])}")
    print(f"within-region comparisons           : {within['observed_comparisons']}"
          f"/{within['expected_comparisons']}")
    print(f"max within-region |d ROC-AUC|       : {within['max_abs_roc_auc_difference']!r}")
    print(f"CORAL directions                    : {coral['observed_directed_directions']}"
          f"/{coral['expected_directed_directions']}")
    print(f"max CORAL |d ROC-AUC|               : {coral['max_abs_roc_auc_difference']!r}")
    print(f"report                              : {payload['_report_path']}")
    if payload["failures"]:
        print("FAILURES:")
        for failure in payload["failures"]:
            print(f"  - {failure}")
    print("=" * 72)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
