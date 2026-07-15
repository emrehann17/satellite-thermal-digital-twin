"""Thin runner for the preregistered Step8 large-spatial-block robustness
analysis on the FORMAL Step8B primary population (all_valid)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.step8_large_block_robustness_primary_all_valid import run_analysis


def main(dry_run: bool = False, force: bool = False, run_large_block_fit: bool = False) -> dict:
    """Dispatch without duplicating any model, block, or bootstrap logic."""
    return run_analysis(dry_run=dry_run, force=force, run_large_block_fit=run_large_block_fit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preregistered Step8 large-block robustness analysis for the "
            "formal Step8B primary population (all_valid)."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--run-large-block-fit", action="store_true",
        help=(
            "Only after reviewing the 2-cell equivalence audit: proceed to "
            "fit the 10/20-cell scientific conditions. Without this flag, "
            "the run stops after the equivalence gate and returns its "
            "results for review."
        ),
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = main(dry_run=args.dry_run, force=args.force, run_large_block_fit=args.run_large_block_fit)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())