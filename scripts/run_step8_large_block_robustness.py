"""Thin runner for the frozen Step8 large-spatial-block robustness analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.step8_large_block_robustness import (
    EXPECTED_BLOCK_SIZES,
    EXPECTED_EXPERIMENTS,
    run_analysis,
)


def main(
    experiments: list[str] | None = None,
    block_sizes_cells: list[int] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Dispatch without duplicating any model, block, or bootstrap logic."""
    return run_analysis(
        experiments=list(EXPECTED_EXPERIMENTS) if experiments is None else experiments,
        block_sizes=list(EXPECTED_BLOCK_SIZES) if block_sizes_cells is None else block_sizes_cells,
        dry_run=dry_run,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen Step8 robustness at predefined 10-cell and 20-cell spatial blocks."
    )
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--block-sizes-cells", nargs="+", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = main(
        experiments=args.experiments,
        block_sizes_cells=args.block_sizes_cells,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
