#!/usr/bin/env python3
"""Independent single-AOI window-closure runner (dry-run reviewed only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_region_window_closure.contract import ACTUAL_AOIS, MultiRegionWindowClosureError
from src.multi_region_window_closure.execution import regional_dry_run, render_regional_dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=list(ACTUAL_AOIS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-actual", action="store_true", help="Explicit actual-execution guard; never implied.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiments-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run and args.execute_actual:
        raise MultiRegionWindowClosureError("--dry-run and --execute-actual are mutually exclusive.")
    if args.force and not args.execute_actual:
        raise MultiRegionWindowClosureError("--force requires --execute-actual.")
    if args.resume and args.force:
        raise MultiRegionWindowClosureError("--resume and --force are mutually exclusive.")
    if not args.dry_run:
        if not args.execute_actual:
            raise MultiRegionWindowClosureError("Actual execution requires --execute-actual.")
        from src.multi_region_window_closure.production import ProductionRegionalEngine
        from src.multi_region_window_closure.driver import run_regional_actual
        engine = ProductionRegionalEngine(aoi=args.experiment, experiments_root=args.experiments_root)
        identity = engine.inspect_identity()
        result = run_regional_actual(aoi=args.experiment, analysis_id=identity["analysis_id"], output_root=args.output_root or PROJECT_ROOT/"outputs"/"diagnostics"/"window_closure_region", engine=engine, config_hash=identity["config_hash"], input_hash=identity["input_hash"], resume=args.resume, force=args.force, execute_actual=True)
        print(json.dumps(result, indent=2, default=str)); return 0
    result = regional_dry_run(
        args.experiment, output_root=args.output_root,
        experiments_root=args.experiments_root,
    )
    print(json.dumps(result, indent=2, default=str) if args.json else render_regional_dry_run(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
