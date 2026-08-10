#!/usr/bin/env python3
"""CLI/runner for the generic multi-experiment burned-area spatial-structure
and descriptive comparison audit (src/burned_pattern_audit.py). Thin
dispatcher only -- no scientific logic is implemented here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.burned_pattern_audit import run_analysis


def main(
    experiments: list[str] | None = None,
    all_enabled: bool = False,
    dry_run: bool = False,
    force: bool = False,
    allow_superseded_sensitivity: bool = False,
    output_root: str | Path | None = None,
    scope: str | None = None,
) -> dict:
    # No output path is spelled out here: `src.burned_pattern_audit` owns the
    # canonical root and every namespace under it (see `resolve_layout`).
    return run_analysis(
        experiments=experiments,
        all_enabled=all_enabled,
        dry_run=dry_run,
        force=force,
        allow_superseded_sensitivity=allow_superseded_sensitivity,
        output_root=Path(output_root) if output_root is not None else None,
        scope=scope,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generic multi-experiment burned-area spatial-structure and "
            "descriptive comparison audit: spatially connected burned-cell "
            "components, patch-size distribution, elevation distribution, "
            "land-cover composition. Read-only against Step8A."
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
        "--output-root", default=None,
        help=(
            "Alternative output root. Defaults to the canonical "
            "outputs/diagnostics/burned_pattern_audit/. Each analysis scope gets "
            "its own namespace UNDER the root, so scopes never collide and a "
            "previous result never has to be preserved by renaming the root."
        ),
    )
    parser.add_argument(
        "--scope", default=None,
        help=(
            "Namespace under the output root. Derived from the selection when "
            "omitted ('all_enabled', or the sorted experiment IDs). Pass an "
            "empty string for the legacy flat layout."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve experiments/inputs and print the plan; write no files, compute no components.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs produced by a different analysis_id.")
    parser.add_argument(
        "--allow-superseded-sensitivity", action="store_true",
        help=(
            "Explicit-selection only: permit a superseded legacy AOI variant "
            "alongside its canonical successor as an AOI-definition sensitivity "
            "comparator. Never restores the legacy experiment to the canonical "
            "cohort, is rejected with --all-enabled, and is rejected if the "
            "selection contains no superseded experiment."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(
        main(
            experiments=args.experiments,
            all_enabled=args.all_enabled,
            dry_run=args.dry_run,
            force=args.force,
            allow_superseded_sensitivity=args.allow_superseded_sensitivity,
            output_root=args.output_root,
            scope=args.scope,
        ),
        indent=2, default=str,
    ))
