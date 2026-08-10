#!/usr/bin/env python3
"""Manavgat LOCAL RECOMPUTATION PROBE -- really refit, really re-bootstrap.

Separate from both the production regional CLI and the reference-replay CLI.
There is no `--experiment` argument: the AOI is fixed to the read-only
reference AOI.

The probe reuses the frozen pre-fit artefacts (plan, predictor export, local
downstream) and REGENERATES the model and compare stages with the production
stage functions. It never contacts Earth Engine, never downloads anything, and
writes only into `outputs/diagnostics/window_closure_region_recompute_probe/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_region_window_closure.contract import (
    MultiRegionWindowClosureError, REFERENCE_AOI,
)
from src.multi_region_window_closure.reference_replay import (
    compare_guarded_snapshots, frozen_manavgat_root, replay_contract_preflight,
)
from src.multi_region_window_closure.recompute_probe import (
    RecomputeProbeError, build_probe_report, materialize_upstream,
    probe_output_root, recompute_model_and_compare, render_probe_report,
    replay_namespace_root, snapshot_probe_guarded_trees,
    summarize_existing_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--execute-probe", action="store_true",
        help="Explicit recomputation guard; never implied.",
    )
    parser.add_argument(
        "--report-only", type=Path, metavar="PROBE_NAMESPACE",
        help=(
            "Rebuild the comparison report from an existing probe namespace "
            "without refitting. Read-only with respect to model/ and compare/."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--experiments-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only and args.execute_probe:
        raise RecomputeProbeError(
            "--preflight-only and --execute-probe are mutually exclusive."
        )
    frozen_source = Path(args.frozen_source) if args.frozen_source else frozen_manavgat_root()
    output_root = probe_output_root(args.output_root)

    preflight = replay_contract_preflight(frozen_source, args.experiments_root)
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, default=str))
        print("\nSTOP -- RECOMPUTATION NOT AUTHORIZED", file=sys.stderr)
        return 2
    if args.preflight_only or not args.execute_probe:
        print(json.dumps(
            {k: v for k, v in preflight.items() if k != "comparisons"},
            indent=2, default=str,
        ))
        if not args.preflight_only:
            print("Preflight PASS. The probe requires --execute-probe.", file=sys.stderr)
            return 3
        return 0

    replay_root = replay_namespace_root()
    before = snapshot_probe_guarded_trees()

    if args.report_only:
        probe_root = Path(args.report_only)
        if not (probe_root / REFERENCE_AOI / "model").is_dir():
            raise RecomputeProbeError(
                f"BLOCKER: PROBE_NAMESPACE_INCOMPLETE -- {probe_root} has no "
                "recomputed model/ tree to report on."
            )
        materialization, recomputation = summarize_existing_probe(
            frozen_source=frozen_source, probe_root=probe_root,
        )
    else:
        probe_root = output_root / preflight["derived_production_analysis_id"]
        if probe_root.exists():
            raise RecomputeProbeError(
                f"BLOCKER: PROBE_NAMESPACE_ALREADY_EXISTS -- {probe_root}. The probe "
                "never overwrites an existing namespace."
            )
        probe_root.mkdir(parents=True)
        materialization = materialize_upstream(frozen_source, probe_root)
        recomputation = recompute_model_and_compare(
            frozen_source=frozen_source, probe_root=probe_root,
            experiments_root=args.experiments_root,
        )
    mutation_guard = compare_guarded_snapshots(before, snapshot_probe_guarded_trees())

    report = build_probe_report(
        frozen_source=frozen_source, probe_root=probe_root, replay_root=replay_root,
        preflight=preflight, materialization=materialization,
        recomputation=recomputation, mutation_guard=mutation_guard,
    )
    (probe_root / "determinism_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    (probe_root / "determinism_report.md").write_text(
        render_probe_report(report), encoding="utf-8",
    )
    payload = {
        "probe_namespace": str(probe_root),
        "preflight_status": preflight["status"],
        "model_fit": recomputation["model_fit"],
        "bootstrap_run": recomputation["bootstrap_run"],
        "determinism_verdict": report["determinism_verdict"],
        "equivalence_verdict": report["equivalence_verdict"],
        "max_numeric_abs_diff": report["max_numeric_abs_diff"],
        "exact_bit_for_bit": report["exact_bit_for_bit"],
        "sources_unchanged": mutation_guard["all_unchanged"],
        "safe_to_retire_old_physical_layout": report["safe_to_retire_old_physical_layout"],
        "determinism_report": str(probe_root / "determinism_report.json"),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if report["equivalence_verdict"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MultiRegionWindowClosureError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
