#!/usr/bin/env python3
"""Manavgat REFERENCE REPLAY -- frozen result replayed through the regional wrapper.

This entry point is deliberately separate from the production regional CLI
(`scripts/run_window_closure_region.py`), which constrains `--experiment` to the
four `ACTUAL_AOIS` and can never reach this path. There is no `--experiment`
argument here at all: the AOI is fixed to the read-only reference AOI, so the
replay cannot be pointed at a production AOI or at an arbitrary new one.

The replay:

* reuses the frozen Manavgat scientific artefacts verbatim (verified copy),
* never contacts Earth Engine and never downloads anything,
* writes ONLY into `outputs/diagnostics/window_closure_region_replay/`,
* leaves the frozen Manavgat tree and the four production regional trees byte-
  and mtime-identical, which is asserted before and after,
* and emits `equivalence_report.json` / `.md` as the provenance record behind a
  later migration decision. It performs no migration itself.
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
    REFERENCE_AOI, MultiRegionWindowClosureError, reference_replay_scope,
)
from src.multi_region_window_closure.reference_replay import (
    ManavgatReferenceReplayEngine, ReferenceReplayError, build_equivalence_report,
    compare_guarded_snapshots, frozen_manavgat_root, render_equivalence_report,
    replay_contract_preflight, replay_output_root, snapshot_guarded_trees,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run the read-only contract preflight and stop.",
    )
    parser.add_argument(
        "--execute-replay", action="store_true",
        help="Explicit replay-execution guard; never implied.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path, help="Replay namespace root.")
    parser.add_argument(
        "--frozen-source", type=Path,
        help="Frozen Manavgat namespace (read-only). Defaults to the canonical one.",
    )
    parser.add_argument("--experiments-root", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only and args.execute_replay:
        raise ReferenceReplayError(
            "--preflight-only and --execute-replay are mutually exclusive."
        )
    if args.force and not args.execute_replay:
        raise ReferenceReplayError("--force requires --execute-replay.")
    if args.resume and args.force:
        raise ReferenceReplayError("--resume and --force are mutually exclusive.")

    frozen_source = Path(args.frozen_source) if args.frozen_source else frozen_manavgat_root()
    output_root = replay_output_root(args.output_root)

    preflight = replay_contract_preflight(frozen_source, args.experiments_root)
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, default=str))
        print("\nSTOP -- REPLAY NOT AUTHORIZED", file=sys.stderr)
        for mismatch in preflight["mismatches"]:
            print(f"  contract mismatch: {mismatch['field']}", file=sys.stderr)
        return 2

    if args.preflight_only or not args.execute_replay:
        if not args.preflight_only:
            print(
                "Preflight PASS. Actual replay requires --execute-replay.",
                file=sys.stderr,
            )
        print(json.dumps(preflight, indent=2, default=str) if args.json
              else _render_preflight(preflight))
        return 0 if args.preflight_only else 3

    # --- source mutation guard, before ---------------------------------------
    before = snapshot_guarded_trees()

    with reference_replay_scope(REFERENCE_AOI):
        from src.multi_region_window_closure.driver import run_regional_actual
        from src.multi_region_window_closure.schema import verify_manifest_digest
        from src.multi_region_window_closure.validation import evaluate_regional

        engine = ManavgatReferenceReplayEngine(
            frozen_source=frozen_source, experiments_root=args.experiments_root,
        )
        identity = engine.inspect_identity()
        replay_result = run_regional_actual(
            aoi=REFERENCE_AOI, analysis_id=identity["analysis_id"],
            output_root=output_root, engine=engine,
            config_hash=identity["config_hash"], input_hash=identity["input_hash"],
            resume=args.resume, force=args.force, execute_actual=True,
        )
        replay_root = Path(replay_result["namespace"])
        validator = json.loads(
            (replay_root / "validator_results.json").read_text(encoding="utf-8")
        )
        validator_summary = json.loads(
            (replay_root / "validator_summary.json").read_text(encoding="utf-8")
        )
        validator = {**validator_summary, "checks": validator}

        # --- source mutation guard, after ------------------------------------
        mutation_guard = compare_guarded_snapshots(before, snapshot_guarded_trees())

        report = build_equivalence_report(
            frozen_source=frozen_source, replay_root=replay_root,
            preflight=preflight, validator=validator,
            mutation_guard=mutation_guard,
            replay_result={**replay_result, "analysis_id": identity["analysis_id"]},
        )
        (replay_root / "equivalence_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        (replay_root / "equivalence_report.md").write_text(
            render_equivalence_report(report), encoding="utf-8",
        )

        # The two report files are new members of the namespace, so the manifest
        # is rebuilt and the validator re-run over the final tree. Skipping this
        # would leave a namespace whose own manifest no longer describes it.
        #
        # ORDER MATTERS: nothing may be written into the namespace after
        # `finalize_manifest`. An earlier version recorded the post-validation
        # status back into equivalence_report.json here, which rewrote a file the
        # manifest had already hashed and left REG-MANIFEST-COMPLETE failing on a
        # stale entry. The final status is therefore reported to stdout only, and
        # never folded back into a manifested artefact.
        context = {
            "aoi": REFERENCE_AOI, "analysis_id": identity["analysis_id"],
            "config_hash": identity["config_hash"], "input_hash": identity["input_hash"],
        }
        engine.finalize_manifest(replay_root, context)
        final_validator = evaluate_regional(
            replay_root, REFERENCE_AOI, write_results=True, require_final_status=True,
        )
        digest_ok, digest_evidence = verify_manifest_digest(replay_root)
        if final_validator["overall_status"] != "PASS" or not digest_ok:
            raise MultiRegionWindowClosureError(
                "BLOCKER: REPLAY_NAMESPACE_NOT_SELF_CONSISTENT -- final validator "
                f"{final_validator['overall_status']}, manifest digest ok={digest_ok} "
                f"({digest_evidence})."
            )

    payload = {
        "replay": replay_result,
        "preflight_status": preflight["status"],
        "validator_status": validator.get("overall_status"),
        "validator_status_after_report": final_validator["overall_status"],
        "sources_unchanged": mutation_guard["all_unchanged"],
        "equivalence_verdict": report["equivalence_verdict"],
        "migration_safe": report["migration_safe"],
        "equivalence_report": str(replay_root / "equivalence_report.json"),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if report["equivalence_verdict"] == "PASS" else 1


def _render_preflight(preflight: dict) -> str:
    lines = [
        "MANAVGAT REFERENCE REPLAY -- READ-ONLY CONTRACT PREFLIGHT",
        f"aoi: {preflight['aoi']} (role: {preflight['aoi_role']})",
        f"frozen source: {preflight['frozen_source']}",
        f"frozen analysis_id: {preflight['frozen_analysis_id']}",
        f"scientific fields compared: {preflight['scientific_field_count']}",
        f"scientific mismatches: {len(preflight['mismatches'])}",
    ]
    for difference in preflight["non_scientific_differences"]:
        lines.append(
            f"non-scientific difference: {difference['field']}: "
            f"{difference['old']} -> {difference['new']}"
        )
    for mismatch in preflight["mismatches"]:
        lines.append(f"MISMATCH: {mismatch['field']}")
    lines.append(preflight["verdict"])
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MultiRegionWindowClosureError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
