"""Execution-grain contracts for independent regional runs and synthesis.

Pure planning/validation helpers live here.  No function initializes Earth
Engine, fits a model, runs a bootstrap, or writes an analysis namespace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.multi_region_window_closure.bootstrap import SERIES_PER_AOI
from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    REFERENCE_AOI,
    REGIONAL_DIAGNOSTIC_NAMESPACE,
    REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS,
    REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS,
    REGIONAL_EXPECTED_TOTAL_FITS,
    REGIONAL_SCHEMA_VERSION,
    SCIENTIFIC_CONTRACT_ID,
    SYNTHESIS_AOIS,
    SYNTHESIS_DIAGNOSTIC_NAMESPACE,
    SYNTHESIS_SCHEMA_VERSION,
    MultiRegionWindowClosureError,
    assert_regional_aoi,
)

REGIONAL_DRY_RUN_PASS = "WINDOW-CLOSURE REGIONAL DRY-RUN PASS"
REGIONAL_DRY_RUN_BLOCKED = "WINDOW-CLOSURE REGIONAL DRY-RUN BLOCKED"
SYNTHESIS_DRY_RUN_BLOCKED = "WINDOW-CLOSURE SYNTHESIS DRY-RUN BLOCKED"
SYNTHESIS_DRY_RUN_PASS = "WINDOW-CLOSURE SYNTHESIS DRY-RUN PASS"

REGIONAL_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "config.json", "input_hashes.json", "repository_inventory.json",
    "window_dates.csv", "export_plan.csv", "cohort_inventory.csv",
    "fold_mapping.parquet", "variant_artifact_index.csv", "metrics.csv",
    "oof_predictions.parquet", "bootstrap_replicates.parquet",
    "bootstrap_summary.csv", "regional_summary.csv", "summary.json",
    "report.md", "manifest.json", "manifest.sha256", "validator_results.json",
    "validator_summary.json",
)

SYNTHESIS_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "config.json", "input_references.json", "input_hashes.json",
    "regional_result_index.csv", "four_region_synthesis.csv", "summary.json",
    "report.md", "validator_results.json", "validator_summary.json",
    "manifest.json", "manifest.sha256",
)


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def regional_analysis_id(
    aoi: str,
    frozen_config: Mapping[str, Any],
    canonical_hash: str,
    input_hashes: Mapping[str, Any],
) -> str:
    """Deterministic identity; AOI identity is an explicit hash input."""
    selected = assert_regional_aoi(aoi)
    return _digest({
        "aoi": selected,
        "schema_version": REGIONAL_SCHEMA_VERSION,
        "scientific_contract_id": SCIENTIFIC_CONTRACT_ID,
        "frozen_config": dict(frozen_config),
        "canonical_hash": canonical_hash,
        "input_hashes": dict(input_hashes),
    })


def synthesis_id(
    manavgat_reference_hash: str,
    regional_analysis_ids: Mapping[str, str],
    regional_manifest_hashes: Mapping[str, str],
) -> str:
    if set(regional_analysis_ids) != set(ACTUAL_AOIS):
        raise MultiRegionWindowClosureError("Synthesis requires exactly three regional analysis IDs.")
    if set(regional_manifest_hashes) != set(ACTUAL_AOIS):
        raise MultiRegionWindowClosureError("Synthesis requires exactly three regional manifest hashes.")
    return _digest({
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "manavgat_reference_hash": manavgat_reference_hash,
        "regional_analysis_ids": dict(sorted(regional_analysis_ids.items())),
        "regional_manifest_hashes": dict(sorted(regional_manifest_hashes.items())),
    })


def regional_namespace(aoi: str, analysis_id: str, output_root: Path | None = None) -> Path:
    selected = assert_regional_aoi(aoi)
    if not analysis_id:
        raise MultiRegionWindowClosureError("analysis_id must be non-empty.")
    root = output_root or Path("outputs") / "diagnostics" / REGIONAL_DIAGNOSTIC_NAMESPACE
    return Path(root) / selected / analysis_id


def synthesis_namespace(synth_id: str, output_root: Path | None = None) -> Path:
    if not synth_id:
        raise MultiRegionWindowClosureError("synthesis_id must be non-empty.")
    root = output_root or Path("outputs") / "diagnostics" / SYNTHESIS_DIAGNOSTIC_NAMESPACE
    return Path(root) / synth_id


def quarantine_regional_namespace(
    aoi: str, analysis_id: str, reason: str, output_root: Path,
    timestamp: str,
) -> Path | None:
    """Move only the selected AOI/analysis namespace into its AOI quarantine."""
    import shutil

    selected = assert_regional_aoi(aoi)
    source = regional_namespace(selected, analysis_id, output_root)
    if not source.exists():
        return None
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason).strip("_") or "force"
    target = source.parent / "_quarantine" / f"{timestamp}_{analysis_id}_{safe_reason}"
    if target.exists():
        raise MultiRegionWindowClosureError(f"Quarantine target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def regional_expected_counts() -> dict[str, int]:
    return {
        "expected_primary_estimator_fits": REGIONAL_EXPECTED_PRIMARY_ESTIMATOR_FITS,
        "expected_auxiliary_downscaling_fits": REGIONAL_EXPECTED_AUXILIARY_DOWNSCALING_FITS,
        "expected_total_fits": REGIONAL_EXPECTED_TOTAL_FITS,
        "comparison_series": SERIES_PER_AOI,
        "bootstrap_summary_rows": SERIES_PER_AOI,
        "requested_bootstrap_replicate_rows": SERIES_PER_AOI * 1000,
    }


def assert_rows_belong_to_aoi(rows: Sequence[Mapping[str, Any]], aoi: str) -> None:
    selected = assert_regional_aoi(aoi)
    seen = {str(row.get("aoi")) for row in rows}
    if seen != {selected}:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: REGIONAL_SCOPE_ESCAPE -- expected only {selected}, got {sorted(seen)}."
        )


def independent_statuses(statuses: Mapping[str, str]) -> dict[str, str]:
    """Normalize statuses independently; never derive one AOI from another."""
    unknown = set(statuses) - set(ACTUAL_AOIS)
    if unknown:
        raise MultiRegionWindowClosureError(f"Unknown regional status AOI(s): {sorted(unknown)}")
    return {aoi: str(statuses[aoi]) for aoi in statuses}


def regional_resume_matches(
    state: Mapping[str, Any], *, aoi: str, analysis_id: str,
    config_hash: str, input_hash: str,
) -> bool:
    """Resume is valid only for the same AOI and all frozen identities."""
    selected = assert_regional_aoi(aoi)
    return all((
        state.get("aoi") == selected,
        state.get("analysis_id") == analysis_id,
        state.get("config_hash") == config_hash,
        state.get("input_hash") == input_hash,
        state.get("status") == "PASS",
        bool(state.get("stage_output_hash")),
        isinstance(state.get("stage_output_inventory"), list),
    ))


def synthesis_gate(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Read-only gate over Manavgat plus the three independently validated AOIs."""
    blockers: list[str] = []
    required = set(SYNTHESIS_AOIS)
    missing = sorted(required - set(records))
    if missing:
        blockers.append(f"MISSING_REQUIRED_INPUT: {missing}")
    extra = sorted(set(records) - required)
    if extra:
        blockers.append(f"UNEXPECTED_INPUT: {extra}")
    if "evia_2021" in records:
        blockers.append("EXCLUDED_AOI_PRESENT: evia_2021")
    contract_ids: set[str] = set()
    for aoi in sorted(required & set(records)):
        record = records[aoi]
        if record.get("validator_status") != "PASS":
            blockers.append(f"REGIONAL_VALIDATOR_NOT_PASS: {aoi}")
        if not record.get("manifest_hash"):
            blockers.append(f"MISSING_MANIFEST_HASH: {aoi}")
        if not record.get("hashes_match", False):
            blockers.append(f"INPUT_HASH_DRIFT: {aoi}")
        if not record.get("summary_aoi_matches", False):
            blockers.append(f"SUMMARY_AOI_MISMATCH: {aoi}")
        if not record.get("analysis_ids_match", False):
            blockers.append(f"ANALYSIS_ID_MISMATCH: {aoi}")
        if not record.get("schemas_match", False):
            blockers.append(f"SCHEMA_MISMATCH: {aoi}")
        if not record.get("manifest_files_complete", False):
            blockers.append(f"MANIFEST_INCOMPLETE: {aoi}")
        if not record.get("canonical_hash"):
            blockers.append(f"CANONICAL_HASH_MISSING: {aoi}")
        contract_ids.add(str(record.get("scientific_contract_id", "")))
        expected_schema = (
            "window_closure_sensitivity.v1"
            if aoi == REFERENCE_AOI else REGIONAL_SCHEMA_VERSION
        )
        if record.get("schema_version") != expected_schema:
            blockers.append(f"SCHEMA_MISMATCH: {aoi}")
    if contract_ids and contract_ids != {SCIENTIFIC_CONTRACT_ID}:
        blockers.append("SCIENTIFIC_CONTRACT_MISMATCH")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "required_aois": list(SYNTHESIS_AOIS),
        "expected_rows": len(SYNTHESIS_AOIS) * SERIES_PER_AOI,
        "pooled_inference": False,
        "model_fit": False,
        "bootstrap_run": False,
    }


def assert_synthesis_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(SYNTHESIS_AOIS) * SERIES_PER_AOI:
        raise MultiRegionWindowClosureError("BLOCKER: SYNTHESIS_ROW_COUNT_MISMATCH")
    if {str(row.get("aoi")) for row in rows} != set(SYNTHESIS_AOIS):
        raise MultiRegionWindowClosureError("BLOCKER: SYNTHESIS_AOI_SET_MISMATCH")
    forbidden = {"pooled_estimate", "pooled_ci_low", "pooled_ci_high", "weight", "meta_analysis"}
    present = forbidden & {key for row in rows for key in row}
    if present:
        raise MultiRegionWindowClosureError(f"BLOCKER: POOLED_INFERENCE_DETECTED -- {sorted(present)}")
