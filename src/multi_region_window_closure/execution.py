"""Read-only planners for per-AOI execution and four-region synthesis."""
from __future__ import annotations

import hashlib,json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.multi_region_window_closure.contract import (
    DEFAULT_SHIFTS, REFERENCE_AOI, REGIONAL_SCHEMA_VERSION,
    SCIENTIFIC_CONTRACT_ID, SYNTHESIS_SCHEMA_VERSION, VARIANTS,
    assert_regional_aoi, frozen_bootstrap_configuration,
    frozen_model_configuration, modis_season_policy,
)
from src.multi_region_window_closure.per_aoi import (
    REGIONAL_DRY_RUN_BLOCKED, REGIONAL_DRY_RUN_PASS,
    SYNTHESIS_DRY_RUN_BLOCKED, SYNTHESIS_DRY_RUN_PASS,
    regional_analysis_id, regional_expected_counts, regional_namespace,
    synthesis_gate,
)


def regional_dry_run(
    aoi: str,
    shifts: Sequence[int] = DEFAULT_SHIFTS,
    output_root: Path | None = None,
    experiments_root: Path | None = None,
) -> dict[str, Any]:
    """Plan exactly one AOI without creating a namespace or doing computation."""
    from src.multi_region_window_closure import cohort, dates, inputs, plan

    selected = assert_regional_aoi(aoi)
    blockers: list[str] = []
    hash_result = inputs.verify_canonical_hashes((selected,), experiments_root)
    if not hash_result["all_match"]:
        blockers.append("CANONICAL_HASH_DRIFT")
    inventory = inputs.frozen_input_inventory((selected,), experiments_root)
    missing = inputs.missing_required_inputs(inventory)
    if missing:
        blockers.append(f"MISSING_FROZEN_INPUT: {missing}")
    date_rows = dates.window_date_rows((selected,), shifts)
    dates.assert_date_contract(date_rows)
    export_rows = plan.export_plan_rows((selected,), output_root, experiments_root)
    plan.assert_export_plan(export_rows)
    requests = plan.request_accounting((selected,))
    feasibility = cohort.canonical_feasibility_inventory((selected,), experiments_root)
    feasibility_problems = cohort.feasibility_blockers(feasibility)
    blockers.extend(feasibility_problems)

    input_doc = inputs.input_hashes_document(inventory)
    canonical_record = hash_result["records"][selected]
    canonical_hash = canonical_record["actual_sha256"] or canonical_record["expected_sha256"]
    config = {
        "schema_version": REGIONAL_SCHEMA_VERSION,
        "scientific_contract_id": SCIENTIFIC_CONTRACT_ID,
        "aoi": selected,
        "variants": list(VARIANTS),
        "shift_days": [int(value) for value in shifts],
        "model_configuration": frozen_model_configuration(),
        "bootstrap_configuration": frozen_bootstrap_configuration(),
        "modis_season_policy": modis_season_policy(),
        **regional_expected_counts(),
    }
    analysis_id = regional_analysis_id(selected, config, canonical_hash, input_doc)
    namespace = regional_namespace(selected, analysis_id, output_root)
    return {
        "mode": "regional-dry-run",
        "schema_version": REGIONAL_SCHEMA_VERSION,
        "scientific_contract_id": SCIENTIFIC_CONTRACT_ID,
        "analysis_id": analysis_id,
        "aoi": selected,
        "namespace": str(namespace),
        "namespace_exists": namespace.exists(),
        "variants": list(VARIANTS),
        "window_dates": date_rows,
        "export_plan_row_count": len(export_rows),
        "request_accounting": requests,
        "cohort_feasibility": feasibility,
        "canonical_cohort_feasibility": feasibility.get(selected, {}),
        "shifted_common_cohort_status": "NOT_YET_MEASURED",
        "shifted_fold_feasibility_status": "NOT_YET_MEASURED",
        "canonical_hash_expected": canonical_record["expected_sha256"],
        "canonical_hash_actual": canonical_record["actual_sha256"],
        "fit_accounting": regional_expected_counts(),
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "files_written": False,
        "output_namespace_created": False,
        "network_calls": 0,
        "gee_initialize_calls": 0,
        "model_fit_calls": 0,
        "bootstrap_calls": 0,
        "final_token": REGIONAL_DRY_RUN_PASS if not blockers else REGIONAL_DRY_RUN_BLOCKED,
    }


def _read_input_record(aoi: str, reference: str | Path | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    root = Path(reference)
    if not root.is_dir():
        return None
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    validator_path = root / "validator_summary.json"
    if not (summary_path.is_file() and manifest_path.is_file() and validator_path.is_file()):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validator = json.loads(validator_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    from src.multi_region_window_closure.schema import verify_manifest_digest
    digest_path=root/"manifest.sha256"
    digest_ok,digest_evidence=verify_manifest_digest(root)
    digest=digest_path.read_text(encoding="ascii").strip() if digest_ok else None
    analysis_id = summary.get("analysis_id")
    expected_schema = (
        "window_closure_sensitivity.v1" if aoi == REFERENCE_AOI
        else REGIONAL_SCHEMA_VERSION
    )
    manifest_files_complete = True
    for entry in manifest.get("files", []):
        artifact = root / str(entry.get("relative_path", ""))
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry.get("sha256"):
            manifest_files_complete = False
            break
    return {
        "schema_version": summary.get("schema_version"),
        "scientific_contract_id": summary.get("scientific_contract_id"),
        "validator_status": validator.get("overall_status"),
        "validator_manifest_matches": validator.get("validated_manifest_sha256")==digest,
        "manifest_hash": digest,
        "hashes_match": digest_ok,
        "manifest_digest_evidence": digest_evidence,
        "analysis_id": analysis_id,
        "aoi": aoi,
        "manifest_declared_analysis_id": manifest.get("analysis_id"),
        "validator_declared_analysis_id": validator.get("analysis_id"),
        "summary_aoi_matches": summary.get("aoi") == aoi,
        "analysis_ids_match": (
            bool(analysis_id)
            and manifest.get("analysis_id") == analysis_id
            and validator.get("analysis_id") == analysis_id
        ),
        "schemas_match": (
            summary.get("schema_version") == expected_schema
            and manifest.get("schema_version") == expected_schema
        ),
        "manifest_files_complete": manifest_files_complete,
        "canonical_hash": summary.get("canonical_hash"),
    }


def synthesis_dry_run(references: Mapping[str, str | Path | None]) -> dict[str, Any]:
    """Read-only input gate; missing actual inputs are an expected blocker."""
    records: dict[str, Mapping[str, Any]] = {}
    missing_references: list[str] = []
    for aoi, reference in references.items():
        record = _read_input_record(aoi, reference)
        if record is None:
            missing_references.append(aoi)
        else:
            records[aoi] = record
    gate = synthesis_gate(records)
    blockers = list(gate["blockers"])
    blockers.extend(f"MISSING_ACTUAL_RESULT: {aoi}" for aoi in sorted(missing_references))
    status = "PASS" if not blockers else "BLOCKED"
    return {
        "mode": "synthesis-dry-run",
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "scientific_contract_id": SCIENTIFIC_CONTRACT_ID,
        "reference_aoi": REFERENCE_AOI,
        "input_gate": gate,
        "blockers": blockers,
        "status": status,
        "expected_output_rows": gate["expected_rows"],
        "files_written": False,
        "output_namespace_created": False,
        "regional_inputs_modified": False,
        "network_calls": 0,
        "gee_initialize_calls": 0,
        "model_fit_calls": 0,
        "bootstrap_calls": 0,
        "final_token": SYNTHESIS_DRY_RUN_PASS if status == "PASS" else SYNTHESIS_DRY_RUN_BLOCKED,
    }


def render_regional_dry_run(result: Mapping[str, Any]) -> str:
    requests = result["request_accounting"]
    counts = result["fit_accounting"]
    lines = [
        "WINDOW-CLOSURE REGIONAL -- READ-ONLY DRY-RUN",
        f"aoi: {result['aoi']}",
        f"schema_version: {result['schema_version']}",
        f"analysis_id: {result['analysis_id']}",
        f"namespace (planned only): {result['namespace']}",
        f"canonical hash expected: {result['canonical_hash_expected']}",
        f"canonical hash actual: {result['canonical_hash_actual']}",
        "predictor windows:",
        *[f"  {row['variant']}: {row['predictor_start']} .. {row['predictor_end']}" for row in result["window_dates"]],
        f"label window: {result['window_dates'][0]['label_start']} .. {result['window_dates'][0]['label_end']}",
        f"canonical cohort rows/positives: {result['canonical_cohort_feasibility'].get('complete_case_rows')} / {result['canonical_cohort_feasibility'].get('complete_case_positives')}",
        f"shifted common-cohort status: {result['shifted_common_cohort_status']}",
        f"shifted fold-feasibility status: {result['shifted_fold_feasibility_status']}",
        f"logical raster artefacts: {requests['logical_raster_artifact_count']}",
        f"minimum requests: {requests['minimum_download_request_count']}",
        f"expected requests: {requests['expected_download_request_count']}",
        f"planning upper bound: {requests['planning_upper_bound_request_count']}",
        f"hard-supported ceiling: {requests['hard_supported_request_ceiling']}",
        f"primary/auxiliary/total fits: {counts['expected_primary_estimator_fits']}/"
        f"{counts['expected_auxiliary_downscaling_fits']}/{counts['expected_total_fits']}",
        "network/GEE/model/bootstrap calls: 0/0/0/0",
        f"blockers: {result['blockers'] or 'none'}",
        str(result["final_token"]),
    ]
    return "\n".join(lines)


def render_synthesis_dry_run(result: Mapping[str, Any]) -> str:
    return "\n".join([
        "WINDOW-CLOSURE SYNTHESIS -- READ-ONLY DRY-RUN",
        f"schema_version: {result['schema_version']}",
        f"expected descriptive rows: {result['expected_output_rows']}",
        "pooled inference/model/bootstrap/network: forbidden/0/0/0",
        f"blockers: {result['blockers'] or 'none'}",
        str(result["final_token"]),
    ])
