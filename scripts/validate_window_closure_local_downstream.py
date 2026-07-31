#!/usr/bin/env python3
"""Deterministic validator for the window-closure LOCAL-DOWNSTREAM stage.

This script NEVER runs a stage: no dry run, no downstream chain, no Earth
Engine call, no model fit and no bootstrap. It only reads

  * a log file the user produced (``--mode dry-run``), and/or
  * the files already on disk in the dedicated diagnostics namespace
    (``--mode actual``),

and re-checks the technical, scientific and namespace/provenance contracts
against them.

Usage
-----
    python scripts/validate_window_closure_local_downstream.py \
      --experiment <experiment_id> \
      --shifts 0 7 14 \
      --mode dry-run \
      --log logs/window_closure_local_downstream_dryrun.log

    python scripts/validate_window_closure_local_downstream.py \
      --experiment <experiment_id> \
      --shifts 0 7 14 \
      --mode actual

Exit code is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.window_closure_sensitivity as wcs
from scripts.validate_window_closure_predictor_export import (
    Report,
    expected_variant_ids,
    parse_json_object_from_log,
    preregistration_analysis_id,
)

# Stages that must STILL be refused by an actual run once local-downstream is
# implemented. If either of them ever appears in IMPLEMENTED_ACTUAL_STAGES
# without its own review, this validator fails.
# Every stage of this analysis is now implemented and reviewed, so no
# stage remains locked. The guard is still asserted so that adding a NEW
# stage without implementing it is caught here.
LOCKED_ACTUAL_STAGES: tuple[str, ...] = ()

# --- Artefact classification --------------------------------------------------
# A serialized model in this namespace is NOT automatically a violation: the
# production downstream chain trains the Step7C MODIS->Landsat downscaling
# random forest and persists it. That artefact is production-equivalent and is
# exactly what Step7D predicts from. What must never appear is a FIRE-RISK
# model, or anything belonging to the still-locked model/compare stages.
MODEL_ARTIFACT_SUFFIXES: tuple[str, ...] = (".pkl", ".joblib", ".sav", ".onnx", ".h5")

#: Directory (inside a variant downstream tree) that owns the allowed model.
DOWNSCALING_MODEL_STAGE_DIR = "step7c"

#: Fire-risk / model-stage artefacts. None of these may ever exist here.
FIRE_RISK_ARTIFACT_TOKENS: tuple[str, ...] = (
    "step8b", "step8c", "step8d", "step8e",
    "baseline_vs_thermal", "thermal_model", "fire_risk", "model_comparison",
)
#: Compare-stage / bootstrap / cohort artefacts. Also always forbidden.
COMPARE_ARTIFACT_TOKENS: tuple[str, ...] = (
    "common_cohort", "shared_fold", "bootstrap",
    "paired_window_changes", "variant_metrics", "thermal_contribution",
    "window_closure_summary",
)

CLASS_OK = "ok"
CLASS_DOWNSCALING_MODEL = "downscaling_model"
CLASS_FIRE_RISK = "fire_risk_model"
CLASS_COMPARE = "compare_bootstrap_cohort"


def classify_stage_artifact(relative_path: str) -> str:
    """Which artefact class a namespace-relative path belongs to."""
    lowered = relative_path.lower()
    parts = lowered.split("/")
    name = parts[-1]

    if any(token in lowered for token in FIRE_RISK_ARTIFACT_TOKENS):
        return CLASS_FIRE_RISK
    if any(token in lowered for token in COMPARE_ARTIFACT_TOKENS):
        return CLASS_COMPARE
    if name.endswith(MODEL_ARTIFACT_SUFFIXES):
        # A persisted model is allowed ONLY as the production Step7C
        # downscaling model, inside a variant's own downstream tree.
        if wcs.LOCAL_DOWNSTREAM_ROOT_DIR in parts and DOWNSCALING_MODEL_STAGE_DIR in parts:
            return CLASS_DOWNSCALING_MODEL
        return CLASS_FIRE_RISK
    return CLASS_OK


def _sha256(path: Path) -> Optional[str]:
    return wcs.sha256_file(path) if path.is_file() else None


def _is_inside(path, root: Path) -> bool:
    resolved = Path(str(path)).resolve()
    root = Path(root).resolve()
    return resolved == root or root in resolved.parents


# =============================================================================
# Shared contract checks
# =============================================================================
def check_stage_lock(report: Report) -> None:
    report.technical(
        wcs.LOCAL_DOWNSTREAM_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES,
        "local-downstream is an implemented actual stage",
        f"implemented={list(wcs.IMPLEMENTED_ACTUAL_STAGES)}",
    )
    still_locked = [
        stage for stage in LOCKED_ACTUAL_STAGES
        if stage not in wcs.IMPLEMENTED_ACTUAL_STAGES
    ]
    report.namespace(
        sorted(still_locked) == sorted(LOCKED_ACTUAL_STAGES),
        "no unimplemented stage is reachable",
        f"unlocked={[s for s in LOCKED_ACTUAL_STAGES if s in wcs.IMPLEMENTED_ACTUAL_STAGES]}",
    )
    report.scientific(
        list(wcs.PRODUCTION_STAGE_SEQUENCE) == [
            "step5", "step5c", "step7a", "step7b", "step7c", "step7d", "step7e", "step8a",
        ],
        "the production downstream stage sequence is the canonical one",
        f"sequence={list(wcs.PRODUCTION_STAGE_SEQUENCE)}",
    )


def check_artifact_classes(
    report: Report, root: Path, label: str, *,
    downscaling_model_allowed: bool,
    created: Sequence[str] = (),
    modified: Sequence[str] = (),
) -> None:
    """Split the namespace into allowed and forbidden artefact classes.

    The production Step7C downscaling model is allowed when the run declares
    that it fits one AND the run did not create or modify it. Fire-risk models
    and compare/bootstrap/cohort artefacts are refused unconditionally.
    """
    fire_risk: list[str] = []
    compare: list[str] = []
    downscaling: list[str] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            kind = classify_stage_artifact(relative)
            if kind == CLASS_FIRE_RISK:
                fire_risk.append(relative)
            elif kind == CLASS_COMPARE:
                compare.append(relative)
            elif kind == CLASS_DOWNSCALING_MODEL:
                downscaling.append(relative)

    report.namespace(
        not fire_risk, f"{label}: no fire-risk model artefact exists",
        f"found={fire_risk[:4]}",
    )
    report.namespace(
        not compare, f"{label}: no compare/bootstrap/common-cohort artefact exists",
        f"found={compare[:4]}",
    )

    if not downscaling:
        return
    touched = sorted(
        relative for relative in downscaling
        if relative in set(created) | set(modified)
    )
    report.namespace(
        downscaling_model_allowed and not touched,
        f"{label}: pre-existing Step7C downscaling model is allowed and unchanged",
        f"declared_allowed={downscaling_model_allowed} touched={touched[:4]} "
        f"models={downscaling[:4]}",
    )


def check_variant_namespace(
    report: Report, variant_id: str, paths: Sequence[str], root: Path,
) -> None:
    experiment_paths = _PROJECT_ROOT / "outputs" / "experiments"
    canonical_variant = root / "variants" / wcs.CANONICAL_VARIANT_ID
    predictor_data = root / "variants" / variant_id / "data"
    prelabel = root / "prelabel_censor"
    downstream = root / "variants" / variant_id / wcs.LOCAL_DOWNSTREAM_ROOT_DIR

    outside = [p for p in paths if not _is_inside(p, downstream)]
    report.namespace(
        not outside,
        f"{variant_id} every downstream artefact lives under downstream/",
        f"outside={outside[:4]}",
    )
    for forbidden, name in (
        (experiment_paths, "outputs/experiments"),
        (canonical_variant, "the canonical variant"),
        (predictor_data, "the predictor-export data tree"),
        (prelabel, "the pre-label namespace"),
    ):
        leaked = [p for p in paths if _is_inside(p, forbidden)]
        report.namespace(
            not leaked,
            f"{variant_id} writes nothing into {name}",
            f"leaked={leaked[:4]}",
        )


# =============================================================================
# Dry-run mode
# =============================================================================
def validate_dry_run(
    report: Report, experiment_id: str, shifts: Sequence[int],
    log_path: Path, root: Path,
) -> None:
    if not log_path.is_file():
        report.technical(False, "dry-run log exists", f"missing: {log_path}")
        return
    payload = parse_json_object_from_log(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if payload is None:
        report.technical(False, "dry-run log contains a parsable JSON payload", str(log_path))
        return
    report.technical(True, "dry-run log contains a parsable JSON payload")

    frozen_id = preregistration_analysis_id(root)
    report.scientific(
        frozen_id is not None and payload.get("analysis_id") == frozen_id,
        "analysis_id matches preregistration",
        f"log={payload.get('analysis_id')!r} preregistration={frozen_id!r}",
    )
    report.technical(
        payload.get("experiment_id") == experiment_id,
        "experiment_id matches",
        f"log={payload.get('experiment_id')!r}",
    )
    report.technical(
        payload.get("dry_run") is True and payload.get("ran") is False,
        "payload is a dry run",
        f"dry_run={payload.get('dry_run')!r} ran={payload.get('ran')!r}",
    )
    report.technical(
        payload.get("planned_stages") == [wcs.LOCAL_DOWNSTREAM_STAGE],
        "planned stage is local-downstream only",
        f"planned_stages={payload.get('planned_stages')!r}",
    )
    report.namespace(
        payload.get("files_written") is False,
        "no dry-run file writes detected",
        f"files_written={payload.get('files_written')!r}",
    )
    for flag in ("gee_queries_run", "gee_exports_run", "model_fit", "bootstrap_run"):
        report.namespace(
            payload.get(flag) is False, f"{flag} is false in the dry run",
            f"{flag}={payload.get(flag)!r}",
        )

    summary = payload.get("local_downstream_summary")
    if not isinstance(summary, dict):
        report.technical(False, "dry run carries local_downstream_summary")
        return
    report.technical(True, "dry run carries local_downstream_summary")

    report.scientific(
        summary.get("canonical_processing_enabled") is False
        and summary.get("canonical_frozen_reference_only") is True,
        "canonical variant is frozen-reference-only",
        json.dumps({
            k: summary.get(k) for k in
            ("canonical_processing_enabled", "canonical_frozen_reference_only")
        }),
    )
    report.scientific(
        summary.get("common_cohort_created") is False,
        "no common cohort is planned in this stage",
        f"common_cohort_created={summary.get('common_cohort_created')!r}",
    )
    for flag in ("gee_queries_run", "gee_exports_run", "model_fit",
                 "downscaling_model_fit", "fire_risk_model_fit",
                 "fire_risk_model_stage_run", "bootstrap_run"):
        report.scientific(
            summary.get(flag) is False,
            f"local_downstream_summary.{flag} is false",
            f"{flag}={summary.get(flag)!r}",
        )
    # A dry run fits nothing, but it must DECLARE that an actual run would
    # train the production Step7C downscaling model -- and only that model.
    report.scientific(
        summary.get("downscaling_model_fit_planned") is True,
        "the dry run declares the production downscaling model fit",
        f"downscaling_model_fit_planned={summary.get('downscaling_model_fit_planned')!r}",
    )
    report.namespace(
        summary.get("all_paths_inside_dedicated_namespace") is True,
        "all planned downstream paths are contained",
        f"all_paths_inside_dedicated_namespace="
        f"{summary.get('all_paths_inside_dedicated_namespace')!r}",
    )
    report.scientific(
        list(summary.get("production_stage_sequence") or [])
        == list(wcs.PRODUCTION_STAGE_SEQUENCE),
        "the production stage sequence is declared explicitly",
        f"sequence={summary.get('production_stage_sequence')!r}",
    )
    helpers = summary.get("production_helpers") or {}
    report.scientific(
        sorted(helpers) == sorted(wcs.PRODUCTION_STAGE_SEQUENCE)
        and all(isinstance(value, str) and "." in value for value in helpers.values()),
        "every production stage names its reused helper",
        json.dumps(helpers, sort_keys=True),
    )

    wanted = expected_variant_ids(shifts)
    got = list(summary.get("nonzero_variant_ids") or [])
    report.technical(
        got == wanted, "every preregistered non-zero variant is planned",
        f"expected={wanted} got={got}",
    )
    report.technical(
        summary.get("predictor_binding_ready") is True,
        "the predictor binding is ready for every variant",
        f"predictor_binding_ready={summary.get('predictor_binding_ready')!r}",
    )
    report.technical(
        summary.get("all_predictor_artifacts_present") is True
        and summary.get("all_predictor_hashes_match") is True,
        "every predictor artefact is present and hash-matched",
        json.dumps({
            k: summary.get(k) for k in
            ("all_predictor_artifacts_present", "all_predictor_hashes_match")
        }),
    )

    plans = summary.get("variant_plans") or {}
    canonical_plan = plans.get(wcs.CANONICAL_VARIANT_ID) or {}
    report.scientific(
        canonical_plan.get("export_enabled") is False
        and canonical_plan.get("frozen_reference_only") is True
        and canonical_plan.get("planned_output_count") == 0,
        "canonical plan is frozen-reference-only with zero planned outputs",
        json.dumps({
            k: canonical_plan.get(k) for k in
            ("export_enabled", "frozen_reference_only", "planned_output_count")
        }),
    )

    expected_artifacts = summary.get("predictor_artifacts_per_variant")
    for variant_id in wanted:
        plan = plans.get(variant_id)
        if not isinstance(plan, dict):
            report.technical(False, f"{variant_id} is present in the plan")
            continue
        report.technical(
            plan.get("export_enabled") is True,
            f"{variant_id} is downstream-enabled",
            f"export_enabled={plan.get('export_enabled')!r}",
        )
        report.technical(
            plan.get("predictor_artifact_count") == expected_artifacts,
            f"{variant_id} binds {expected_artifacts} predictor artefacts",
            f"predictor_artifact_count={plan.get('predictor_artifact_count')!r}",
        )
        report.technical(
            plan.get("predictor_binding_ready") is True,
            f"{variant_id} predictor binding is ready",
            f"reason={plan.get('predictor_binding_reason')!r}",
        )
        report.scientific(
            sorted(plan.get("planned_stage_outputs") or {})
            == sorted(wcs.PRODUCTION_STAGE_SEQUENCE),
            f"{variant_id} plans exactly the production stage outputs",
            f"planned_stage_outputs={sorted(plan.get('planned_stage_outputs') or {})}",
        )
        report.scientific(
            plan.get("static_invariance_check_planned") is True
            and plan.get("label_invariance_check_planned") is True,
            f"{variant_id} plans the static and label invariance checks",
            json.dumps({
                k: plan.get(k) for k in
                ("static_invariance_check_planned", "label_invariance_check_planned")
            }),
        )
        report.scientific(
            bool(plan.get("feature_contract_source")),
            f"{variant_id} names the canonical Step8A feature contract source",
            f"feature_contract_source={plan.get('feature_contract_source')!r}",
        )
        report.scientific(
            plan.get("downscaling_model_fit_planned") is True
            and plan.get("fire_risk_model_fit") is False,
            f"{variant_id} plans the downscaling model fit and no fire-risk model",
            json.dumps({
                k: plan.get(k) for k in
                ("downscaling_model_fit_planned", "fire_risk_model_fit")
            }),
        )
        report.scientific(
            plan.get("baseline_binding_source") == "predictor_export_metadata"
            and plan.get("baseline_directory_scan_used") is False,
            f"{variant_id} plans an explicit baseline binding, not a directory scan",
            json.dumps({
                k: plan.get(k) for k in
                ("baseline_binding_source", "baseline_directory_scan_used")
            }),
        )
        planned_paths = [
            plan.get("planned_step8a_path"), plan.get("planned_step8a_stats_path"),
            plan.get("planned_input_root"), plan.get("planned_metadata_path"),
        ]
        step8a = plan.get("planned_step8a_path")
        expected_step8a = (
            root / "variants" / variant_id / wcs.LOCAL_DOWNSTREAM_ROOT_DIR
            / "step8a" / wcs.STEP8A_DATASET_NAME
        )
        report.namespace(
            step8a is not None and Path(step8a).resolve() == expected_step8a.resolve(),
            f"{variant_id} planned Step8A path is the dedicated namespace path",
            f"planned={step8a!r} expected={expected_step8a}",
        )
        stage_paths = list((plan.get("planned_stage_outputs") or {}).values())
        check_variant_namespace(
            report, variant_id,
            [p for p in planned_paths + stage_paths if p and p != plan.get("planned_metadata_path")],
            root,
        )

    # --- The dry run must CHANGE nothing (not: find nothing) ----------------
    # A namespace that has already been run against legitimately carries a
    # partial downstream tree. What has to hold is that the dry run created,
    # modified and deleted nothing -- proved by the bracketing snapshots the
    # dry run itself recorded, and re-checked against disk below.
    check_dry_run_state(report, payload, root)
    check_artifact_classes(
        report, root, "dry run",
        downscaling_model_allowed=bool(
            summary.get("downscaling_model_fit_planned")
        ),
        created=payload.get("dry_run_created_paths") or [],
        modified=payload.get("dry_run_modified_paths") or [],
    )
    check_frozen_hashes_from_log(report, payload)


def check_semantic_dtype_record(report: Report, variant_id: str, metadata: dict) -> None:
    """Every literal dtype difference must be an EXPLICITLY accepted one.

    A dtype difference is only tolerable when the metadata names it, classifies
    it as a declared discrete production code and records the codebook it was
    checked against. An unexplained difference, or an accepted one outside the
    declared code columns, fails.
    """
    contract = metadata.get("semantic_dtype_contract")
    literal = metadata.get("literal_dtype_differences")
    accepted = metadata.get("accepted_semantic_dtype_compatibilities")
    if not isinstance(contract, dict) or literal is None or accepted is None:
        report.scientific(
            False, f"{variant_id} records the semantic dtype contract",
            "semantic_dtype_contract / literal_dtype_differences / "
            "accepted_semantic_dtype_compatibilities missing",
        )
        return
    report.scientific(True, f"{variant_id} records the semantic dtype contract")

    declared = set(contract.get("discrete_code_columns") or [])
    domains = contract.get("production_code_domains") or {}
    accepted_columns = {str(record.get("column")) for record in accepted}
    literal_columns = {str(record.get("column")) for record in literal}

    unexplained = sorted(literal_columns - accepted_columns)
    report.scientific(
        not unexplained,
        f"{variant_id} every literal dtype difference is explicitly accepted",
        f"unexplained={unexplained[:4]}",
    )
    undeclared = sorted(accepted_columns - declared)
    report.scientific(
        not undeclared,
        f"{variant_id} accepted dtype compatibilities stay inside the declared "
        "discrete-code columns",
        f"undeclared={undeclared[:4]}",
    )

    bad: list[str] = []
    for record in accepted:
        column = str(record.get("column"))
        domain = set(domains.get(column) or [])
        codes = set(record.get("canonical_codes_present") or []) | set(
            record.get("variant_codes_present") or []
        )
        if record.get("compatibility") != "pass":
            bad.append(f"{column}: compatibility={record.get('compatibility')!r}")
        if record.get("semantic_type") != "nullable_integer_categorical_code":
            bad.append(f"{column}: semantic_type={record.get('semantic_type')!r}")
        if not domain or not codes <= domain:
            bad.append(f"{column}: codes {sorted(codes)} outside {sorted(domain)}")
        if record.get("nulls_preserved") is not True:
            bad.append(f"{column}: nulls were not preserved")
    report.scientific(
        not bad,
        f"{variant_id} accepted dtype compatibilities are production codes only",
        f"violations={bad[:4]}",
    )
    report.scientific(
        contract.get("exact_dtype_required_elsewhere") is True,
        f"{variant_id} keeps the exact dtype contract everywhere else",
        f"exact_dtype_required_elsewhere="
        f"{contract.get('exact_dtype_required_elsewhere')!r}",
    )


def check_dry_run_state(report: Report, payload: dict, root: Path) -> None:
    """Stage-owned state must be byte-identical before and after the dry run."""
    before = payload.get("stage_owned_snapshot_before")
    after = payload.get("stage_owned_snapshot_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        report.namespace(
            False, "the dry run recorded a stage-owned state snapshot",
            "stage_owned_snapshot_before/after missing from the payload",
        )
        return
    report.namespace(True, "the dry run recorded a stage-owned state snapshot")

    created = payload.get("dry_run_created_paths") or []
    modified = payload.get("dry_run_modified_paths") or []
    deleted = payload.get("dry_run_deleted_paths") or []
    report.namespace(
        not created and not modified and not deleted,
        "the dry run created, modified and deleted no stage-owned path",
        f"created={created[:4]} modified={modified[:4]} deleted={deleted[:4]}",
    )
    report.namespace(
        payload.get("stage_owned_snapshot_before_sha256") == before.get("digest")
        and payload.get("stage_owned_snapshot_after_sha256") == after.get("digest")
        and before.get("digest") == after.get("digest")
        and payload.get("stage_owned_snapshot_unchanged") is True,
        "pre-existing partial downstream state was unchanged",
        f"before={payload.get('stage_owned_snapshot_before_sha256')!r} "
        f"after={payload.get('stage_owned_snapshot_after_sha256')!r} "
        f"unchanged={payload.get('stage_owned_snapshot_unchanged')!r}",
    )

    # Re-hash what the snapshot recorded: nothing may have moved since.
    drifted: list[str] = []
    for relative, record in sorted((after.get("files") or {}).items()):
        if not isinstance(record, dict) or not record.get("path"):
            continue
        if _sha256(Path(record["path"])) != record.get("sha256"):
            drifted.append(relative)
    report.namespace(
        not drifted,
        "every recorded stage-owned file still hashes as the dry run saw it",
        f"drifted={drifted[:4]}",
    )

    # A dry run may never publish a status=pass local-downstream metadata.
    published = sorted(
        relative for relative in created
        if relative.endswith(wcs.LOCAL_DOWNSTREAM_METADATA_NAME)
    )
    report.namespace(
        not published,
        "the dry run published no local-downstream metadata",
        f"created={published[:4]}",
    )

    # The canonical variant must have no downstream tree at all, before or after.
    canonical_dir = root / "variants" / wcs.CANONICAL_VARIANT_ID
    canonical_stray = [
        str(canonical_dir / name)
        for name in (wcs.LOCAL_DOWNSTREAM_ROOT_DIR, wcs.LOCAL_DOWNSTREAM_METADATA_NAME)
        if (canonical_dir / name).exists()
    ]
    report.namespace(
        not canonical_stray,
        "the canonical variant has no downstream tree",
        f"found={canonical_stray[:4]}",
    )


def check_frozen_hashes_from_log(report: Report, payload: dict) -> None:
    """Re-hash the frozen inventory the dry run recorded and compare with disk."""
    inventory = payload.get("frozen_input_inventory") or {}
    if not inventory:
        report.namespace(False, "frozen hashes are unchanged", "no inventory in the log")
        return
    missing = wcs.missing_required_frozen_hashes(
        inventory, wcs.REQUIRED_FROZEN_INPUT_ROLES,
    )
    drifted: list[str] = []
    for role, entry in sorted(inventory.items()):
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        expected = entry.get("sha256")
        if expected is None:
            continue
        if _sha256(Path(entry["path"])) != expected:
            drifted.append(role)
    summary = payload.get("local_downstream_summary") or {}
    canonical_sha = summary.get("canonical_step8a_sha256")
    canonical_path = summary.get("canonical_step8a_path")
    canonical_ok = (
        canonical_sha is None or canonical_path is None
        or _sha256(Path(canonical_path)) == canonical_sha
    )
    report.namespace(
        not missing and not drifted and canonical_ok,
        "frozen hashes are unchanged",
        f"missing_required={missing} drifted={drifted[:4]} canonical_ok={canonical_ok}",
    )


# =============================================================================
# Actual mode
# =============================================================================
def validate_actual(
    report: Report, experiment_id: str, shifts: Sequence[int], root: Path,
    experiments_root: Optional[Path] = None,
) -> None:
    frozen_id = preregistration_analysis_id(root)
    if not report.technical(
        frozen_id is not None, "preregistration is readable",
        f"missing or unreadable under {root}",
    ):
        return

    wanted = expected_variant_ids(shifts)
    prelabel = wcs.prelabel_raster_path(experiment_id, root.parent)
    prelabel_sha = _sha256(prelabel)
    canonical_dataset = wcs.canonical_step8a_path(experiment_id, experiments_root)
    canonical_stats = wcs.canonical_step8a_stats_path(experiment_id, experiments_root)

    canonical_frame = None
    try:
        import pandas as pd

        if canonical_dataset.is_file():
            canonical_frame = pd.read_parquet(canonical_dataset)
    except Exception as exc:  # noqa: BLE001
        report.technical(False, "the frozen canonical Step8A dataset is readable", str(exc))

    lineage = None
    try:
        lineage = wcs.step8a_predictor_lineage(experiment_id, experiments_root)
    except wcs.WindowClosureError as exc:
        report.scientific(False, "the Step8A predictor lineage is derivable", str(exc))

    report.technical(
        canonical_dataset.is_file() and canonical_stats.is_file(),
        "the frozen canonical Step8A dataset and stats exist",
        f"dataset={canonical_dataset} stats={canonical_stats}",
    )

    seen_analysis_ids: set = set()
    # Whether EVERY completed variant declares the production downscaling model
    # fit. Only then is a persisted Step7C model an expected artefact.
    downscaling_declarations: list[bool] = []
    for variant_id in wanted:
        variant_dir = root / "variants" / variant_id
        metadata_path = variant_dir / wcs.LOCAL_DOWNSTREAM_METADATA_NAME
        if not report.technical(
            metadata_path.is_file(), f"{variant_id} local-downstream metadata exists",
            f"missing: {metadata_path}",
        ):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            report.technical(False, f"{variant_id} metadata is readable", str(exc))
            continue

        seen_analysis_ids.add(metadata.get("analysis_id"))
        report.technical(
            metadata.get("schema_version") == wcs.LOCAL_DOWNSTREAM_METADATA_SCHEMA,
            f"{variant_id} metadata schema is {wcs.LOCAL_DOWNSTREAM_METADATA_SCHEMA}",
            f"schema={metadata.get('schema_version')!r}",
        )
        report.scientific(
            metadata.get("analysis_id") == frozen_id,
            f"{variant_id} metadata analysis_id matches preregistration",
            f"metadata={metadata.get('analysis_id')!r}",
        )
        report.technical(
            metadata.get("status") == wcs.STATUS_PASS,
            f"{variant_id} metadata status is pass",
            f"status={metadata.get('status')!r}",
        )
        report.technical(
            metadata.get("experiment_id") == experiment_id
            and metadata.get("variant_id") == variant_id,
            f"{variant_id} metadata identifies its own experiment and variant",
            json.dumps({k: metadata.get(k) for k in ("experiment_id", "variant_id")}),
        )

        # --- Predictor binding re-verification ------------------------------
        predictor_metadata_path = variant_dir / wcs.PREDICTOR_METADATA_NAME
        report.scientific(
            metadata.get("predictor_metadata_sha256") == _sha256(predictor_metadata_path),
            f"{variant_id} predictor metadata hash still matches",
            f"recorded={metadata.get('predictor_metadata_sha256')!r} "
            f"current={_sha256(predictor_metadata_path)!r}",
        )
        recorded_predictor_hashes = metadata.get("predictor_artifact_sha256") or {}
        report.technical(
            len(recorded_predictor_hashes) == metadata.get("predictor_artifact_count"),
            f"{variant_id} records a hash for every bound predictor artefact",
            f"hashes={len(recorded_predictor_hashes)} "
            f"count={metadata.get('predictor_artifact_count')!r}",
        )
        drifted_predictors: list[str] = []
        try:
            predictor_metadata = json.loads(
                predictor_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, UnicodeDecodeError):
            predictor_metadata = {}
        for record in (predictor_metadata.get("artifact_inventory") or []):
            artifact_id = str((record or {}).get("artifact_id"))
            current = _sha256(Path(str((record or {}).get("path") or "")))
            if recorded_predictor_hashes.get(artifact_id) != current:
                drifted_predictors.append(artifact_id)
        report.namespace(
            not drifted_predictors,
            f"{variant_id} every bound predictor raster is unchanged",
            f"drifted={sorted(drifted_predictors)[:4]}",
        )

        # --- Artefact inventory ---------------------------------------------
        artifacts = metadata.get("artifact_inventory") or []
        report.technical(
            bool(artifacts) and len(artifacts) == metadata.get("artifact_count"),
            f"{variant_id} artefact inventory is complete",
            f"inventory={len(artifacts)} artifact_count={metadata.get('artifact_count')!r}",
        )
        missing: list[str] = []
        bad_hash: list[str] = []
        bad_contract: list[str] = []
        prelabel_leak: list[str] = []
        stages_seen: set = set()
        for record in artifacts:
            path = Path(str((record or {}).get("path") or ""))
            artifact_id = str((record or {}).get("artifact_id"))
            stages_seen.add((record or {}).get("stage"))
            if not path.is_file():
                missing.append(str(path))
                continue
            digest = _sha256(path)
            if digest != record.get("sha256") or digest != (
                metadata.get("artifact_sha256") or {}
            ).get(artifact_id):
                bad_hash.append(artifact_id)
            if prelabel_sha is not None and digest == prelabel_sha:
                prelabel_leak.append(artifact_id)
            if path.resolve() == prelabel.resolve():
                prelabel_leak.append(artifact_id)
            for field in ("stage", "role", "media_type", "producer", "input_roles",
                          "variant_derived", "status", "bytes"):
                if record.get(field) is None:
                    bad_contract.append(f"{artifact_id}: missing {field}")
            suffix = path.suffix.lower()
            if suffix in (".tif", ".tiff"):
                for field in ("band_count", "dtype", "width", "height", "crs",
                              "transform", "grid_signature", "finite_cell_count"):
                    if field not in record:
                        bad_contract.append(f"{artifact_id}: raster field {field} missing")
            elif suffix == ".parquet":
                for field in ("row_count", "column_count", "columns", "dtypes",
                              "key_column", "duplicate_key_count"):
                    if field not in record:
                        bad_contract.append(f"{artifact_id}: parquet field {field} missing")
            elif suffix in (".json", ".geojson"):
                if "deterministic_sha256" not in record:
                    bad_contract.append(f"{artifact_id}: json deterministic hash missing")

        report.technical(
            not missing, f"{variant_id} every recorded artefact exists",
            f"missing={missing[:4]}",
        )
        report.technical(
            not bad_hash, f"{variant_id} every artefact hash matches the metadata",
            f"mismatched={bad_hash[:4]}",
        )
        report.technical(
            not bad_contract,
            f"{variant_id} every artefact carries its full inventory contract",
            f"incomplete={bad_contract[:4]}",
        )
        report.namespace(
            not prelabel_leak,
            f"{variant_id} does not carry the pre-label raster as an artefact",
            f"leaked={sorted(set(prelabel_leak))[:4]}",
        )
        report.scientific(
            stages_seen == set(wcs.PRODUCTION_STAGE_SEQUENCE),
            f"{variant_id} produced an artefact for every production stage",
            f"stages={sorted(str(s) for s in stages_seen)}",
        )
        report.scientific(
            list(metadata.get("production_stage_sequence") or [])
            == list(wcs.PRODUCTION_STAGE_SEQUENCE),
            f"{variant_id} records the deterministic production stage sequence",
            f"sequence={metadata.get('production_stage_sequence')!r}",
        )

        check_variant_namespace(
            report, variant_id,
            [str(record.get("path")) for record in artifacts if record.get("path")],
            root,
        )

        # --- Step8A dataset, feature contract, invariance ---------------------
        dataset_path = Path(str(metadata.get("step8a_dataset_path") or ""))
        expected_dataset = (
            variant_dir / wcs.LOCAL_DOWNSTREAM_ROOT_DIR / "step8a" / wcs.STEP8A_DATASET_NAME
        )
        report.namespace(
            dataset_path.resolve() == expected_dataset.resolve(),
            f"{variant_id} Step8A dataset is at the dedicated namespace path",
            f"recorded={dataset_path} expected={expected_dataset}",
        )
        report.technical(
            dataset_path.is_file()
            and _sha256(dataset_path) == metadata.get("step8a_dataset_sha256"),
            f"{variant_id} Step8A dataset exists and hashes as recorded",
            f"sha256={_sha256(dataset_path)!r} "
            f"recorded={metadata.get('step8a_dataset_sha256')!r}",
        )
        report.scientific(
            metadata.get("canonical_step8a_sha256") == _sha256(canonical_dataset),
            f"{variant_id} pins the unchanged frozen canonical Step8A hash",
            f"recorded={metadata.get('canonical_step8a_sha256')!r} "
            f"current={_sha256(canonical_dataset)!r}",
        )
        for flag in ("feature_contract_passed", "static_invariance_passed",
                     "label_invariance_passed", "key_uniqueness_passed",
                     "reference_grid_matches_canonical", "frozen_hashes_unchanged"):
            report.scientific(
                metadata.get(flag) is True, f"{variant_id} {flag}",
                f"{flag}={metadata.get(flag)!r}",
            )
        check_semantic_dtype_record(report, variant_id, metadata)
        for flag in ("prelabel_used_as_predictor", "common_cohort_created",
                     "canonical_downstream_attempted", "canonical_outputs_modified",
                     "gee_queries_run", "gee_exports_run",
                     "fire_risk_model_fit", "fire_risk_model_stage_run",
                     "bootstrap_run", "primary_population_filter_applied"):
            report.namespace(
                metadata.get(flag) is False, f"{variant_id} {flag} is false",
                f"{flag}={metadata.get(flag)!r}",
            )
        # The production chain DOES train the Step7C downscaling random forest.
        # Reporting model_fit=false would misdescribe the run, so the metadata
        # must say so -- while the fire-risk model stays untrained (above).
        downscaling_declarations.append(
            metadata.get("downscaling_model_fit") is True
            and metadata.get("downscaling_model_stage") == DOWNSCALING_MODEL_STAGE_DIR
        )
        report.scientific(
            metadata.get("model_fit") is True
            and metadata.get("downscaling_model_fit") is True
            and metadata.get("downscaling_model_stage") == DOWNSCALING_MODEL_STAGE_DIR,
            f"{variant_id} reports the production downscaling model fit honestly",
            json.dumps({
                k: metadata.get(k) for k in
                ("model_fit", "downscaling_model_fit", "downscaling_model_stage")
            }),
        )
        # Step5's baseline stack must come from the hash-pinned predictor
        # inventory; a directory scan could admit an unmanaged raster.
        report.scientific(
            metadata.get("baseline_binding_source") == "predictor_export_metadata"
            and metadata.get("baseline_directory_scan_used") is False,
            f"{variant_id} bound its baseline stack from the predictor metadata",
            json.dumps({
                k: metadata.get(k) for k in
                ("baseline_binding_source", "baseline_directory_scan_used")
            }),
        )
        baseline_records = metadata.get("baseline_lst_binding") or []
        recorded_predictor = metadata.get("predictor_artifact_sha256") or {}
        bad_baseline = [
            str(record.get("baseline_year")) for record in baseline_records
            if record.get("product") != "scene_weighted_median"
            or recorded_predictor.get(record.get("source_artifact_id"))
            != record.get("source_sha256")
        ]
        report.scientific(
            bool(baseline_records)
            and len(baseline_records) == len(metadata.get("baseline_years") or [])
            and not bad_baseline,
            f"{variant_id} baseline binding is one hash-pinned median per year",
            f"bad_years={bad_baseline[:4]} records={len(baseline_records)}",
        )
        report.scientific(
            metadata.get("prelabel_positive_cell_count") == 0,
            f"{variant_id} records the pre-label positive cell count",
            f"prelabel_positive_cell_count={metadata.get('prelabel_positive_cell_count')!r}",
        )
        report.scientific(
            metadata.get("primary_population") == wcs.PRIMARY_POPULATION
            and isinstance(metadata.get("primary_population_row_count"), int),
            f"{variant_id} reports the primary population row count",
            json.dumps({
                k: metadata.get(k) for k in
                ("primary_population", "primary_population_row_count")
            }),
        )

        counts_ok = all(
            isinstance(metadata.get(key), int) for key in (
                "variant_row_count", "canonical_row_count", "overlap_row_count",
                "variant_only_row_count", "canonical_only_row_count",
                "burned_count", "unburned_count",
            )
        )
        report.scientific(
            counts_ok,
            f"{variant_id} reports every row-count field",
            json.dumps({
                k: metadata.get(k) for k in
                ("variant_row_count", "canonical_row_count", "overlap_row_count",
                 "variant_only_row_count", "canonical_only_row_count")
            }),
        )

        # --- Independent re-verification against the datasets ----------------
        if canonical_frame is not None and lineage is not None and dataset_path.is_file():
            try:
                import pandas as pd

                variant_frame = pd.read_parquet(dataset_path)
                contract = wcs.assert_step8a_feature_contract(
                    variant_frame, canonical_frame, lineage,
                )
                invariance = wcs.compare_step8a_invariance(
                    variant_frame, canonical_frame, contract, contract["key_column"],
                )
            except wcs.WindowClosureError as exc:
                report.scientific(
                    False,
                    f"{variant_id} Step8A contract and static/label invariance "
                    "re-verify against the frozen canonical dataset",
                    str(exc),
                )
            except Exception as exc:  # noqa: BLE001
                report.technical(
                    False, f"{variant_id} Step8A dataset is readable", str(exc)
                )
            else:
                report.scientific(
                    True,
                    f"{variant_id} Step8A contract and static/label invariance "
                    "re-verify against the frozen canonical dataset",
                )
                report.scientific(
                    invariance["overlap_row_count"] == metadata.get("overlap_row_count")
                    and invariance["variant_row_count"] == metadata.get("variant_row_count")
                    and invariance["canonical_row_count"] == metadata.get("canonical_row_count"),
                    f"{variant_id} recorded row counts are reproducible",
                    json.dumps({
                        "recomputed": {
                            k: invariance[k] for k in
                            ("variant_row_count", "canonical_row_count", "overlap_row_count")
                        },
                        "recorded": {
                            k: metadata.get(k) for k in
                            ("variant_row_count", "canonical_row_count", "overlap_row_count")
                        },
                    }),
                )

    report.scientific(
        bool(seen_analysis_ids) and seen_analysis_ids == {frozen_id},
        "analysis_id matches preregistration in every variant",
        f"preregistration={frozen_id!r} metadata={sorted(str(v) for v in seen_analysis_ids)}",
    )

    # --- Canonical isolation -------------------------------------------------
    canonical_dir = root / "variants" / wcs.CANONICAL_VARIANT_ID
    stray_canonical = sorted(
        str(p) for p in canonical_dir.rglob("*")
        if p.is_file() and p.name != "frozen_reference.json"
    ) if canonical_dir.exists() else []
    report.namespace(
        not stray_canonical,
        "no downstream file exists under the canonical variant",
        f"found={stray_canonical[:4]}",
    )
    report.namespace(
        not (canonical_dir / wcs.LOCAL_DOWNSTREAM_ROOT_DIR).exists(),
        "no canonical downstream directory was created",
        f"found={canonical_dir / wcs.LOCAL_DOWNSTREAM_ROOT_DIR}",
    )
    # An actual run legitimately persists the production Step7C downscaling
    # model; a fire-risk model or a compare/bootstrap/cohort artefact never.
    check_artifact_classes(
        report, root, "actual run",
        downscaling_model_allowed=bool(downscaling_declarations)
        and all(downscaling_declarations),
    )

    # --- Frozen input audit --------------------------------------------------
    inventory = wcs.frozen_input_inventory(experiment_id, experiments_root)
    missing = wcs.missing_required_frozen_hashes(
        inventory, wcs.REQUIRED_FROZEN_INPUT_ROLES,
    )
    report.namespace(
        not missing,
        "all required frozen identity inputs exist and hash",
        f"missing_or_unhashed={missing}",
    )
    drifted: list[str] = []
    for variant_id in wanted:
        metadata_path = root / "variants" / variant_id / wcs.LOCAL_DOWNSTREAM_METADATA_NAME
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        recorded = metadata.get("frozen_input_sha256_after") or {}
        for role, expected in sorted(recorded.items()):
            if expected is None:
                continue
            entry = inventory.get(role)
            path = Path(str((entry or {}).get("path") or ""))
            if entry is None:
                continue
            if _sha256(path) != expected:
                drifted.append(f"{variant_id}/{role}")
    report.namespace(
        not drifted, "frozen inputs are unchanged since the local downstream ran",
        f"drifted={sorted(set(drifted))[:4]}",
    )
    report.namespace(
        prelabel.is_file(),
        "the shared pre-label raster is still present and untouched",
        f"missing: {prelabel}",
    )


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the window-closure local-downstream stage against a log "
            "(--mode dry-run) and/or the files on disk (--mode actual). This "
            "script never runs a stage, a model fit or a bootstrap."
        )
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID.")
    parser.add_argument(
        "--shifts", nargs="+", type=int, default=list(wcs.DEFAULT_SHIFTS),
        help="Preregistered closure shifts in days (default: 0 7 14).",
    )
    parser.add_argument("--mode", required=True, choices=["dry-run", "actual"])
    parser.add_argument("--log", help="Log file to validate (required for --mode dry-run).")
    parser.add_argument(
        "--output-root",
        help="Diagnostics root override; defaults to the canonical namespace.",
    )
    parser.add_argument(
        "--experiments-root",
        help=(
            "Canonical outputs/experiments root override; defaults to the "
            "canonical namespace."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    experiments_root = Path(args.experiments_root) if args.experiments_root else None
    root = wcs.experiment_root(args.experiment, output_root)

    report = Report()
    check_stage_lock(report)

    if args.mode == "dry-run":
        if not args.log:
            report.technical(False, "--log is required for --mode dry-run")
        else:
            validate_dry_run(report, args.experiment, args.shifts, Path(args.log), root)
    else:
        validate_actual(report, args.experiment, args.shifts, root, experiments_root)

    body, exit_code = report.render()
    print(body)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
