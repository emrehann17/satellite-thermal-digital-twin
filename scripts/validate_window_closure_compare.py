#!/usr/bin/env python3
"""Deterministic validator for the window-closure COMPARE stage.

This script NEVER runs a stage: no dry run, no model fit, no bootstrap, no
Earth Engine call. It only reads a log (``--mode dry-run``) and/or the files on
disk (``--mode actual``), and RE-DERIVES every published number from the
model-stage artefacts, so a mis-stated table cannot pass.

Technical artefact validation, scientific-contract compliance and namespace /
provenance safety are reported SEPARATELY; no single scientific verdict is
produced by this stage or by this validator.

Usage
-----
    python scripts/validate_window_closure_compare.py \
      --experiment <experiment_id> --shifts 0 7 14 \
      --mode dry-run --log logs/window_closure_compare_dryrun.log

    python scripts/validate_window_closure_compare.py \
      --experiment <experiment_id> --shifts 0 7 14 --mode actual
"""
from __future__ import annotations

import argparse
import csv
import io
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
    parse_json_object_from_log,
    preregistration_analysis_id,
)

#: No stage exists after compare; the lock must therefore cover nothing, and
#: that is asserted explicitly rather than assumed.
STAGES_AFTER_COMPARE: tuple[str, ...] = tuple(
    stage for stage in wcs.STAGES[wcs.STAGES.index(wcs.COMPARE_STAGE) + 1:]
)
TOLERANCE = 1e-9

#: The one file under ``compare/`` that is NOT an output artefact: it is the
#: stage-control document that RECORDS the output hashes. Listing it in its own
#: ``output_artifacts`` would create a self-hash cycle (the recorded digest is
#: part of the file being digested), so it is allow-listed here instead --
#: explicitly, by exact relative path, and it is the only entry. Everything else
#: found on disk must be recorded, or it is a stray file and the stage fails.
COMPARE_CONTROL_FILES: frozenset[str] = frozenset({
    f"{wcs.COMPARE_ROOT_DIR}/{wcs.COMPARE_METADATA_NAME}",
})


def _sha256(path: Path) -> Optional[str]:
    return wcs.sha256_file(path) if path.is_file() else None


def _is_inside(path, root: Path) -> bool:
    resolved = Path(str(path)).resolve()
    root = Path(root).resolve()
    return resolved == root or root in resolved.parents


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _number(value) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


# =============================================================================
# Shared checks
# =============================================================================
def check_stage_lock(report: Report) -> None:
    report.technical(
        wcs.COMPARE_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES,
        "compare is an implemented actual stage",
        f"implemented={list(wcs.IMPLEMENTED_ACTUAL_STAGES)}",
    )
    report.technical(
        wcs.MODEL_STAGE in wcs.IMPLEMENTED_ACTUAL_STAGES,
        "the model stage remains implemented",
    )
    report.namespace(
        STAGES_AFTER_COMPARE == (),
        "no stage exists after compare",
        f"later_stages={list(STAGES_AFTER_COMPARE)}",
    )
    report.scientific(
        list(wcs.COMPARE_FAMILY_ORDER) == [
            wcs.COMPARISON_THERMAL_CONTRIBUTION, wcs.COMPARISON_CLOSURE_CHANGE,
            wcs.COMPARISON_CONTRIBUTION_CHANGE,
        ]
        and list(wcs.MODEL_METRICS) == ["roc_auc", "pr_auc", "brier"]
        and list(wcs.MODEL_FAMILIES) == ["baseline", "thermal"],
        "the deterministic comparison/metric/model-family orders are canonical",
        f"families={list(wcs.COMPARE_FAMILY_ORDER)} metrics={list(wcs.MODEL_METRICS)}",
    )


def check_wording(report: Report, label: str, payload) -> None:
    """Prose only: machine key names are identifiers, not claims."""
    try:
        wcs.assert_compare_wording(payload, label)
    except wcs.WindowClosureError as exc:
        report.scientific(False, f"{label}: no forbidden wording", str(exc))
        return
    report.scientific(True, f"{label}: no forbidden wording")


def check_containment(report: Report, root: Path, paths: Sequence[str], label: str) -> None:
    compare_dir = root / wcs.COMPARE_ROOT_DIR
    outside = [p for p in paths if not _is_inside(p, compare_dir)]
    report.namespace(
        not outside, f"{label}: every output lives under compare/",
        f"outside={outside[:4]}",
    )
    for name, base in (
        ("outputs/experiments", _PROJECT_ROOT / "outputs" / "experiments"),
        ("model/", root / wcs.MODEL_ROOT_DIR),
        ("variants/", root / "variants"),
        ("prelabel_censor/", root / "prelabel_censor"),
        ("config/", root / "config"),
    ):
        leaked = [p for p in paths if _is_inside(p, base)]
        report.namespace(
            not leaked, f"{label}: nothing is written into {name}",
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
        f"log={payload.get('analysis_id')!r}",
    )
    report.technical(
        payload.get("experiment_id") == experiment_id, "experiment_id matches",
    )
    report.technical(
        payload.get("dry_run") is True and payload.get("ran") is False,
        "payload is a dry run",
    )
    report.technical(
        payload.get("planned_stages") == [wcs.COMPARE_STAGE],
        "planned stage is compare only",
        f"planned_stages={payload.get('planned_stages')!r}",
    )
    report.namespace(
        payload.get("files_written") is False or payload.get("files_written") == [],
        "no dry-run file writes detected",
        f"files_written={payload.get('files_written')!r}",
    )

    summary = payload.get("compare_stage_summary")
    if not isinstance(summary, dict):
        report.technical(False, "dry run carries compare_stage_summary")
        return
    report.technical(True, "dry run carries compare_stage_summary")

    for flag in ("model_fit", "fire_risk_model_fit", "downscaling_model_fit",
                 "bootstrap_run", "bootstrap_recomputed", "compare_run",
                 "gee_queries_run", "gee_exports_run", "model_refit_planned",
                 "bootstrap_recompute_planned"):
        report.namespace(
            summary.get(flag) is False, f"{flag} is false in the dry run",
            f"{flag}={summary.get(flag)!r}",
        )
    for flag in ("compare_run_planned", "report_generation_planned",
                 "tables_generation_planned"):
        report.scientific(
            summary.get(flag) is True, f"the dry run declares {flag}",
            f"{flag}={summary.get(flag)!r}",
        )

    report.technical(
        summary.get("source_model_metadata_present") is True
        and summary.get("source_model_schema") == wcs.MODEL_METADATA_SCHEMA
        and summary.get("source_model_status") == wcs.STATUS_PASS,
        "the source model stage is bound and marked pass",
        json.dumps({k: summary.get(k) for k in
                    ("source_model_metadata_present", "source_model_schema",
                     "source_model_status")}),
    )
    report.namespace(
        summary.get("source_model_metadata_sha256")
        == _sha256(Path(str(summary.get("source_model_metadata_path") or ""))),
        "the model stage metadata hash matches disk",
    )
    drifted = [
        relative for relative, record in sorted(
            (summary.get("input_artifacts") or {}).items()
        )
        if not record.get("present") or not record.get("sha256_matches")
    ]
    report.namespace(
        not drifted, "every bound model artefact is present and hash-matched",
        f"drifted={drifted[:4]}",
    )
    report.technical(
        summary.get("input_binding_ready") is True,
        "the model-stage binding is ready",
        f"input_binding_ready={summary.get('input_binding_ready')!r}",
    )

    expected = wcs.compare_expected_cardinalities(summary.get("variant_order") or [])
    report.scientific(
        summary.get("expected_cardinalities") == expected,
        "the planned row cardinalities are exact",
        f"planned={summary.get('expected_cardinalities')!r} expected={expected}",
    )
    report.scientific(
        summary.get("variant_order") == wcs.compare_variant_order(
            summary.get("variant_order") or []
        )
        and summary.get("model_family_order") == list(wcs.MODEL_FAMILIES)
        and summary.get("metric_order") == list(wcs.MODEL_METRICS)
        and summary.get("comparison_family_order") == list(wcs.COMPARE_FAMILY_ORDER),
        "the deterministic orderings are declared",
        json.dumps({k: summary.get(k) for k in
                    ("variant_order", "model_family_order", "metric_order")}),
    )
    definitions = summary.get("raw_delta_definitions") or {}
    report.scientific(
        definitions.get(wcs.COMPARISON_THERMAL_CONTRIBUTION) == "thermal - baseline (raw)"
        and definitions.get(wcs.COMPARISON_CLOSURE_CHANGE)
        == "earlier_closure - canonical (raw)",
        "the raw delta directions are declared",
        json.dumps(definitions),
    )
    report.scientific(
        sorted(summary.get("allowed_statuses") or []) == sorted([
            wcs.INTERVAL_SUPPORTED_INCREASE, wcs.INTERVAL_SUPPORTED_DECREASE,
            wcs.INTERVAL_INCLUDES_ZERO,
        ]),
        "only the allowed bootstrap statuses are planned",
    )
    report.namespace(
        summary.get("all_paths_inside_compare_namespace") is True,
        "every planned output is contained in compare/",
    )
    check_containment(report, root, summary.get("planned_output_paths") or [], "dry run")
    check_dry_run_state(report, payload, root)
    check_wording(report, "dry run", summary)
    check_upstream_frozen(report, payload)


def check_dry_run_state(report: Report, payload: dict, root: Path) -> None:
    before = payload.get("compare_stage_owned_snapshot_before")
    after = payload.get("compare_stage_owned_snapshot_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        report.namespace(
            False, "the dry run recorded a compare stage-owned snapshot",
            "compare_stage_owned_snapshot_before/after missing",
        )
        return
    report.namespace(True, "the dry run recorded a compare stage-owned snapshot")
    created = payload.get("compare_dry_run_created_paths") or []
    modified = payload.get("compare_dry_run_modified_paths") or []
    deleted = payload.get("compare_dry_run_deleted_paths") or []
    report.namespace(
        not created and not modified and not deleted
        and payload.get("compare_stage_owned_snapshot_unchanged") is True
        and before.get("digest") == after.get("digest"),
        "pre-existing compare state was unchanged",
        f"created={created[:4]} modified={modified[:4]} deleted={deleted[:4]}",
    )
    # The model tree must be untouched by a compare dry run as well.
    report.namespace(
        (payload.get("model_dry_run_created_paths") or []) == []
        and (payload.get("model_dry_run_modified_paths") or []) == []
        and (payload.get("model_dry_run_deleted_paths") or []) == [],
        "the dry run touched no model stage-owned path",
    )


def check_upstream_frozen(report: Report, payload: dict) -> None:
    inventory = payload.get("frozen_input_inventory") or {}
    drifted = [
        role for role, entry in sorted(inventory.items())
        if isinstance(entry, dict) and entry.get("path") and entry.get("sha256")
        and _sha256(Path(entry["path"])) != entry["sha256"]
    ]
    report.namespace(
        not drifted, "canonical and upstream frozen inputs are unchanged",
        f"drifted={drifted[:4]}",
    )


# =============================================================================
# Actual mode
# =============================================================================
def validate_actual(
    report: Report, experiment_id: str, shifts: Sequence[int], root: Path,
    experiments_root: Optional[Path] = None,
) -> None:
    import pandas as pd

    frozen_id = preregistration_analysis_id(root)
    if not report.technical(
        frozen_id is not None, "preregistration is readable",
        f"missing or unreadable under {root}",
    ):
        return

    compare_dir = root / wcs.COMPARE_ROOT_DIR
    metadata_path = compare_dir / wcs.COMPARE_METADATA_NAME
    if not report.technical(
        metadata_path.is_file(), "compare stage metadata exists",
        f"missing: {metadata_path}",
    ):
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        report.technical(False, "compare stage metadata is readable", str(exc))
        return

    report.technical(
        metadata.get("schema_version") == wcs.COMPARE_METADATA_SCHEMA,
        f"metadata schema is {wcs.COMPARE_METADATA_SCHEMA}",
        f"schema={metadata.get('schema_version')!r}",
    )
    report.technical(
        metadata.get("status") == wcs.STATUS_PASS, "metadata status is pass",
        f"status={metadata.get('status')!r}",
    )
    report.scientific(
        metadata.get("analysis_id") == frozen_id
        and metadata.get("experiment_id") == experiment_id,
        "analysis_id and experiment_id match the preregistration",
    )
    report.scientific(
        metadata.get("primary_population") == wcs.PRIMARY_POPULATION,
        "the primary population is the preregistered one",
        f"primary_population={metadata.get('primary_population')!r}",
    )

    # --- Compare-stage execution flags describe COMPARE only ----------------
    for flag in ("model_fit", "fire_risk_model_fit", "downscaling_model_fit",
                 "bootstrap_run", "bootstrap_recomputed", "gee_queries_run",
                 "gee_exports_run", "canonical_outputs_modified",
                 "upstream_outputs_modified"):
        report.namespace(
            metadata.get(flag) is False, f"{flag} is false for the compare stage",
            f"{flag}={metadata.get(flag)!r}",
        )
    report.scientific(
        metadata.get("compare_run") is True, "compare_run is true",
        f"compare_run={metadata.get('compare_run')!r}",
    )
    report.namespace(
        metadata.get("frozen_hashes_unchanged") is True,
        "frozen hashes were unchanged during the compare stage",
    )

    # --- Source model binding ------------------------------------------------
    model_metadata_path = wcs.model_metadata_path(experiment_id, root.parent)
    report.namespace(
        metadata.get("source_model_metadata_sha256") == _sha256(model_metadata_path),
        "the source model stage metadata is unchanged",
        f"recorded={metadata.get('source_model_metadata_sha256')!r}",
    )
    report.technical(
        metadata.get("source_model_schema") == wcs.MODEL_METADATA_SCHEMA,
        "the source model schema is the model-stage schema",
    )
    drifted_inputs = [
        str(record.get("relative_path"))
        for record in (metadata.get("input_artifacts") or [])
        if _sha256(Path(str(record.get("path") or ""))) != record.get("sha256")
    ]
    report.namespace(
        not drifted_inputs, "every bound model artefact is unchanged",
        f"drifted={drifted_inputs[:4]}",
    )

    # --- Output artefacts ----------------------------------------------------
    outputs = metadata.get("output_artifacts") or []
    missing, bad_hash = [], []
    for record in outputs:
        path = Path(str((record or {}).get("path") or ""))
        if not path.is_file():
            missing.append(str(record.get("relative_path")))
        elif _sha256(path) != record.get("sha256"):
            bad_hash.append(str(record.get("relative_path")))
    report.technical(not missing, "every recorded output exists", f"missing={missing[:4]}")
    report.technical(
        not bad_hash, "every output hash matches the metadata", f"bad={bad_hash[:4]}",
    )
    # Containment is exact set algebra over resolved paths:
    #
    #   stray   = on disk - recorded - allow-listed control files
    #   missing = recorded - on disk
    #
    # so an unrecorded artefact still fails, and the stage-control metadata is
    # excused WITHOUT being recorded (and therefore without a fake or empty
    # SHA) in the very list it publishes.
    recorded_outputs = {
        Path(str(record.get("path"))).resolve()
        for record in outputs if (record or {}).get("path")
    }
    allowed_control_files = {
        (root / relative).resolve() for relative in COMPARE_CONTROL_FILES
    }
    actual_compare_files = {
        path.resolve() for path in compare_dir.rglob("*") if path.is_file()
    }

    def _label(path: Path) -> str:
        resolved, base = Path(path).resolve(), root.resolve()
        return (
            resolved.relative_to(base).as_posix()
            if base in resolved.parents else str(resolved)
        )

    stray = sorted(
        _label(p) for p in
        actual_compare_files - recorded_outputs - allowed_control_files
    )
    unrecorded_missing = sorted(
        _label(p) for p in recorded_outputs - actual_compare_files
    )
    report.namespace(not stray, "compare/ contains no unrecorded file", f"stray={stray[:4]}")
    report.namespace(
        not unrecorded_missing, "every recorded output is present under compare/",
        f"missing={unrecorded_missing[:4]}",
    )
    report.technical(
        metadata_path.resolve() not in recorded_outputs,
        "the stage-control metadata is not recorded in its own output_artifacts",
        f"self_recorded={metadata_path}",
    )
    check_containment(
        report, root, [str(r.get("path")) for r in outputs if r.get("path")], "actual run",
    )
    model_dir = root / wcs.MODEL_ROOT_DIR
    created_in_model = [
        str(r.get("relative_path")) for r in outputs
        if _is_inside(str(r.get("path") or ""), model_dir)
    ]
    report.namespace(
        not created_in_model, "no compare artefact was written under model/",
        f"found={created_in_model[:4]}",
    )

    # --- Cardinalities -------------------------------------------------------
    layout = wcs.compare_relative_layout()
    tables = {
        key: _read_csv(compare_dir / layout[key])
        for key in ("point_metrics_long", "thermal_contributions",
                    "closure_changes", "thermal_contribution_changes",
                    "bootstrap_evidence_matrix")
        if (compare_dir / layout[key]).is_file()
    }
    expected = wcs.compare_expected_cardinalities(metadata.get("variant_order") or [])
    for key, want in (
        ("point_metrics_long", expected["point_metrics"]),
        ("thermal_contributions", expected["thermal_contributions"]),
        ("closure_changes", expected["closure_changes"]),
        ("thermal_contribution_changes", expected["thermal_contribution_changes"]),
        ("bootstrap_evidence_matrix", expected["bootstrap_evidence_matrix"]),
    ):
        rows = tables.get(key)
        report.technical(
            rows is not None and len(rows) == want,
            f"{key} has exactly {want} rows",
            f"got={len(rows) if rows is not None else 'missing'}",
        )

    # --- Duplicates and vocabulary -------------------------------------------
    def _duplicates(rows, keys) -> list[str]:
        seen, dupes = set(), []
        for row in rows:
            identity = tuple(row.get(key) for key in keys)
            if identity in seen:
                dupes.append(str(identity))
            seen.add(identity)
        return dupes

    duplicate_reports = []
    for key, keys in (
        ("point_metrics_long", ("variant_id", "model_family", "metric")),
        ("thermal_contributions", ("variant_id", "metric")),
        ("closure_changes", ("variant_id", "model_family", "metric")),
        ("thermal_contribution_changes", ("variant_id", "metric")),
        ("bootstrap_evidence_matrix",
         ("comparison_family", "variant_id", "model_family", "metric")),
    ):
        duplicate_reports += [f"{key}: {d}" for d in _duplicates(tables.get(key, []), keys)]
    report.technical(
        not duplicate_reports, "no duplicate scientific row exists",
        f"duplicates={duplicate_reports[:4]}",
    )

    variants = set(metadata.get("variant_order") or [])
    bad_vocabulary: list[str] = []
    for key, rows in tables.items():
        for row in rows:
            if row.get("metric") and row["metric"] not in wcs.MODEL_METRICS:
                bad_vocabulary.append(f"{key}: metric {row['metric']}")
            if row.get("variant_id") and row["variant_id"] not in variants:
                bad_vocabulary.append(f"{key}: variant {row['variant_id']}")
            family = row.get("model_family")
            if family and family not in set(wcs.MODEL_FAMILIES) | {"thermal_minus_baseline"}:
                bad_vocabulary.append(f"{key}: model family {family}")
    report.scientific(
        not bad_vocabulary, "no unexpected metric, variant or model family appears",
        f"unexpected={bad_vocabulary[:4]}",
    )

    # --- Re-derivation from the model stage ----------------------------------
    point_of = {
        (row["variant_id"], row["model_family"], row["metric"]): _number(row["value"])
        for row in tables.get("point_metrics_long", [])
    }
    problems: list[str] = []
    for row in tables.get("thermal_contributions", []):
        expected_delta = (
            point_of[(row["variant_id"], "thermal", row["metric"])]
            - point_of[(row["variant_id"], "baseline", row["metric"])]
        )
        if abs(_number(row["contribution_delta"]) - expected_delta) > TOLERANCE:
            problems.append(f"contribution {row['variant_id']}/{row['metric']}")
        if row.get("raw_delta_definition") != "thermal - baseline (raw)":
            problems.append(f"contribution definition {row['variant_id']}")
    report.scientific(
        not problems, "thermal contribution recomputes as thermal minus baseline",
        f"problems={problems[:4]}",
    )

    closure_problems: list[str] = []
    for row in tables.get("closure_changes", []):
        expected_delta = (
            point_of[(row["variant_id"], row["model_family"], row["metric"])]
            - point_of[(wcs.CANONICAL_VARIANT_ID, row["model_family"], row["metric"])]
        )
        if abs(_number(row["closure_delta"]) - expected_delta) > TOLERANCE:
            closure_problems.append(f"{row['variant_id']}/{row['metric']}")
        if row.get("raw_delta_definition") != "earlier_closure - canonical (raw)":
            closure_problems.append(f"definition {row['variant_id']}")
        if row.get("reference_variant_id") != wcs.CANONICAL_VARIANT_ID:
            closure_problems.append(f"reference {row['variant_id']}")
    report.scientific(
        not closure_problems,
        "closure change recomputes as earlier minus canonical",
        f"problems={closure_problems[:4]}",
    )

    contribution_of = {
        (row["variant_id"], row["metric"]): _number(row["contribution_delta"])
        for row in tables.get("thermal_contributions", [])
    }
    change_problems: list[str] = []
    for row in tables.get("thermal_contribution_changes", []):
        expected_delta = (
            contribution_of[(row["variant_id"], row["metric"])]
            - contribution_of[(wcs.CANONICAL_VARIANT_ID, row["metric"])]
        )
        if abs(_number(row["contribution_change_delta"]) - expected_delta) > TOLERANCE:
            change_problems.append(f"{row['variant_id']}/{row['metric']}")
    report.scientific(
        not change_problems, "thermal-contribution change recomputes exactly",
        f"problems={change_problems[:4]}",
    )

    # --- Evidence matrix vs the model-stage bootstrap summary ----------------
    try:
        model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        model_metadata = {}
    model_rows = {
        (row.get("comparison"), row.get("variant_id"), row.get("model_family"),
         row.get("metric")): row
        for row in (model_metadata.get("comparisons") or [])
    }
    evidence_problems: list[str] = []
    status_problems: list[str] = []
    wording_problems: list[str] = []
    for row in tables.get("bootstrap_evidence_matrix", []):
        key = (row["comparison_family"], row["variant_id"], row["model_family"],
               row["metric"])
        source = model_rows.get(key)
        if source is None:
            evidence_problems.append(f"{key}: not in the model summary")
            continue
        for field, column in (("ci_low", "ci_low"), ("ci_high", "ci_high"),
                              ("point_delta", "point_delta")):
            left, right = _number(row[column]), source.get(field)
            if left is None or right is None:
                continue
            if abs(left - float(right)) > TOLERANCE:
                evidence_problems.append(f"{key}: {field}")
        for field in ("requested_replicates", "valid_replicates",
                      "invalid_replicates"):
            if int(row[field]) != int(source.get(field)):
                evidence_problems.append(f"{key}: {field}")
        expected_status = wcs.classify_change_interval(
            _number(row["ci_low"]), _number(row["ci_high"]),
        )
        if row["status"] != expected_status:
            status_problems.append(f"{key}: {row['status']} != {expected_status}")
        if row["evidence_statement"] != wcs.evidence_statement(row["metric"], row["status"]):
            wording_problems.append(f"{key}: evidence statement")
        if row["status"] == wcs.INTERVAL_INCLUDES_ZERO and (
            "uncertainty remains" not in row["evidence_statement"]
        ):
            wording_problems.append(f"{key}: missing uncertainty wording")
        try:
            wcs.validate_saved_bootstrap_replicate_counts(
                row["requested_replicates"], row["valid_replicates"],
                row["invalid_replicates"], int(source["valid_replicates"]),
            )
        except wcs.WindowClosureError:
            evidence_problems.append(f"{key}: replicate counts")
    report.scientific(
        not evidence_problems,
        "every evidence row matches the model-stage bootstrap summary",
        f"problems={evidence_problems[:4]}",
    )
    report.scientific(
        not status_problems, "every status follows exactly from its interval",
        f"problems={status_problems[:4]}",
    )
    report.scientific(
        not wording_problems,
        "every evidence statement matches its metric and status",
        f"problems={wording_problems[:4]}",
    )
    brier_rows = [
        row for row in tables.get("bootstrap_evidence_matrix", [])
        if row["metric"] == "brier"
    ]
    report.scientific(
        all(
            "Brier score" in row["evidence_statement"]
            for row in brier_rows
            if row["status"] != wcs.INTERVAL_INCLUDES_ZERO
        ) and any(
            "lower Brier score" in row["evidence_statement"] or
            "higher Brier score" in row["evidence_statement"] or
            "direction of the metric change" in row["evidence_statement"]
            for row in brier_rows
        ) if brier_rows else True,
        "Brier evidence wording distinguishes lower and higher scores",
    )

    # --- Conclusions: no single overall verdict ------------------------------
    conclusions_path = compare_dir / layout["scientific_conclusions"]
    if conclusions_path.is_file():
        conclusions = json.loads(conclusions_path.read_text(encoding="utf-8"))
        report.scientific(
            conclusions.get("single_overall_scientific_verdict_produced") is False
            and conclusions.get("majority_vote_across_metrics_taken") is False
            and metadata.get("single_overall_scientific_verdict_produced") is False,
            "no single overall scientific verdict and no majority vote is produced",
        )
        report.scientific(
            sorted(conclusions.get("conclusions_by_metric") or {})
            == sorted(wcs.MODEL_METRICS),
            "conclusions are grouped by metric",
        )
        report.scientific(
            sorted(conclusions.get("conclusions_by_comparison_family") or {})
            == sorted(wcs.COMPARE_FAMILY_ORDER),
            "conclusions are grouped by comparison family",
        )
        report.technical(
            conclusions.get("technical_validation_status") == wcs.STATUS_PASS,
            "technical validation is reported separately from the evidence",
        )
        check_wording(report, "scientific conclusions", conclusions)
    else:
        report.technical(False, "the scientific conclusions document exists")

    # --- Markdown: display rounding only --------------------------------------
    report_path = compare_dir / layout["report"]
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        check_wording(report, "markdown report", text)
        rounding_problems: list[str] = []
        for row in tables.get("bootstrap_evidence_matrix", []):
            value = _number(row["point_delta"])
            if value is None:
                continue
            shown = f"{value:.{wcs.COMPARE_DISPLAY_DECIMALS}f}"
            if shown not in text:
                rounding_problems.append(f"{row['comparison_family']}/{row['metric']}")
        report.technical(
            not rounding_problems,
            "markdown displays the machine-readable values at the declared rounding",
            f"missing={rounding_problems[:4]}",
        )
        report.scientific(
            "Negative raw deltas indicate lower Brier scores." in text,
            "the markdown states the Brier raw-delta convention",
        )
        report.scientific(
            metadata.get("machine_readable_values_rounded") is False,
            "machine-readable values are stored unrounded",
        )
    else:
        report.technical(False, "the markdown report exists")

    check_wording(report, "compare stage metadata", metadata)

    # --- Upstream provenance --------------------------------------------------
    current = wcs.frozen_input_inventory(experiment_id, experiments_root)
    recorded_after = metadata.get("frozen_input_sha256_after") or {}
    drifted = sorted(
        role for role, entry in current.items()
        if isinstance(entry, dict) and entry.get("sha256") is not None
        and role in recorded_after and recorded_after[role] != entry["sha256"]
    )
    report.namespace(
        not drifted, "canonical and upstream hashes are unchanged",
        f"drifted={drifted[:4]}",
    )
    model_drift = sorted(
        role for role, digest in recorded_after.items()
        if role.startswith("model_artifact__") and digest is not None
    )
    report.namespace(
        all(
            _sha256(model_dir / role[len("model_artifact__"):]) == recorded_after[role]
            for role in model_drift
        ),
        "every model artefact is unchanged since the compare stage ran",
    )


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the window-closure compare stage against a log "
            "(--mode dry-run) and/or the files on disk (--mode actual). This "
            "script never runs a stage, a model fit or a bootstrap."
        )
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--shifts", nargs="+", type=int, default=list(wcs.DEFAULT_SHIFTS),
    )
    parser.add_argument("--mode", required=True, choices=["dry-run", "actual"])
    parser.add_argument("--log")
    parser.add_argument("--output-root")
    parser.add_argument("--experiments-root")
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
