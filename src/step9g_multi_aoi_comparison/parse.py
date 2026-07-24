"""Parse a single canonical Step9G v1 pair report (+ its sibling
preregistration.json) into normalized region-feature / pair-feature records,
validating the fixed scientific contract. Recomputes NOTHING: every
AUC/CI/direction/support_status/reversal_status value is read verbatim from
the frozen report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.multi_aoi_transfer_synthesis.schema_adapters import _resolve_experiment_prefix
from src.step8_large_block_robustness import sha256_file
import src.step9g_univariate_feature_auc_direction_reversal as step9g

NUMERIC_FEATURES = step9g.NUMERIC_FEATURES
EXPECTED_PRIMARY_POPULATION = step9g.PRIMARY_POPULATION
EXPECTED_BLOCK_SIZE_CELLS = step9g.BLOCK_SIZE_CELLS
EXPECTED_NOMINAL_BLOCK_SCALE = step9g.NOMINAL_BLOCK_SCALE
EXPECTED_BOOTSTRAP_REPLICATES = step9g.BOOTSTRAP_REPLICATES
EXPECTED_BOOTSTRAP_SEED = step9g.BOOTSTRAP_SEED


class ScientificContractError(ValueError):
    """Raised when a discovered pair report fails the required, fixed
    scientific contract (population / feature set / block-size /
    bootstrap configuration). Values are always READ from the report/
    preregistration, never assumed."""


def _interval_excludes_chance(ci_low: Optional[float], ci_high: Optional[float]) -> Optional[bool]:
    if ci_low is None or ci_high is None:
        return None
    return bool(ci_high < 0.5 or ci_low > 0.5)


def validate_contract(report: dict[str, Any], preregistration: dict[str, Any], pair_id: str) -> None:
    population = report.get("primary_population")
    if population != EXPECTED_PRIMARY_POPULATION:
        raise ScientificContractError(
            f"{pair_id}: primary_population={population!r}, expected {EXPECTED_PRIMARY_POPULATION!r}."
        )
    config = preregistration.get("scientific_configuration", preregistration)
    block_size = config.get("block_size_cells")
    if block_size != EXPECTED_BLOCK_SIZE_CELLS:
        raise ScientificContractError(
            f"{pair_id}: block_size_cells={block_size!r}, expected {EXPECTED_BLOCK_SIZE_CELLS!r}."
        )
    nominal_scale = config.get("nominal_block_scale")
    if nominal_scale != EXPECTED_NOMINAL_BLOCK_SCALE:
        raise ScientificContractError(
            f"{pair_id}: nominal_block_scale={nominal_scale!r}, expected {EXPECTED_NOMINAL_BLOCK_SCALE!r}."
        )
    bootstrap = config.get("bootstrap", {}) or {}
    if bootstrap.get("replicates") != EXPECTED_BOOTSTRAP_REPLICATES:
        raise ScientificContractError(
            f"{pair_id}: bootstrap.replicates={bootstrap.get('replicates')!r}, "
            f"expected {EXPECTED_BOOTSTRAP_REPLICATES!r}."
        )
    if bootstrap.get("seed") != EXPECTED_BOOTSTRAP_SEED:
        raise ScientificContractError(
            f"{pair_id}: bootstrap.seed={bootstrap.get('seed')!r}, expected {EXPECTED_BOOTSTRAP_SEED!r}."
        )
    rows = report.get("direction_reversal_table", [])
    features_present = {row.get("feature") for row in rows}
    missing_features = set(NUMERIC_FEATURES) - features_present
    if missing_features:
        raise ScientificContractError(
            f"{pair_id}: missing required feature(s) {sorted(missing_features)} in direction_reversal_table."
        )


def _side_record(
    row: dict[str, Any], prefix: str, experiment_id: str, feature: str,
    input_audit: dict[str, Any], report_primary_population: Optional[str],
) -> dict[str, Any]:
    audit = input_audit.get(experiment_id, {})
    ci_low = row.get(f"{prefix}_ci_low")
    ci_high = row.get(f"{prefix}_ci_high")
    return {
        "experiment_id": experiment_id,
        "feature": feature,
        "auc": row.get(f"{prefix}_auc"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "direction": row.get(f"{prefix}_direction"),
        "support_status": row.get(f"{prefix}_support_status"),
        "interval_excludes_chance": _interval_excludes_chance(ci_low, ci_high),
        "n_rows": audit.get("n_rows"),
        "n_burned": audit.get("n_burned"),
        "n_unburned": audit.get("n_unburned"),
        "n_large_blocks": audit.get("n_large_blocks"),
        "primary_population": audit.get("primary_population", report_primary_population),
    }


def parse_pair_report(report_path: Path) -> dict[str, Any]:
    """Reads one pair report + sibling preregistration; returns:
        {
          "pair_id": str, "report_path": str, "report_sha256": str,
          "source_experiment_id": str, "target_experiment_id": str,
          "region_records": {(experiment_id, feature): {...}},
          "pair_records": {(sorted_a, sorted_b, feature): {...}},
        }
    Recomputes nothing; every value is read verbatim from the frozen report.
    """
    pair_dir = report_path.parent
    pair_id = pair_dir.name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prereg_path = pair_dir / "step9g_preregistration.json"
    preregistration = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.is_file() else {}

    validate_contract(report, preregistration, pair_id)

    rows = report["direction_reversal_table"]
    input_audit = report.get("input_audit", {})
    report_population = report.get("primary_population")

    # Prefer the per-row source/target fields (present in every report
    # generated after that field was added); fall back to the pair
    # directory name itself ("<source>__<target>", by construction of
    # output_root_for()) for the one frozen legacy report that predates it.
    first_row = rows[0]
    source_id = first_row.get("source_experiment_id")
    target_id = first_row.get("target_experiment_id")
    if source_id is None or target_id is None:
        source_id, target_id = pair_id.split("__", 1)

    region_records: dict[tuple[str, str], dict[str, Any]] = {}
    pair_records: dict[tuple[str, str, str], dict[str, Any]] = {}

    a, b = sorted((source_id, target_id))
    source_is_a = source_id == a

    for row in rows:
        feature = row["feature"]
        prefix_source = _resolve_experiment_prefix(row, source_id)
        prefix_target = _resolve_experiment_prefix(row, target_id)
        if prefix_source is None or prefix_target is None:
            raise ScientificContractError(
                f"{pair_id}: could not resolve AUC field prefix for feature '{feature}'."
            )

        source_side = _side_record(row, prefix_source, source_id, feature, input_audit, report_population)
        target_side = _side_record(row, prefix_target, target_id, feature, input_audit, report_population)
        region_records[(source_id, feature)] = source_side
        region_records[(target_id, feature)] = target_side

        side_a, side_b = (source_side, target_side) if source_is_a else (target_side, source_side)
        auc_a, auc_b = side_a["auc"], side_b["auc"]
        auc_difference = (auc_b - auc_a) if auc_a is not None and auc_b is not None else None

        diff_ci_low = row.get("auc_difference_ci_low")
        diff_ci_high = row.get("auc_difference_ci_high")
        # The report's stored diff CI is always (target - source).
        # Canonicalize to (b - a): negate + swap if the report's source is b.
        if diff_ci_low is not None and diff_ci_high is not None:
            if source_is_a:
                canon_ci_low, canon_ci_high = diff_ci_low, diff_ci_high
            else:
                canon_ci_low, canon_ci_high = -diff_ci_high, -diff_ci_low
        else:
            canon_ci_low = canon_ci_high = None

        reversal_status = row.get("reversal_status")
        pair_records[(a, b, feature)] = {
            "experiment_a": a, "experiment_b": b, "feature": feature,
            "experiment_a_auc": auc_a, "experiment_b_auc": auc_b,
            "point_direction_reversal": row.get("point_direction_reversal"),
            "reversal_status": reversal_status,
            "bootstrap_supported_direction_reversal": reversal_status == "bootstrap_supported_direction_reversal",
            "auc_difference": auc_difference,
            "auc_difference_ci_low": canon_ci_low,
            "auc_difference_ci_high": canon_ci_high,
        }

    return {
        "pair_id": pair_id,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "underlying_analysis_id": report.get("analysis_id"),
        "report_schema_version": report.get("report_schema_version", report.get("schema_version")),
        "region_records": region_records,
        "pair_records": pair_records,
    }
