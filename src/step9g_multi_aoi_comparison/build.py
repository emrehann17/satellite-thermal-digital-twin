"""Top-level orchestration for the generic multi-experiment Step9G
univariate-AUC comparison: resolve experiments -> discover canonical pair
reports -> parse (report-only, no recomputation) -> cross-report
consistency validation -> assemble long/wide/pairwise tables -> manifest /
analysis_id. No experiment ID is hard-coded anywhere in this module.
"""
from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.regions import get_experiment
from src.step8_large_block_robustness import canonical_json, sha256_bytes, _git_commit
import src.step9g_univariate_feature_auc_direction_reversal as step9g

from .consistency import ConsistencyError, merge_pair_records, merge_region_records
from .discovery import discover_pairs, pair_report_root
from .parse import (
    EXPECTED_BLOCK_SIZE_CELLS,
    EXPECTED_BOOTSTRAP_REPLICATES,
    EXPECTED_BOOTSTRAP_SEED,
    EXPECTED_NOMINAL_BLOCK_SCALE,
    EXPECTED_PRIMARY_POPULATION,
    NUMERIC_FEATURES,
    ScientificContractError,
    parse_pair_report,
)

COMPARISON_SCHEMA_VERSION = "step9g.multi_aoi_univariate_auc_comparison.v1"
ADVISOR_CRITICAL_FEATURE = "elevation_mean"


class ComparisonError(SystemExit):
    """Fail-fast error for the generic multi-experiment Step9G comparison."""


def comparison_output_root() -> Path:
    """outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/comparison/"""
    return pair_report_root() / "comparison"


def comparison_output_dir(sorted_ids: tuple[str, ...]) -> Path:
    return comparison_output_root() / "__".join(sorted_ids)


def resolve_experiments(experiments: Optional[list[str]]) -> tuple[str, ...]:
    """Validate the caller-supplied explicit experiment list. No fixed
    allowed-AOI list: any experiment_id present in the core.regions
    registry is accepted, current or future."""
    if not experiments or not isinstance(experiments, (list, tuple)):
        raise ComparisonError("--experiments must be a non-empty list of experiment IDs.")
    if len(experiments) < 2:
        raise ComparisonError(f"At least 2 experiment IDs are required for a comparison; got {len(experiments)}.")
    seen: dict[str, int] = {}
    for entry in experiments:
        seen[entry] = seen.get(entry, 0) + 1
    duplicates = sorted(k for k, v in seen.items() if v > 1)
    if duplicates:
        raise ComparisonError(f"Duplicate --experiments entries are not allowed: {duplicates}.")
    for experiment_id in experiments:
        get_experiment(experiment_id)  # raises ValueError if unknown
    return tuple(experiments)


def _package_versions() -> dict[str, str]:
    names = {"numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn"}
    return {key: importlib.metadata.version(pkg) for key, pkg in names.items()}


def scientific_contract_summary() -> dict[str, Any]:
    return {
        "primary_population": EXPECTED_PRIMARY_POPULATION,
        "numeric_features": list(NUMERIC_FEATURES),
        "block_size_cells": EXPECTED_BLOCK_SIZE_CELLS,
        "nominal_block_scale": EXPECTED_NOMINAL_BLOCK_SCALE,
        "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "raw_auc_semantics": (
            "burned is the positive class; the raw (untransformed) feature "
            "value is used as the ranking score; AUC below 0.5 is a "
            "direction and is never inverted."
        ),
        "landcover_excluded_reason": step9g.LANDCOVER_EXCLUSION_REASON,
    }


def build_analysis_id(sorted_ids: tuple[str, ...], input_hashes: dict[str, str]) -> str:
    content = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "resolved_experiment_ids": sorted(sorted_ids),
        "input_report_sha256": dict(sorted(input_hashes.items())),
        "scientific_contract": scientific_contract_summary(),
    }
    return sha256_bytes(canonical_json(content).encode("utf-8"))


def _long_rows(sorted_ids: tuple[str, ...], region_records: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for experiment_id in sorted_ids:
        for feature in NUMERIC_FEATURES:
            record = region_records.get((experiment_id, feature))
            if record is None:
                continue
            rows.append({
                "experiment_id": experiment_id,
                "feature": feature,
                "auc": record["auc"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "direction": record["direction"],
                "support_status": record["support_status"],
                "interval_excludes_chance": record["interval_excludes_chance"],
                "n_rows": record["n_rows"],
                "n_burned": record["n_burned"],
                "n_unburned": record["n_unburned"],
                "n_large_blocks": record["n_large_blocks"],
                "primary_population": record["primary_population"],
                "source_pair_reports": ";".join(record["source_pair_reports"]),
            })
    return rows


def _wide_rows(sorted_ids: tuple[str, ...], region_records: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for feature in NUMERIC_FEATURES:
        row: dict[str, Any] = {"feature": feature}
        for experiment_id in sorted_ids:
            record = region_records.get((experiment_id, feature))
            row[f"{experiment_id}_auc"] = record["auc"] if record else None
            row[f"{experiment_id}_ci_low"] = record["ci_low"] if record else None
            row[f"{experiment_id}_ci_high"] = record["ci_high"] if record else None
            row[f"{experiment_id}_direction"] = record["direction"] if record else None
            row[f"{experiment_id}_support_status"] = record["support_status"] if record else None
        rows.append(row)
    return rows


def _pairwise_rows(
    available_pairs: list[tuple[str, str]], pair_records: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for a, b in available_pairs:
        for feature in NUMERIC_FEATURES:
            record = pair_records.get((a, b, feature))
            if record is None:
                continue
            rows.append({
                "experiment_a": a, "experiment_b": b, "feature": feature,
                "experiment_a_auc": record["experiment_a_auc"],
                "experiment_b_auc": record["experiment_b_auc"],
                "point_direction_reversal": record["point_direction_reversal"],
                "reversal_status": record["reversal_status"],
                "bootstrap_supported_direction_reversal": record["bootstrap_supported_direction_reversal"],
                "auc_difference": record["auc_difference"],
                "auc_difference_ci_low": record["auc_difference_ci_low"],
                "auc_difference_ci_high": record["auc_difference_ci_high"],
            })
    return rows


def advisor_critical_summary(
    sorted_ids: tuple[str, ...], region_records: dict[tuple[str, str], dict[str, Any]],
    feature: str = ADVISOR_CRITICAL_FEATURE,
) -> dict[str, Any]:
    rows = []
    for experiment_id in sorted_ids:
        record = region_records.get((experiment_id, feature))
        if record is None:
            continue
        rows.append({
            "experiment_id": experiment_id,
            "auc": record["auc"], "ci_low": record["ci_low"], "ci_high": record["ci_high"],
            "direction": record["direction"], "support_status": record["support_status"],
            "interval_excludes_chance": record["interval_excludes_chance"],
        })
    bootstrap_supported_higher = [
        r["experiment_id"] for r in rows
        if r["direction"] == "higher_values_rank_burned" and "bootstrap_supported" in (r["support_status"] or "")
    ]
    bootstrap_supported_lower = [
        r["experiment_id"] for r in rows
        if r["direction"] == "lower_values_rank_burned" and "bootstrap_supported" in (r["support_status"] or "")
    ]
    return {
        "feature": feature,
        "rows": rows,
        "bootstrap_supported_higher": bootstrap_supported_higher,
        "bootstrap_supported_lower": bootstrap_supported_lower,
    }


def pairwise_reversal_findings(
    available_pairs: list[tuple[str, str]], pair_records: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    """For every available unordered pair, which features (if any) show a
    bootstrap-supported direction reversal."""
    findings: dict[tuple[str, str], list[str]] = {}
    for a, b in available_pairs:
        supported = sorted(
            feature for feature in NUMERIC_FEATURES
            if (record := pair_records.get((a, b, feature))) is not None
            and record["bootstrap_supported_direction_reversal"]
        )
        findings[(a, b)] = supported
    return findings


def build_comparison(experiments: list[str], dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    requested_ids = tuple(experiments)
    resolve_experiments(experiments)
    sorted_ids = tuple(sorted(set(requested_ids)))

    discovery = discover_pairs(sorted_ids)
    available_paths = discovery["available"]  # {(a, b): Path}
    missing_pairs = discovery["missing"]  # [(a, b), ...]
    available_pairs = sorted(available_paths.keys())
    output_dir = comparison_output_dir(sorted_ids)

    planned_paths = {
        "multi_aoi_univariate_auc_comparison_json": str(output_dir / "multi_aoi_univariate_auc_comparison.json"),
        "multi_aoi_univariate_auc_long_csv": str(output_dir / "multi_aoi_univariate_auc_long.csv"),
        "multi_aoi_univariate_auc_wide_csv": str(output_dir / "multi_aoi_univariate_auc_wide.csv"),
        "pairwise_direction_reversal_summary_csv": str(output_dir / "pairwise_direction_reversal_summary.csv"),
        "multi_aoi_univariate_auc_comparison_md": str(output_dir / "multi_aoi_univariate_auc_comparison.md"),
        "manifest_json": str(output_dir / "manifest.json"),
    }

    if dry_run:
        return {
            "ran": False,
            "dry_run": True,
            "requested_experiment_ids": list(requested_ids),
            "resolved_experiment_ids": list(sorted_ids),
            "available_pairs": [f"{a}__{b}" for a, b in available_pairs],
            "missing_pairs": [f"{a}__{b}" for a, b in missing_pairs],
            "complete_pairwise_matrix": len(missing_pairs) == 0,
            "scientific_contract": scientific_contract_summary(),
            "output_root": str(output_dir),
            "planned_output_paths": planned_paths,
        }

    parsed = [parse_pair_report(path) for path in available_paths.values()]

    region_records = merge_region_records(parsed)
    pair_records = merge_pair_records(parsed)

    experiments_with_data = {experiment_id for (experiment_id, _feature) in region_records}
    missing_regions = [eid for eid in sorted_ids if eid not in experiments_with_data]
    if missing_regions:
        raise ComparisonError(
            f"No canonical Step9G result found for experiment(s): {missing_regions}. "
            "A multi-region comparison requires at least one valid canonical "
            "Step9G pair result per selected region."
        )

    input_hashes = {p["pair_id"]: p["report_sha256"] for p in parsed}
    analysis_id = build_analysis_id(sorted_ids, input_hashes)

    long_rows = _long_rows(sorted_ids, region_records)
    wide_rows = _wide_rows(sorted_ids, region_records)
    pairwise_rows = _pairwise_rows(available_pairs, pair_records)
    advisor_critical = advisor_critical_summary(sorted_ids, region_records)
    pairwise_findings = pairwise_reversal_findings(available_pairs, pair_records)

    manifest = {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "requested_experiment_ids": list(requested_ids),
        "resolved_experiment_ids": list(sorted_ids),
        "discovered_pair_report_paths": {f"{a}__{b}": str(available_paths[(a, b)]) for a, b in available_pairs},
        "unavailable_pair_reports": [f"{a}__{b}" for a, b in missing_pairs],
        "complete_pairwise_matrix": len(missing_pairs) == 0,
        "input_report_sha256": input_hashes,
        "underlying_step9g_analysis_ids": {p["pair_id"]: p["underlying_analysis_id"] for p in parsed},
        "report_schema_versions": {p["pair_id"]: p["report_schema_version"] for p in parsed},
        "scientific_contract": scientific_contract_summary(),
        "package_versions": _package_versions(),
        "output_schema_version": COMPARISON_SCHEMA_VERSION,
    }

    return {
        "ran": True,
        "dry_run": False,
        "force": force,
        "requested_experiment_ids": list(requested_ids),
        "resolved_experiment_ids": list(sorted_ids),
        "available_pairs": [f"{a}__{b}" for a, b in available_pairs],
        "missing_pairs": [f"{a}__{b}" for a, b in missing_pairs],
        "complete_pairwise_matrix": len(missing_pairs) == 0,
        "output_dir": output_dir,
        "manifest": manifest,
        "long_rows": long_rows,
        "wide_rows": wide_rows,
        "pairwise_rows": pairwise_rows,
        "advisor_critical": advisor_critical,
        "pairwise_findings": {f"{a}__{b}": features for (a, b), features in pairwise_findings.items()},
    }


def _existing_manifest_analysis_id(output_dir: Path) -> Optional[str]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        import json
        return json.loads(manifest_path.read_text()).get("analysis_id")
    except (OSError, ValueError):
        return None


def run_comparison(experiments: list[str], dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Public entry point: compute the comparison (or dry-run plan) and, for
    a real run, write outputs under the generic comparison namespace. Guards
    against silently overwriting a previous run produced by a DIFFERENT
    analysis_id unless --force is given."""
    result = build_comparison(experiments, dry_run=dry_run, force=force)
    if dry_run:
        return result

    output_dir = result["output_dir"]
    existing_id = _existing_manifest_analysis_id(output_dir)
    new_id = result["manifest"]["analysis_id"]
    if existing_id is not None and existing_id != new_id and not force:
        raise ComparisonError(
            f"concept-shift-compare: existing output at {output_dir} was produced by a "
            f"different analysis_id ({existing_id} != {new_id}). Use --force to overwrite."
        )

    from .render import write_outputs
    output_paths = write_outputs(result)

    return {
        "ran": True,
        "dry_run": False,
        "requested_experiment_ids": result["requested_experiment_ids"],
        "resolved_experiment_ids": result["resolved_experiment_ids"],
        "available_pairs": result["available_pairs"],
        "missing_pairs": result["missing_pairs"],
        "complete_pairwise_matrix": result["complete_pairwise_matrix"],
        "analysis_id": new_id,
        "output_dir": str(output_dir),
        "output_paths": output_paths,
    }
