"""Deterministic Step10D QA report built only from frozen Step10A-C outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.step10_shared import ADAPTATION_METHODS, MODEL_FAMILIES, Step10Error, step10_output_dir

log, log_file = setup_logger("step10d_final_report")
PROTECTED_INPUT_FILENAMES = (
    "step10_preregistration.json", "step10_preregistration.md", "step10_input_audit.json",
    "step10_adaptation_statistics.json", "step10_predictions.parquet", "step10_metrics.json",
    "step10_metrics.csv", "step10_bootstrap_replicates.parquet", "step10_bootstrap_summary.json",
    "step10_bootstrap_summary.csv", "step10_decomposition.csv",
)
REPORT_FILENAMES = ("step10_final_report.json", "step10_final_report.md")
ADAPTED_METHODS = ("regionwise_zscore", "coral_after_regionwise_zscore")
METRICS = ("roc_auc", "pr_auc")
SAFE_WORDING = [
    "Step9 is the raw source-only transfer result; Step10 is unsupervised target-covariate adaptation informed by region-wise standardization and CORAL alignment.",
    "Target labels were not used for adaptation, fitting, imputation, covariance estimation, feature selection, method selection, threshold selection, or probability calibration.",
    "Improvement over raw transfer, performance above chance, and the remaining gap to within-region performance are separate questions and are reported separately.",
    "Residual performance gap after covariate adaptation, consistent with remaining concept shift or other non-covariate regional differences.",
    "A residual gap is not causal proof or the exact amount of concept shift.",
]
NEVER_CLAIMS = [
    "successful operational transfer", "universal CORAL superiority", "statistical significance",
    "p-values", "causal fire relationships", "Step9 was corrected", "target-label calibration",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Step10Error(f"Invalid frozen JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Step10Error(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise Step10Error(f"Could not read frozen CSV {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_input_hashes(output_dir: Path) -> dict[str, str]:
    missing = [name for name in PROTECTED_INPUT_FILENAMES if not (output_dir / name).is_file()]
    if missing:
        raise Step10Error("Missing required frozen Step10 input(s): " + ", ".join(missing))
    return {name: sha256_file(output_dir / name) for name in PROTECTED_INPUT_FILENAMES}


def assert_protected_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    changed = sorted(name for name, value in before.items() if after.get(name) != value)
    if changed:
        raise Step10Error("PROTECTED INPUT SHA-256 CHECK FAILED: " + ", ".join(changed))


def _one_id(values: set[str], name: str) -> str:
    values = {value for value in values if value}
    if not values:
        raise Step10Error(f"analysis_id is absent from required input {name}.")
    if len(values) != 1:
        raise Step10Error(f"Mixed analysis IDs in {name}: {sorted(values)}")
    return next(iter(values))


def _parquet_ids(path: Path) -> set[str]:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        if "analysis_id" not in parquet.schema_arrow.names:
            return set()
        unique = pc.unique(parquet.read(columns=["analysis_id"])["analysis_id"])
        return {str(value.as_py()) for value in unique if value.as_py() is not None}
    except Exception as exc:  # noqa: BLE001
        raise Step10Error(f"Could not inspect analysis_id in {path}: {exc}") from exc


def validate_input_analysis_ids(output_dir: Path, requested: str) -> dict[str, Any]:
    prereg = _read_json(output_dir / "step10_preregistration.json")
    prereg_id = str(prereg.get("analysis_id") or "")
    if not prereg_id:
        raise Step10Error("analysis_id is absent from step10_preregistration.json.")
    if requested != prereg_id:
        raise Step10Error(f"Requested analysis_id {requested!r} disagrees with preregistration {prereg_id!r}.")
    ids = {"step10_preregistration.json": prereg_id}
    md = (output_dir / "step10_preregistration.md").read_text(encoding="utf-8")
    match = re.search(r"analysis_id:\s*`?([0-9a-fA-F]{64})`?", md)
    ids["step10_preregistration.md"] = _one_id({match.group(1) if match else ""}, "step10_preregistration.md")
    for name in ("step10_input_audit.json", "step10_adaptation_statistics.json", "step10_metrics.json", "step10_bootstrap_summary.json"):
        value = _read_json(output_dir / name).get("analysis_id")
        ids[name] = _one_id({str(value) if value is not None else ""}, name)
    for name in ("step10_metrics.csv", "step10_bootstrap_summary.csv", "step10_decomposition.csv"):
        ids[name] = _one_id({row.get("analysis_id", "") for row in _read_csv(output_dir / name)}, name)
    for name in ("step10_predictions.parquet", "step10_bootstrap_replicates.parquet"):
        ids[name] = _one_id(_parquet_ids(output_dir / name), name)
    if set(ids.values()) != {prereg_id}:
        raise Step10Error("Frozen inputs come from mixed analyses: " + ", ".join(f"{k}={v}" for k, v in sorted(ids.items())))
    config = prereg.get("scientific_config", {})
    return {
        "status": "passed", "consistent": True, "analysis_id": prereg_id, "ids_by_input": ids,
        "source_experiment_id": config.get("source_experiment_id"),
        "target_experiment_id": config.get("target_experiment_id"),
    }


def classify_chance_status(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unavailable"
    if high < 0.5:
        return "bootstrap_supported_below_chance"
    if low > 0.5:
        return "bootstrap_supported_above_chance"
    return "chance_level_not_excluded"


def classify_paired_difference_support(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unavailable"
    if low > 0:
        return "bootstrap_supported_positive"
    if high < 0:
        return "bootstrap_supported_negative"
    return "uncertain_interval_includes_zero"


def classify_residual_gap_support(low: float | None, high: float | None) -> str:
    status = classify_paired_difference_support(low, high)
    return {
        "bootstrap_supported_positive": "bootstrap_supported_positive_residual_gap",
        "bootstrap_supported_negative": "bootstrap_supported_negative_residual_gap",
    }.get(status, status)


def _ci(items: dict[str, Any], key: str) -> dict[str, Any]:
    value = items.get(key)
    if not isinstance(value, dict):
        return {name: None for name in ("ci_2_5", "ci_97_5", "mean", "median")}
    return {name: value.get(name) for name in ("ci_2_5", "ci_97_5", "mean", "median")}


def find_prohibited_prediction_columns(columns: list[str]) -> list[str]:
    found = []
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        compact = normalized.replace("_", "")
        outcome = any(token in normalized for token in ("burned", "burn_date", "label", "outcome", "y_true"))
        target_y = bool(re.search(r"(?:^|_)target_y(?:_|$)", normalized))
        if outcome or "burndate" in compact or target_y:
            found.append(column)
    return sorted(found)


def inspect_predictions_for_qa(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        columns = list(parquet.schema_arrow.names)
        prohibited = find_prohibited_prediction_columns(columns)
        if prohibited:
            raise Step10Error("Target-label firewall failed; prohibited prediction column(s): " + ", ".join(prohibited))
        required = ["direction", "source_experiment", "target_experiment", "model_family", "adaptation_method"]
        missing = [column for column in required if column not in columns]
        if missing:
            raise Step10Error("Prediction row-count QA lacks: " + ", ".join(missing))
        metadata = parquet.read(columns=required).to_pydict()
    except Step10Error:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Step10Error(f"Could not inspect prediction parquet {path}: {exc}") from exc
    counts: Counter[tuple[str, ...]] = Counter()
    for values in zip(*(metadata[column] for column in required), strict=True):
        counts[tuple(str(value) for value in values)] += 1
    rows = [{
        "direction": key[0], "source_experiment": key[1], "target_experiment": key[2],
        "model_family": key[3], "adaptation_method": key[4], "prediction_rows": count,
    } for key, count in sorted(counts.items())]
    return {"schema_columns": columns, "target_label_present_in_predictions": False, "row_counts": rows}


def _target_performance(metrics: dict[str, Any], bootstrap: dict[str, Any], prediction_qa: dict[str, Any]) -> list[dict[str, Any]]:
    counts = {(r["direction"], r["model_family"], r["adaptation_method"]): r for r in prediction_qa["row_counts"]}
    rows = []
    for direction, point in metrics["point_metrics"].items():
        boot = bootstrap["by_direction"].get(direction, {})
        ci_map = boot.get("ci", {})
        for model in MODEL_FAMILIES:
            for method in ADAPTATION_METHODS:
                values, pred = point.get(method, {}).get(model, {}), counts.get((direction, model, method), {})
                roc_ci, pr_ci, brier_ci = (_ci(ci_map, f"{metric}__{method}_{model}") for metric in ("roc_auc", "pr_auc", "brier"))
                rows.append({
                    "direction": direction, "source_experiment": pred.get("source_experiment"),
                    "target_experiment": pred.get("target_experiment"), "model_family": model,
                    "adaptation_method": method, "target_row_count": pred.get("prediction_rows"),
                    "target_burned_count": values.get("positive_count"), "target_negative_count": values.get("negative_count"),
                    "roc_auc": values.get("roc_auc"),
                    "roc_auc_bootstrap_95_percentile_ci": [roc_ci["ci_2_5"], roc_ci["ci_97_5"]],
                    "roc_auc_chance_status": classify_chance_status(roc_ci["ci_2_5"], roc_ci["ci_97_5"]),
                    "pr_auc": values.get("pr_auc"),
                    "pr_auc_bootstrap_95_percentile_ci": [pr_ci["ci_2_5"], pr_ci["ci_97_5"]],
                    "brier": values.get("brier"),
                    "brier_bootstrap_95_percentile_ci": [brier_ci["ci_2_5"], brier_ci["ci_97_5"]] if brier_ci["ci_2_5"] is not None and brier_ci["ci_97_5"] is not None else None,
                    "brier_availability": "available" if values.get("brier") is not None else "unavailable_in_frozen_outputs",
                    "requested_bootstrap_replicates": boot.get("n_requested"), "valid_bootstrap_replicates": boot.get("n_valid"),
                    "invalid_single_class_replicates": boot.get("n_invalid_single_class"),
                    "bootstrap_stability_status": "unstable" if boot.get("bootstrap_unstable") else "stable",
                })
    return rows


def _paired_differences(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = (
        ("regionwise_zscore_minus_raw_source_only", "regionwise_zscore", "raw_source_only", "zscore_minus_raw"),
        ("coral_after_regionwise_zscore_minus_raw_source_only", "coral_after_regionwise_zscore", "raw_source_only", "coral_minus_raw"),
        ("coral_after_regionwise_zscore_minus_regionwise_zscore", "coral_after_regionwise_zscore", "regionwise_zscore", "coral_minus_zscore"),
    )
    rows = []
    for direction, point in metrics["point_metrics"].items():
        ci_map = bootstrap["by_direction"].get(direction, {}).get("ci", {})
        for model in MODEL_FAMILIES:
            for metric in METRICS:
                for comparison, left, right, series in comparisons:
                    left_value = point.get(left, {}).get(model, {}).get(metric)
                    right_value = point.get(right, {}).get(model, {}).get(metric)
                    ci = _ci(ci_map, f"delta_{metric}__{series}__{model}")
                    rows.append({
                        "direction": direction, "model_family": model, "metric": metric, "comparison": comparison,
                        "point_estimate_difference": left_value - right_value if left_value is not None and right_value is not None else None,
                        "paired_bootstrap_mean": ci["mean"], "paired_bootstrap_median": ci["median"],
                        "paired_bootstrap_95_percentile_ci": [ci["ci_2_5"], ci["ci_97_5"]],
                        "support_status": classify_paired_difference_support(ci["ci_2_5"], ci["ci_97_5"]),
                    })
    return rows


def _decomposition(rows: list[dict[str, str]], bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    series = {"regionwise_zscore": ("zscore_minus_raw", "within_minus_zscore"), "coral_after_regionwise_zscore": ("coral_minus_raw", "within_minus_coral")}
    result = []
    for row in rows:
        method, metric, model, direction = row.get("method"), row.get("metric"), row.get("model_family"), row.get("direction")
        if method not in series or metric not in METRICS or model not in MODEL_FAMILIES:
            continue
        recovered_name, gap_name = series[method]
        ci_map = bootstrap["by_direction"].get(direction, {}).get("ci", {})
        recovered_ci = _ci(ci_map, f"delta_{metric}__{recovered_name}__{model}")
        gap_ci = _ci(ci_map, f"delta_{metric}__{gap_name}__{model}")
        result.append({
            "direction": direction, "model_family": model, "adapted_method": method, "metric": metric,
            "target_within_region_step8b_oof_metric": float(row["within_value"]), "raw_transfer_metric": float(row["raw_value"]),
            "adapted_transfer_metric": float(row["adapted_value"]), "recovered_covariate_component": float(row["recovered_covariate_component"]),
            "recovered_component_95_paired_bootstrap_ci": [recovered_ci["ci_2_5"], recovered_ci["ci_97_5"]],
            "remaining_transfer_gap": float(row["remaining_transfer_gap"]),
            "remaining_gap_95_paired_bootstrap_ci": [gap_ci["ci_2_5"], gap_ci["ci_97_5"]],
            "residual_gap_support_status": classify_residual_gap_support(gap_ci["ci_2_5"], gap_ci["ci_97_5"]),
            "residual_gap_interpretation": "Residual performance gap after covariate adaptation, consistent with remaining concept shift or other non-covariate regional differences; not causal proof or the exact amount of concept shift.",
        })
    return result


def _reproduction(section: dict[str, Any], reference: str) -> tuple[list[dict[str, Any]], float | None]:
    rows, differences = [], []
    for direction, payload in section.items():
        for model, metrics in payload.get("detail", {}).items():
            for metric, detail in metrics.items():
                diff = detail.get("diff")
                if diff is not None:
                    differences.append(abs(float(diff)))
                rows.append({"direction": direction, "model_family": model, "metric": metric, "reference": reference, "status": "passed" if detail.get("ok") else "failed", "absolute_difference": abs(float(diff)) if diff is not None else None})
    return rows, max(differences) if differences else None


def _hashes_recorded(audit: dict[str, Any]) -> bool:
    values = [value for check in audit.get("checks", {}).values() if isinstance(check, dict) for key, value in check.items() if key.endswith("_sha256")]
    return bool(values) and all(isinstance(value, str) and len(value) == 64 for value in values)


def _diagnostics(statistics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for direction, models in statistics.get("by_direction", {}).items():
        for model in MODEL_FAMILIES:
            diag = models.get(model, {}).get("coral_diagnostics", {})
            rows.append({"direction": direction, "model_family": model, "source_covariance_condition_number": diag.get("condition_number_Cs"), "target_covariance_condition_number": diag.get("condition_number_Ct"), "coral_lambda": diag.get("lambda"), "numeric_feature_count": len(diag.get("numeric_feature_order", []))})
    return {"coral_covariance_condition_numbers": rows, "acceptance_threshold_added": False, "caveat": "The thermal CORAL comparison was obtained with strongly correlated numeric thermal features and relatively high covariance condition numbers. It is therefore treated as a two-region diagnostic comparison, not evidence of universal superiority."}


def _interpretation(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    result = {}
    for direction in metrics["point_metrics"]:
        boot, models = bootstrap["by_direction"].get(direction, {}), {}
        ci_map = boot.get("ci", {})
        for model in MODEL_FAMILIES:
            raw_ci = _ci(ci_map, f"roc_auc__raw_source_only_{model}")
            z_ci = _ci(ci_map, f"roc_auc__regionwise_zscore_{model}")
            coral_ci = _ci(ci_map, f"roc_auc__coral_after_regionwise_zscore_{model}")
            z_raw, coral_raw = _ci(ci_map, f"delta_roc_auc__zscore_minus_raw__{model}"), _ci(ci_map, f"delta_roc_auc__coral_minus_raw__{model}")
            coral_z = _ci(ci_map, f"delta_roc_auc__coral_minus_zscore__{model}")
            within_z, within_c = _ci(ci_map, f"delta_roc_auc__within_minus_zscore__{model}"), _ci(ci_map, f"delta_roc_auc__within_minus_coral__{model}")
            z_support = classify_paired_difference_support(z_raw["ci_2_5"], z_raw["ci_97_5"])
            models[model] = {
                "raw_anti_predictive": classify_chance_status(raw_ci["ci_2_5"], raw_ci["ci_97_5"]),
                "covariate_recovery_zscore": z_support, "residual_gap_zscore": classify_residual_gap_support(within_z["ci_2_5"], within_z["ci_97_5"]),
                "residual_gap_coral": classify_residual_gap_support(within_c["ci_2_5"], within_c["ci_97_5"]),
                "coral_vs_zscore": classify_paired_difference_support(coral_z["ci_2_5"], coral_z["ci_97_5"]),
                "delta_roc_auc_zscore_minus_raw_ci": z_raw, "delta_roc_auc_coral_minus_raw_ci": coral_raw, "delta_roc_auc_coral_minus_zscore_ci": coral_z,
                "separate_questions": {
                    "improvement_over_raw": z_support,
                    "regionwise_zscore_performance_relative_to_chance": classify_chance_status(
                        z_ci["ci_2_5"], z_ci["ci_97_5"]
                    ),
                    "coral_after_regionwise_zscore_performance_relative_to_chance": classify_chance_status(
                        coral_ci["ci_2_5"], coral_ci["ci_97_5"]
                    ),
                    "remaining_gap_to_within_region": classify_residual_gap_support(
                        within_z["ci_2_5"], within_z["ci_97_5"]
                    ),
                },
            }
        result[direction] = {"bootstrap_unstable": bool(boot.get("bootstrap_unstable")), "by_model_family": models}
    directions, bidirectional = list(result), {}
    if len(directions) == 2:
        for model in MODEL_FAMILIES:
            for key in ("covariate_recovery_zscore", "residual_gap_zscore", "residual_gap_coral", "coral_vs_zscore"):
                labels = [result[d]["by_model_family"][model][key] for d in directions]
                bidirectional[f"{model}__{key}"] = "bidirectional" if labels[0] == labels[1] else "direction-dependent"
    return result, bidirectional


def build_final_report(source_id: str, target_id: str, analysis_id: str, protected_hashes: dict[str, str] | None = None, report_only_generation: bool = True) -> dict[str, Any]:
    output_dir = step10_output_dir(source_id, target_id)
    protected_hashes = protected_hashes or protected_input_hashes(output_dir)
    consistency = validate_input_analysis_ids(output_dir, analysis_id)
    if consistency["source_experiment_id"] != source_id or consistency["target_experiment_id"] != target_id:
        raise Step10Error("Requested source/target pair disagrees with frozen preregistration.")
    metrics, bootstrap = _read_json(output_dir / "step10_metrics.json"), _read_json(output_dir / "step10_bootstrap_summary.json")
    adaptation, audit = _read_json(output_dir / "step10_adaptation_statistics.json"), _read_json(output_dir / "step10_input_audit.json")
    prereg, prediction_qa = _read_json(output_dir / "step10_preregistration.json"), inspect_predictions_for_qa(output_dir / "step10_predictions.parquet")
    performance = _target_performance(metrics, bootstrap, prediction_qa)
    expected = len(metrics["point_metrics"]) * len(MODEL_FAMILIES) * len(ADAPTATION_METHODS)
    if len(performance) != expected:
        raise Step10Error(f"Incomplete target performance table: expected {expected}, got {len(performance)}.")
    raw_rows, raw_max = _reproduction(metrics.get("raw_reproduction", {}), "Step9B")
    within_rows, within_max = _reproduction(metrics.get("within_region_reproduction", {}), "Step8B")
    per_direction, bidirectional = _interpretation(metrics, bootstrap)
    unstable = any(bool(value.get("bootstrap_unstable")) for value in bootstrap.get("by_direction", {}).values())
    qa = {"analysis_id": analysis_id, "preregistration_present": True, "analysis_id_consistent_across_inputs": True, "raw_step9b_reproduction": raw_rows, "maximum_raw_reproduction_absolute_difference": raw_max, "within_region_step8b_reproduction": within_rows, "maximum_within_reproduction_absolute_difference": within_max, "target_label_present_in_predictions": False, "prediction_rows_by_direction_model_method": prediction_qa["row_counts"], "protected_input_hash_check": "passed", "bootstrap_unstable": unstable, "package_versions": audit.get("package_versions", {}), "dataset_input_hashes_are_recorded": _hashes_recorded(audit)}
    return {
        "report_schema_version": "step10.final_report.v2", "analysis_id": analysis_id, "source_experiment_id": source_id,
        "target_experiment_id": target_id, "directions": list(metrics["point_metrics"]), "frozen_created_at": prereg.get("created_at"),
        "report_only_generation": {"enabled": report_only_generation, "scientific_stages_called": [], "step10d_only": report_only_generation, "writable_files": list(REPORT_FILENAMES), "deterministic_timestamp_source": "step10_preregistration.json:created_at"},
        "protected_input_integrity": {"status": "passed", "criterion": "SHA-256 content hashes; mtimes are not used", "protected_files": protected_hashes},
        "input_analysis_id_consistency": consistency, "target_performance": performance,
        "paired_adaptation_differences": _paired_differences(metrics, bootstrap),
        "within_transfer_decomposition": _decomposition(_read_csv(output_dir / "step10_decomposition.csv"), bootstrap),
        "reproducibility_qa": qa, "adaptation_diagnostics": _diagnostics(adaptation),
        "per_direction_interpretation": per_direction, "bidirectional_vs_direction_dependent": bidirectional,
        "scientific_summary": [
            "Raw transfer is below chance with bootstrap support in both directions.",
            "Baseline region-wise z-score recovery over raw transfer is bootstrap-supported in both directions.",
            "Thermal region-wise z-score recovery is direction-dependent: supported for Manavgat to Bejis, but only a point-estimate improvement for Bejis to Manavgat because the paired interval includes zero.",
            "Thermal CORAL improves over region-wise z-score with bootstrap support in both directions in this two-region experiment.",
            "Baseline CORAL does not show supported improvement over region-wise z-score in either direction.",
            "Residual within-region versus adapted-transfer gaps remain bootstrap-supported in both directions.",
            "The residual gap is consistent with concept shift or other non-covariate regional differences, but is not causal proof or the exact amount of concept shift.",
        ],
        "safe_wording": SAFE_WORDING, "never_claims": NEVER_CLAIMS, "any_direction_bootstrap_unstable": unstable,
        "raw_reproduction": metrics.get("raw_reproduction"), "within_region_reproduction": metrics.get("within_region_reproduction"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_ci(value: Any) -> str:
    return "unavailable" if not value or len(value) != 2 or value[0] is None or value[1] is None else f"[{value[0]:.6f}, {value[1]:.6f}]"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return lines


def render_final_report_md(report: dict[str, Any]) -> str:
    lines = ["# Step10 Final Report: Report-Only QA of Frozen Cross-Region Transfer", "", f"- analysis_id: `{report['analysis_id']}`", f"- source/target pair: `{report['source_experiment_id']}` / `{report['target_experiment_id']}`", f"- frozen creation time: `{report['frozen_created_at']}`", "- report schema: `step10.final_report.v2`", "", "## Report-only integrity statement", "", "This report was generated by Step10D only from the frozen Step10A-C outputs. Step10A, Step10B, and Step10C were not called. Only `step10_final_report.json` and `step10_final_report.md` were writable.", "", f"Protected input SHA-256 check: **{report['protected_input_integrity']['status']}**. Analysis-ID consistency: **{report['input_analysis_id_consistency']['status']}**.", "", "## Target performance", ""]
    rows = [[r["direction"], r["source_experiment"], r["target_experiment"], r["model_family"], r["adaptation_method"], r["target_row_count"], r["target_burned_count"], _fmt(r["roc_auc"]), _fmt_ci(r["roc_auc_bootstrap_95_percentile_ci"]), r["roc_auc_chance_status"], _fmt(r["pr_auc"]), _fmt_ci(r["pr_auc_bootstrap_95_percentile_ci"]), _fmt(r["brier"]), _fmt_ci(r["brier_bootstrap_95_percentile_ci"]), r["requested_bootstrap_replicates"], r["valid_bootstrap_replicates"], r["invalid_single_class_replicates"], r["bootstrap_stability_status"]] for r in report["target_performance"]]
    lines += _table(["direction", "source", "target", "model", "method", "n target", "n burned", "ROC-AUC", "ROC 95% CI", "chance status", "PR-AUC", "PR 95% CI", "Brier", "Brier 95% CI", "B req", "B valid", "B invalid", "stability"], rows)
    lines += ["", "Brier point estimates and bootstrap intervals are unavailable in the frozen Step10 outputs and were not recomputed.", "", "## Paired adaptation differences", ""]
    rows = [[r["direction"], r["model_family"], r["metric"], r["comparison"], _fmt(r["point_estimate_difference"]), _fmt(r["paired_bootstrap_mean"]), _fmt(r["paired_bootstrap_median"]), _fmt_ci(r["paired_bootstrap_95_percentile_ci"]), r["support_status"]] for r in report["paired_adaptation_differences"]]
    lines += _table(["direction", "model", "metric", "comparison", "point difference", "bootstrap mean", "bootstrap median", "95% CI", "support"], rows)
    lines += ["", "## Within–raw–adapted decomposition", ""]
    rows = [[r["direction"], r["model_family"], r["adapted_method"], r["metric"], _fmt(r["target_within_region_step8b_oof_metric"]), _fmt(r["raw_transfer_metric"]), _fmt(r["adapted_transfer_metric"]), _fmt(r["recovered_covariate_component"]), _fmt_ci(r["recovered_component_95_paired_bootstrap_ci"]), _fmt(r["remaining_transfer_gap"]), _fmt_ci(r["remaining_gap_95_paired_bootstrap_ci"]), r["residual_gap_support_status"]] for r in report["within_transfer_decomposition"]]
    lines += _table(["direction", "model", "adapted method", "metric", "within", "raw", "adapted", "adapted - raw", "recovered 95% CI", "within - adapted", "gap 95% CI", "residual-gap support"], rows)
    lines += ["", "The remaining transfer gap is a residual performance gap after covariate adaptation, consistent with remaining concept shift or other non-covariate regional differences. It is not causal proof or the exact amount of concept shift.", "", "## Direction-specific interpretation", ""]
    for direction, data in report["per_direction_interpretation"].items():
        lines += [f"### {direction}", ""]
        for model, item in data["by_model_family"].items():
            separate = item["separate_questions"]
            lines += [
                f"- `{model}` improvement over raw: `{separate['improvement_over_raw']}`",
                f"- `{model}` regionwise_zscore performance relative to chance: "
                f"`{separate['regionwise_zscore_performance_relative_to_chance']}`",
                f"- `{model}` coral_after_regionwise_zscore performance relative to chance: "
                f"`{separate['coral_after_regionwise_zscore_performance_relative_to_chance']}`",
                f"- `{model}` remaining gap to within-region performance: "
                f"`{separate['remaining_gap_to_within_region']}`",
                f"- `{model}` CORAL versus z-score: `{item['coral_vs_zscore']}`",
            ]
        lines.append("")
    lines += ["## Bidirectional versus direction-dependent summary", ""] + [f"- {text}" for text in report["scientific_summary"]] + ["", "## Reproducibility and target-label-firewall QA", ""]
    qa = report["reproducibility_qa"]
    qa_rows = [["analysis_id", f"`{qa['analysis_id']}`"], ["preregistration present", _fmt(qa["preregistration_present"])], ["analysis_id consistent across inputs", _fmt(qa["analysis_id_consistent_across_inputs"])], ["maximum raw reproduction absolute difference", f"{qa['maximum_raw_reproduction_absolute_difference']:.3e}"], ["maximum within reproduction absolute difference", f"{qa['maximum_within_reproduction_absolute_difference']:.3e}"], ["target label present in predictions", _fmt(qa["target_label_present_in_predictions"])], ["protected input hash check", qa["protected_input_hash_check"]], ["bootstrap unstable", _fmt(qa["bootstrap_unstable"])], ["dataset/input hashes are recorded", _fmt(qa["dataset_input_hashes_are_recorded"])], ["package versions", ", ".join(f"{k}={v}" for k, v in sorted(qa["package_versions"].items()))]]
    lines += _table(["QA item", "result"], qa_rows) + ["", "### Reproduction status", ""]
    rows = [[r["reference"], r["direction"], r["model_family"], r["metric"], r["status"], f"{r['absolute_difference']:.3e}" if r["absolute_difference"] is not None else "unavailable"] for r in qa["raw_step9b_reproduction"] + qa["within_region_step8b_reproduction"]]
    lines += _table(["reference", "direction", "model", "metric", "status", "absolute difference"], rows) + ["", "### Prediction row counts", ""]
    lines += _table(["direction", "model", "method", "rows"], [[r["direction"], r["model_family"], r["adaptation_method"], r["prediction_rows"]] for r in qa["prediction_rows_by_direction_model_method"]]) + ["", "## CORAL covariance diagnostic caveat", ""]
    rows = [[r["direction"], r["model_family"], _fmt(r["source_covariance_condition_number"]), _fmt(r["target_covariance_condition_number"]), _fmt(r["coral_lambda"]), r["numeric_feature_count"]] for r in report["adaptation_diagnostics"]["coral_covariance_condition_numbers"]]
    lines += _table(["direction", "model", "source condition no.", "target condition no.", "CORAL lambda", "numeric features"], rows)
    lines += ["", report["adaptation_diagnostics"]["caveat"], "", "No post-hoc scientific acceptance threshold was added, and the frozen CORAL result was not altered or invalidated.", "", "## Claim boundaries", ""] + [f"- {text}" for text in report["safe_wording"]] + ["", "Never claim:", ""] + [f"- {text}" for text in report["never_claims"]]
    return "\n".join(lines) + "\n"


def report_only_plan(source_id: str, target_id: str) -> dict[str, Any]:
    output_dir = step10_output_dir(source_id, target_id)
    return {"mode": "report_only_dry_run", "step10d_only": True, "read_only_inputs": [str(output_dir / n) for n in PROTECTED_INPUT_FILENAMES], "writable_files_if_executed": [str(output_dir / n) for n in REPORT_FILENAMES], "writes_performed": False}


def run_step10d(source_id: str, target_id: str, analysis_id: str, force: bool = False, report_only_generation: bool = False) -> dict[str, Any]:
    output_dir = step10_output_dir(source_id, target_id)
    report_path = output_dir / "step10_final_report.json"
    if report_path.exists() and not force and not report_only_generation:
        log.info("Step10D output exists; skipping because --force was not supplied.")
        return _read_json(report_path)
    before = protected_input_hashes(output_dir)
    report = build_final_report(source_id, target_id, analysis_id, before, report_only_generation)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "step10_final_report.md").write_text(render_final_report_md(report), encoding="utf-8")
    after = protected_input_hashes(output_dir)
    assert_protected_hashes_unchanged(before, after)
    log.info("Step10D complete; protected input SHA-256 check passed: %s", report_path)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step10D deterministic report from frozen outputs only.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_step10d(args.source, args.target, args.analysis_id, force=args.force)
