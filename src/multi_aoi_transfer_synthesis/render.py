"""Renders the normalized multi-AOI transfer synthesis object (produced by
`build.build_synthesis`) into the required output files:

    multi_aoi_transfer_synthesis.json
    multi_aoi_transfer_synthesis.md
    multi_aoi_transfer_matrix.csv
    multi_aoi_feature_stability.csv
    multi_aoi_within_region.csv

This module contains NO AOI-specific conditional branches -- only generic
loops over the normalized representation's lists of dicts. It does not
resolve or adapt any frozen input; it only formats what `build.py` already
assembled.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.multi_aoi_transfer_synthesis import status_derivation

WITHIN_REGION_COLUMNS = [
    "experiment_id", "model_family", "roc_auc", "pr_auc",
    "thermal_minus_baseline_roc_auc", "thermal_minus_baseline_pr_auc",
    "large_block_robustness_available", "large_block_support_status",
    "effect_magnitude_stability_status", "large_block_unavailable_reason",
]

TRANSFER_MATRIX_COLUMNS = [
    "source_experiment_id", "target_experiment_id", "model_family", "adaptation_method",
    "primary_population", "target_row_count", "target_burned_count",
    "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "roc_auc_chance_status",
    "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
    "raw_transfer_status",
    "adapted_minus_raw", "adapted_minus_raw_ci_low", "adapted_minus_raw_ci_high",
    "adapted_minus_raw_support_status",
    "within_region_target_metric", "remaining_transfer_gap",
    "remaining_transfer_gap_ci_low", "remaining_transfer_gap_ci_high",
    "residual_gap_status",
    "step9e_shift_categories", "step9e_ranking_reversal_value", "step9e_ranking_reversal_scope",
    "brier", "brier_availability", "brier_protocol_source",
]

FEATURE_STABILITY_COLUMNS = [
    "experiment_a", "experiment_b", "feature",
    "experiment_a_auc", "experiment_a_ci_low", "experiment_a_ci_high", "experiment_a_direction",
    "experiment_b_auc", "experiment_b_ci_low", "experiment_b_ci_high", "experiment_b_direction",
    "point_direction_reversal", "reversal_status", "step9e_relationship_direction_flag",
]


# Explicit, unambiguous prose for each closed-vocabulary status label --
# never collapsed into a bare "N rows are supported" statement, which does
# not say supported-relative-to-what.
_ADAPTED_MINUS_RAW_STATUS_PHRASES = {
    status_derivation.ADAPTATION_SUPPORTED:
        "have bootstrap-supported ROC-AUC improvement over raw transfer",
    status_derivation.ADAPTATION_DEGRADED_TRANSFER:
        "have bootstrap-supported ROC-AUC degradation relative to raw transfer",
    status_derivation.ADAPTATION_EFFECT_UNCERTAIN:
        "have an adapted-minus-raw ROC-AUC effect that is not bootstrap-resolved (CI crosses zero)",
}

_ROC_AUC_CHANCE_STATUS_PHRASES = {
    status_derivation.ROC_AUC_CHANCE_STATUS_ABOVE:
        "have ROC-AUC bootstrap-supported above chance (0.5)",
    status_derivation.ROC_AUC_CHANCE_STATUS_BELOW:
        "have ROC-AUC bootstrap-supported below chance (0.5)",
    status_derivation.ROC_AUC_CHANCE_STATUS_NOT_EXCLUDED:
        "have ROC-AUC for which chance level (0.5) is not excluded",
    status_derivation.ROC_AUC_CHANCE_STATUS_UNAVAILABLE:
        "have an unavailable ROC-AUC chance-level classification",
}

# Recovery-pattern labels describe the bootstrap-supported adapted-minus-raw
# change only -- they say nothing about whether the adapted ROC-AUC itself
# clears chance. The internal enum values (status_derivation) are kept as-is
# for backward compatibility; only the rendered prose is clarified here.
_RECOVERY_PATTERN_STATUS_PHRASES = {
    status_derivation.BIDIRECTIONAL_RECOVERY: "bidirectional adapted-minus-raw recovery",
    status_derivation.DIRECTION_DEPENDENT_RECOVERY: "direction-dependent adapted-minus-raw recovery",
    status_derivation.NO_RECOVERY_SUPPORTED: "no adapted-minus-raw recovery supported",
}


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(row.get(col)) for col in columns})


def render_json(synthesis: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, default=str, sort_keys=False)
        f.write("\n")


def render_within_region_csv(synthesis: dict, path: Path) -> None:
    _write_csv(path, WITHIN_REGION_COLUMNS, synthesis["within_region"])


def render_transfer_matrix_csv(synthesis: dict, path: Path) -> None:
    _write_csv(path, TRANSFER_MATRIX_COLUMNS, synthesis["transfer_matrix"])


def render_feature_stability_csv(synthesis: dict, path: Path) -> None:
    _write_csv(path, FEATURE_STABILITY_COLUMNS, synthesis["feature_stability"])


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_markdown(synthesis: dict) -> str:
    lines: list[str] = []
    lines.append("# Multi-AOI Transfer Synthesis")
    lines.append("")
    lines.append(f"Generated at: {synthesis['generated_at']}")
    lines.append(f"Canonical AOI set id: `{synthesis['canonical_set_id']}`")
    lines.append(f"Primary population: `{synthesis['primary_population']}`")
    lines.append("")
    lines.append("This report is generated purely from already-frozen numeric outputs. "
                 "No modeling, adaptation, prediction, or bootstrap computation was performed "
                 "to produce this document.")
    lines.append("")

    lines.append("## Selected AOIs")
    lines.append("")
    lines.append(f"- Display order: {', '.join(synthesis['aois_display_order'])}")
    lines.append(f"- Canonical order: {', '.join(synthesis['aois_canonical_order'])}")
    lines.append("")

    lines.append("## Within-region performance")
    lines.append("")
    lines.append("| experiment_id | model_family | roc_auc | pr_auc | thermal_minus_baseline_roc_auc | "
                 "large_block_robustness_available | large_block_support_status | "
                 "effect_magnitude_stability_status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in synthesis["within_region"]:
        lines.append(
            f"| {row['experiment_id']} | {row['model_family']} | {_fmt(row['roc_auc'])} | "
            f"{_fmt(row['pr_auc'])} | {_fmt(row['thermal_minus_baseline_roc_auc'])} | "
            f"{row['large_block_robustness_available']} | {_fmt(row['large_block_support_status'])} | "
            f"{_fmt(row['effect_magnitude_stability_status'])} |"
        )
    lines.append("")

    lines.append("## Cross-region transfer matrix")
    lines.append("")
    lines.append("| direction | model_family | adaptation_method | roc_auc | roc_auc_chance_status | pr_auc | "
                 "raw_transfer_status | adapted_minus_raw | adapted_minus_raw_support_status | "
                 "residual_gap_status | step9e_shift_categories | step9e_ranking_reversal_value | "
                 "step9e_ranking_reversal_scope |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in synthesis["transfer_matrix"]:
        direction = f"{row['source_experiment_id']} -> {row['target_experiment_id']}"
        lines.append(
            f"| {direction} | {row['model_family']} | {row['adaptation_method']} | "
            f"{_fmt(row['roc_auc'])} | {_fmt(row['roc_auc_chance_status'])} | {_fmt(row['pr_auc'])} | "
            f"{_fmt(row['raw_transfer_status'])} | "
            f"{_fmt(row['adapted_minus_raw'])} | {_fmt(row['adapted_minus_raw_support_status'])} | "
            f"{_fmt(row['residual_gap_status'])} | {_fmt(row['step9e_shift_categories'])} | "
            f"{_fmt(row['step9e_ranking_reversal_value'])} | {_fmt(row['step9e_ranking_reversal_scope'])} |"
        )
    lines.append("")

    lines.append("## Feature-level direction-reversal stability")
    lines.append("")
    lines.append("| pair | feature | experiment_a_auc | experiment_b_auc | "
                 "point_direction_reversal | reversal_status |")
    lines.append("|---|---|---|---|---|---|")
    for row in synthesis["feature_stability"]:
        pair = f"{row['experiment_a']} / {row['experiment_b']}"
        lines.append(
            f"| {pair} | {row['feature']} | {_fmt(row['experiment_a_auc'])} | "
            f"{_fmt(row['experiment_b_auc'])} | {row['point_direction_reversal']} | "
            f"{row['reversal_status']} |"
        )
    lines.append("")

    summaries = synthesis["summaries"]
    lines.append("## Result-driven summary statements")
    lines.append("")

    strongest = summaries.get("strongest_raw_thermal_direction")
    if strongest:
        lines.append(
            f"- Strongest raw (unadapted) thermal ROC-AUC direction: "
            f"{strongest['source_experiment_id']} -> {strongest['target_experiment_id']} "
            f"(ROC-AUC = {_fmt(strongest['roc_auc'])})."
        )
    weakest = summaries.get("weakest_raw_thermal_direction")
    if weakest:
        lines.append(
            f"- Weakest raw (unadapted) thermal ROC-AUC direction: "
            f"{weakest['source_experiment_id']} -> {weakest['target_experiment_id']} "
            f"(ROC-AUC = {_fmt(weakest['roc_auc'])})."
        )
    below_chance = summaries.get("below_chance_raw_thermal_directions") or []
    if below_chance:
        rendered = ", ".join(f"{d['source_experiment_id']} -> {d['target_experiment_id']}" for d in below_chance)
        lines.append(f"- Directions where raw thermal discrimination was not bootstrap-supported above chance: {rendered}.")
    else:
        lines.append("- No direction showed raw thermal discrimination unsupported above chance.")

    bidirectional_pairs = summaries.get("bidirectional_raw_thermal_support_pairs") or []
    if bidirectional_pairs:
        rendered = ", ".join(f"{p[0]}/{p[1]}" for p in bidirectional_pairs)
        lines.append(f"- Pairs with bidirectional raw thermal ROC-AUC support: {rendered}.")
    else:
        lines.append("- No pair showed bidirectional raw thermal ROC-AUC support.")

    # Kept as two separate loops (never merged into one "supported" count):
    # adapted-minus-raw support is a claim about improvement RELATIVE TO raw
    # transfer; roc_auc_chance_status is a claim about the row's own
    # performance relative to chance. A row can satisfy one without the
    # other, so a method must never be called "successful transfer" from
    # adapted-minus-raw alone.
    for label, value in (summaries.get("adapted_minus_raw_support_status_counts") or {}).items():
        phrase = _ADAPTED_MINUS_RAW_STATUS_PHRASES.get(label, f"have adapted-minus-raw status `{label}`")
        lines.append(f"- {value} row(s) {phrase}.")

    # Filtered to adaptation_method != "raw_source_only" -- a raw_source_only
    # row is not an adapted model, so it must never be described as one.
    for label, value in (summaries.get("adapted_rows_roc_auc_chance_status_counts") or {}).items():
        phrase = _ROC_AUC_CHANCE_STATUS_PHRASES.get(label, f"have roc_auc_chance_status `{label}`")
        lines.append(f"- {value} adapted-transfer row(s) {phrase}.")

    if summaries.get("recovery_patterns"):
        lines.append(
            "- Recovery patterns describe the bootstrap-supported change relative to raw "
            "transfer, not whether the adapted ROC-AUC itself is above chance."
        )
    for pattern in summaries.get("recovery_patterns") or []:
        status = pattern["recovery_pattern_status"]
        phrase = _RECOVERY_PATTERN_STATUS_PHRASES.get(status, f"recovery pattern `{status}`")
        lines.append(
            f"- {pattern['experiment_a']}/{pattern['experiment_b']} "
            f"({pattern['model_family']}, {pattern['adaptation_method']}): {phrase}."
        )

    reversal_pairs = summaries.get("bootstrap_supported_feature_reversal_pairs") or []
    if reversal_pairs:
        rendered = ", ".join(f"{p[0]}/{p[1]}" for p in reversal_pairs)
        lines.append(f"- Pairs with at least one bootstrap-supported feature-level direction reversal: {rendered}.")
    else:
        lines.append("- No pair showed a bootstrap-supported feature-level direction reversal.")

    residual_rows = summaries.get("residual_gap_remaining_rows") or []
    if residual_rows:
        lines.append(f"- {len(residual_rows)} transfer-matrix row(s) show a bootstrap-supported remaining residual gap after adaptation.")
    else:
        lines.append("- No transfer-matrix row shows a bootstrap-supported remaining residual gap after adaptation.")

    roc_range = summaries.get("raw_thermal_roc_auc_range_across_directions")
    if roc_range is not None:
        lines.append(f"- Range of raw thermal ROC-AUC across all resolved directions: {_fmt(roc_range)}.")
    lines.append("")

    lines.append("## Scientific boundaries")
    lines.append("")
    for statement in synthesis["scientific_boundaries"]:
        lines.append(f"- {statement}")
    lines.append("")

    return "\n".join(lines)


def render_all(synthesis: dict, output_dir: Path) -> dict[str, Path]:
    """Write all 4 report files (JSON/MD + 3 CSVs) into `output_dir` and
    return {basename: path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "multi_aoi_transfer_synthesis.json": output_dir / "multi_aoi_transfer_synthesis.json",
        "multi_aoi_transfer_synthesis.md": output_dir / "multi_aoi_transfer_synthesis.md",
        "multi_aoi_transfer_matrix.csv": output_dir / "multi_aoi_transfer_matrix.csv",
        "multi_aoi_feature_stability.csv": output_dir / "multi_aoi_feature_stability.csv",
        "multi_aoi_within_region.csv": output_dir / "multi_aoi_within_region.csv",
    }
    render_json(synthesis, paths["multi_aoi_transfer_synthesis.json"])
    with open(paths["multi_aoi_transfer_synthesis.md"], "w", encoding="utf-8") as f:
        f.write(render_markdown(synthesis))
    render_transfer_matrix_csv(synthesis, paths["multi_aoi_transfer_matrix.csv"])
    render_feature_stability_csv(synthesis, paths["multi_aoi_feature_stability.csv"])
    render_within_region_csv(synthesis, paths["multi_aoi_within_region.csv"])
    return paths
