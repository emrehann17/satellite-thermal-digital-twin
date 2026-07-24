"""
step9g_report_revision.py

Step9G REPORT-ONLY semantic revision (report schema v2) of the frozen v1
final report (step9g_final_report.json / .md), for an arbitrary experiment
pair, IN PLACE at the same canonical path Step9G v1 already writes to
(outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/
<a>__<b>/step9g_final_report.json).

WHAT THIS FIXES (report-layer only; no experiment ID is hard-coded)
--------------------------------------------------------------------
1. `integrated_interpretation` was one identical generic sentence assigned
   to EVERY feature row regardless of `reversal_status`. It is now
   generated per row from `reversal_status`.
2. `thermal_features_consistent_with_step9e` could include `elevation_mean`
   (a baseline, not thermal, feature). It is now restricted to the frozen
   six-feature thermal set. A new general `features_consistent_with_step9e`
   field is added, listing any feature (thermal or baseline) whose Step9G
   point-direction-reversal flag agrees with Step9E.

WHAT THIS DOES NOT DO
---------------------
Recomputes NOTHING numerical: AUC, CI, bootstrap draws, direction labels,
support_status, and reversal_status are read VERBATIM from the existing
report. Every row key except `integrated_interpretation` is asserted
value-identical before/after. `analysis_id` is preserved unchanged. The
original pre-revision report is preserved as a one-time backup file
alongside the revised one (never silently discarded).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.io_utils import setup_logger
import src.step9g_univariate_feature_auc_direction_reversal as step9g

log, log_file = setup_logger("step9g_report_revision")

REPORT_SCHEMA_VERSION_V2 = "step9g.univariate_feature_auc_direction_reversal.report.v2"

# Frozen thermal-feature set -- never includes elevation_mean/ndvi_mean/
# slope_mean/landcover_dominant (see task requirement).
THERMAL_FEATURES = (
    "lst_anomaly_mean",
    "current_lst_mean",
    "current_tvdi_mean",
    "tvdi_difference_mean",
    "downscaled_lst_mean",
    "fused_lst_mean",
)

REVERSAL_STATUS_INTERPRETATIONS = {
    "bootstrap_supported_direction_reversal": (
        "The point directions are opposite across the two regions, and each "
        "region's 95% spatial-block bootstrap interval lies on a different "
        "side of 0.5. This is a bootstrap-supported univariate direction "
        "reversal. It is consistent with feature-label relationship "
        "instability but does not establish causality or prove that this is "
        "the sole transfer-failure mechanism."
    ),
    "point_direction_reversal_interval_uncertain": (
        "The point AUC directions are opposite across the two regions, but "
        "at least one 95% spatial-block bootstrap interval includes 0.5. "
        "This is a point-estimate direction reversal with uncertain "
        "bootstrap support; it must not be reported as a supported reversal."
    ),
    "no_direction_reversal": (
        "The feature has the same point-AUC direction in both regions. No "
        "univariate direction reversal is observed for this feature. "
        "Individual regions may still have intervals that include 0.5."
    ),
}
UNAVAILABLE_INTERPRETATION = (
    "This feature's direction-reversal status could not be determined from "
    "the available Step9G outputs for this pair."
)

REPORT_REVISION_REASON = (
    "Corrects two report-layer semantic defects without recomputing any "
    "numerical result. (1) integrated_interpretation previously repeated one "
    "identical generic sentence for every feature row regardless of "
    "reversal_status; it is now generated per row from reversal_status. "
    "(2) thermal_features_consistent_with_step9e previously could include "
    "elevation_mean (a baseline, not thermal, feature); it is now restricted "
    "to the frozen thermal-feature set (lst_anomaly_mean, current_lst_mean, "
    "current_tvdi_mean, tvdi_difference_mean, downscaled_lst_mean, "
    "fused_lst_mean). A new general features_consistent_with_step9e field "
    "lists any feature (thermal or baseline) whose Step9G point-direction-"
    "reversal flag agrees with the Step9E relationship-direction diagnostic. "
    "AUC, CI, bootstrap draws, direction labels, support_status, and "
    "reversal_status are unchanged (numerical_results_unchanged=true); "
    "analysis_id is preserved verbatim."
)

BACKUP_JSON_NAME = "step9g_final_report.pre_report_v2_backup.json"
BACKUP_MD_NAME = "step9g_final_report.pre_report_v2_backup.md"


class Step9GReportRevisionError(SystemExit):
    """Fail-fast error for the Step9G report-only revision."""


def row_interpretation(reversal_status: Any) -> str:
    return REVERSAL_STATUS_INTERPRETATIONS.get(reversal_status, UNAVAILABLE_INTERPRETATION)


def _pair_report_root() -> Path:
    """`outputs/diagnostics/step9g_univariate_feature_auc_direction_reversal/`
    -- derived generically from the v1 module's own path helper (no
    experiment ID hard-coded)."""
    return step9g.output_root_for("a", "b").parent


def find_pair_dir(experiment_a: str, experiment_b: str) -> Optional[Path]:
    """Find whichever ordering of the pair directory actually exists on
    disk. Returns None if neither exists."""
    root = _pair_report_root()
    for a, b in ((experiment_a, experiment_b), (experiment_b, experiment_a)):
        candidate = root / f"{a}__{b}"
        if (candidate / "step9g_final_report.json").is_file():
            return candidate
    return None


def revise_report(
    source_id: str, target_id: str, dry_run: bool = False, force: bool = False,
) -> dict[str, Any]:
    if source_id == target_id:
        raise Step9GReportRevisionError("source_id and target_id must be different experiment IDs.")

    pair_dir = find_pair_dir(source_id, target_id)
    if pair_dir is None:
        raise Step9GReportRevisionError(
            f"No canonical Step9G v1 pair report found for ({source_id}, {target_id}) "
            f"under {_pair_report_root()}. Run 'concept-shift' first."
        )
    report_path = pair_dir / "step9g_final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    already_revised = report.get("report_schema_version") == REPORT_SCHEMA_VERSION_V2
    if already_revised and not force:
        return {
            "ran": False,
            "reason": "already_revised_use_force",
            "pair_dir": str(pair_dir),
            "analysis_id": report.get("analysis_id"),
        }

    original_analysis_id = report["analysis_id"]
    original_rows = report["direction_reversal_table"]

    revised_rows = []
    for row in original_rows:
        new_row = dict(row)
        new_row["integrated_interpretation"] = row_interpretation(row.get("reversal_status"))
        revised_rows.append(new_row)

    # Defensive: every key except integrated_interpretation must be
    # identical -- this is a REPORT-ONLY fix, never a numerical one.
    for old, new in zip(original_rows, revised_rows):
        for key in old:
            if key == "integrated_interpretation":
                continue
            if old[key] != new[key]:
                raise Step9GReportRevisionError(
                    f"Report revision would change field '{key}' for feature "
                    f"'{old.get('feature')}'; refusing (report-only fix must "
                    "never alter numerical results)."
                )

    features_consistent = [
        r["feature"] for r in revised_rows
        if r.get("step9e_relationship_direction_flag") and r.get("point_direction_reversal")
    ]
    thermal_features_consistent = [f for f in features_consistent if f in THERMAL_FEATURES]

    if dry_run:
        return {
            "ran": False,
            "dry_run": True,
            "pair_dir": str(pair_dir),
            "analysis_id": original_analysis_id,
            "features_consistent_with_step9e": features_consistent,
            "thermal_features_consistent_with_step9e": thermal_features_consistent,
        }

    revised_report = dict(report)
    revised_report["direction_reversal_table"] = revised_rows
    revised_answers = dict(report.get("answers", {}))
    revised_answers["features_consistent_with_step9e"] = features_consistent
    revised_answers["thermal_features_consistent_with_step9e"] = thermal_features_consistent
    revised_report["answers"] = revised_answers
    revised_report["analysis_id"] = original_analysis_id  # preserved verbatim, explicit
    revised_report["report_schema_version"] = REPORT_SCHEMA_VERSION_V2
    revised_report["report_revision_reason"] = REPORT_REVISION_REASON
    revised_report["numerical_results_unchanged"] = True
    revised_report["regenerated_at"] = datetime.now(timezone.utc).isoformat()

    backup_path = pair_dir / BACKUP_JSON_NAME
    if not backup_path.is_file():
        backup_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    report_path.write_text(json.dumps(revised_report, indent=2, default=str) + "\n", encoding="utf-8")

    md_path = pair_dir / "step9g_final_report.md"
    if md_path.is_file():
        md_backup_path = pair_dir / BACKUP_MD_NAME
        if not md_backup_path.is_file():
            md_backup_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        md_path.write_text(
            md_path.read_text(encoding="utf-8") + "\n" + _revision_note_md(revised_report),
            encoding="utf-8",
        )

    return {
        "ran": True,
        "pair_dir": str(pair_dir),
        "analysis_id": original_analysis_id,
        "report_schema_version": REPORT_SCHEMA_VERSION_V2,
        "features_consistent_with_step9e": features_consistent,
        "thermal_features_consistent_with_step9e": thermal_features_consistent,
        "numerical_results_unchanged": True,
    }


def _revision_note_md(revised_report: dict[str, Any]) -> str:
    return "\n".join([
        "## Report revision (report schema v2, report-only)",
        "",
        f"- report_schema_version: `{revised_report['report_schema_version']}`",
        f"- regenerated_at: {revised_report['regenerated_at']}",
        f"- numerical_results_unchanged: {revised_report['numerical_results_unchanged']}",
        "",
        revised_report["report_revision_reason"],
        "",
        f"- features_consistent_with_step9e: {revised_report['answers']['features_consistent_with_step9e']}",
        f"- thermal_features_consistent_with_step9e: {revised_report['answers']['thermal_features_consistent_with_step9e']}",
        "",
    ])
