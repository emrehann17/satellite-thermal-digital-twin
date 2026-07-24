"""Regression tests for the Step9G v1 final-report REPORT-ONLY semantic
revision (src/step9g_report_revision.py).

Uses entirely synthetic/placeholder experiment IDs and hand-built JSON
fixtures (matching this repo's existing convention), redirecting
PROJECT_ROOT so nothing here touches the real repo output tree."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.step9g_report_revision as revision
import src.step9g_univariate_feature_auc_direction_reversal as step9g

FAKE_A = "aoi_alpha_2099"
FAKE_B = "aoi_beta_2099"


def make_row(feature: str, reversal_status: str, point_reversal: bool, step9e_flag=True) -> dict:
    return {
        "feature": feature,
        "source_experiment_id": FAKE_A,
        "target_experiment_id": FAKE_B,
        f"{FAKE_A}_auc": 0.4,
        f"{FAKE_A}_ci_low": 0.3,
        f"{FAKE_A}_ci_high": 0.5,
        f"{FAKE_A}_direction": "lower_values_rank_burned",
        f"{FAKE_A}_support_status": "bootstrap_supported_lower_values_rank_burned",
        f"{FAKE_B}_auc": 0.6,
        f"{FAKE_B}_ci_low": 0.5,
        f"{FAKE_B}_ci_high": 0.7,
        f"{FAKE_B}_direction": "higher_values_rank_burned",
        f"{FAKE_B}_support_status": "bootstrap_supported_higher_values_rank_burned",
        "auc_difference_target_minus_source": 0.2,
        "auc_difference_ci_low": 0.1,
        "auc_difference_ci_high": 0.3,
        "point_direction_reversal": point_reversal,
        "reversal_status": reversal_status,
        "step9e_relationship_direction_flag": step9e_flag,
        "integrated_interpretation": "STALE GENERIC TEXT REPEATED FOR EVERY ROW",
    }


def write_v1_report(root: Path, pair_id: str, rows: list[dict]) -> Path:
    pair_dir = root / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_id": "fixed_frozen_analysis_id_123",
        "schema_version": "step9g.univariate_feature_auc_direction_reversal.v1",
        "source_experiment_id": FAKE_A,
        "target_experiment_id": FAKE_B,
        "primary_population": "burnable_tree_shrub_grass",
        "direction_reversal_table": rows,
        "bootstrap_supported_direction_reversals": [r["feature"] for r in rows if r["reversal_status"] == "bootstrap_supported_direction_reversal"],
        "point_reversals_interval_uncertain": [r["feature"] for r in rows if r["reversal_status"] == "point_direction_reversal_interval_uncertain"],
        "same_direction_features": [r["feature"] for r in rows if r["reversal_status"] == "no_direction_reversal"],
        "answers": {
            "thermal_features_consistent_with_step9e": [
                r["feature"] for r in rows if r["step9e_relationship_direction_flag"] and r["point_direction_reversal"]
            ],
        },
        "claim_boundary": "placeholder",
    }
    (pair_dir / "step9g_final_report.json").write_text(json.dumps(report, indent=2))
    (pair_dir / "step9g_final_report.md").write_text("# placeholder md\n")
    return pair_dir / "step9g_final_report.json"


@pytest.fixture(autouse=True)
def _redirect_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(step9g, "PROJECT_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# 11-13. Per-row interpretation wording
# ---------------------------------------------------------------------------
def test_no_direction_reversal_rows_receive_non_reversal_wording(tmp_path):
    rows = [make_row("elevation_mean", "no_direction_reversal", False)]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    assert result["ran"] is True
    report = json.loads((Path(result["pair_dir"]) / "step9g_final_report.json").read_text())
    text = report["direction_reversal_table"][0]["integrated_interpretation"]
    assert "same point-AUC direction in both regions" in text
    assert "No univariate direction reversal is observed" in text


def test_uncertain_point_reversals_not_called_supported(tmp_path):
    rows = [make_row("elevation_mean", "point_direction_reversal_interval_uncertain", True)]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    report = json.loads((Path(result["pair_dir"]) / "step9g_final_report.json").read_text())
    text = report["direction_reversal_table"][0]["integrated_interpretation"]
    assert "uncertain bootstrap support" in text
    assert "must not be reported as a supported reversal" in text
    assert "bootstrap-supported univariate direction reversal" not in text


def test_supported_reversals_receive_supported_reversal_wording(tmp_path):
    rows = [make_row("elevation_mean", "bootstrap_supported_direction_reversal", True)]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    report = json.loads((Path(result["pair_dir"]) / "step9g_final_report.json").read_text())
    text = report["direction_reversal_table"][0]["integrated_interpretation"]
    assert "bootstrap-supported univariate direction reversal" in text
    assert "does not establish causality" in text


def test_never_states_all_rows_indicate_reversal(tmp_path):
    rows = [
        make_row("elevation_mean", "bootstrap_supported_direction_reversal", True),
        make_row("ndvi_mean", "no_direction_reversal", False),
        make_row("slope_mean", "point_direction_reversal_interval_uncertain", True),
    ]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    report = json.loads((Path(result["pair_dir"]) / "step9g_final_report.json").read_text())
    texts = {row["feature"]: row["integrated_interpretation"] for row in report["direction_reversal_table"]}
    assert texts["elevation_mean"] != texts["ndvi_mean"]
    assert texts["ndvi_mean"] != texts["slope_mean"]
    assert "No univariate direction reversal is observed" in texts["ndvi_mean"]


# ---------------------------------------------------------------------------
# 14. elevation_mean never appears in thermal_features_consistent_with_step9e
# ---------------------------------------------------------------------------
def test_elevation_mean_never_in_thermal_list(tmp_path):
    rows = [
        make_row("elevation_mean", "bootstrap_supported_direction_reversal", True, step9e_flag=True),
        make_row("current_lst_mean", "bootstrap_supported_direction_reversal", True, step9e_flag=True),
    ]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    assert "elevation_mean" not in result["thermal_features_consistent_with_step9e"]
    assert "current_lst_mean" in result["thermal_features_consistent_with_step9e"]
    assert "elevation_mean" in result["features_consistent_with_step9e"]  # general list retains it


# ---------------------------------------------------------------------------
# 15. Numerical fields unchanged during report regeneration
# ---------------------------------------------------------------------------
def test_numerical_fields_unchanged_after_revision(tmp_path):
    rows = [
        make_row("elevation_mean", "bootstrap_supported_direction_reversal", True),
        make_row("ndvi_mean", "no_direction_reversal", False),
    ]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    original = json.loads((tmp_path / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / f"{FAKE_A}__{FAKE_B}" / "step9g_final_report.json").read_text())
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    revised = json.loads((Path(result["pair_dir"]) / "step9g_final_report.json").read_text())

    assert revised["analysis_id"] == original["analysis_id"]
    for orig_row, new_row in zip(original["direction_reversal_table"], revised["direction_reversal_table"]):
        for key in orig_row:
            if key == "integrated_interpretation":
                continue
            assert orig_row[key] == new_row[key], f"field '{key}' changed"


# ---------------------------------------------------------------------------
# Dry-run / idempotency / backup
# ---------------------------------------------------------------------------
def test_dry_run_writes_no_files(tmp_path):
    rows = [make_row("elevation_mean", "bootstrap_supported_direction_reversal", True)]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    result = revision.revise_report(FAKE_A, FAKE_B, dry_run=True)
    assert result["ran"] is False
    assert result["dry_run"] is True
    pair_dir = tmp_path / "outputs" / "diagnostics" / "step9g_univariate_feature_auc_direction_reversal" / f"{FAKE_A}__{FAKE_B}"
    assert not (pair_dir / revision.BACKUP_JSON_NAME).is_file()


def test_backup_created_and_preserved_on_rerun(tmp_path):
    rows = [make_row("elevation_mean", "bootstrap_supported_direction_reversal", True)]
    report_path = write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    pair_dir = report_path.parent

    revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    backup_content_first = (pair_dir / revision.BACKUP_JSON_NAME).read_text()

    # Re-running with force must not overwrite the ORIGINAL backup.
    revision.revise_report(FAKE_A, FAKE_B, dry_run=False, force=True)
    backup_content_second = (pair_dir / revision.BACKUP_JSON_NAME).read_text()
    assert backup_content_first == backup_content_second


def test_already_revised_without_force_is_noop(tmp_path):
    rows = [make_row("elevation_mean", "bootstrap_supported_direction_reversal", True)]
    write_v1_report(tmp_path, f"{FAKE_A}__{FAKE_B}", rows)
    revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
    second = revision.revise_report(FAKE_A, FAKE_B, dry_run=False, force=False)
    assert second["ran"] is False
    assert second["reason"] == "already_revised_use_force"


def test_missing_pair_report_fails_clearly(tmp_path):
    with pytest.raises(revision.Step9GReportRevisionError):
        revision.revise_report(FAKE_A, FAKE_B, dry_run=False)
