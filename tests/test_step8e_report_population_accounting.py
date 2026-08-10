"""
tests/test_step8e_report_population_accounting.py

Contract tests for the Step8E (experiment-aware) final-report population
count accounting in scripts/run_step8_modeling.py.

Background: the report's `step8a_dataset` section mixed pre-exclusion
burned/unburned counts (read directly from step8a_dataset_stats.json's
top-level fields) with a post-exclusion `valid_modeling_cells`, producing
an internally-inconsistent report (burned + unburned == total_500m_cells,
not valid_modeling_cells). The fix computes ALL modeled counts from the
actual Step8A modeling dataset filtered by `valid_for_modeling == True`.

This is a report-generation/provenance test only: no Step8A/8B/8C/8D logic
is invoked, no model is trained, and no canonical experiment artifact is
read or written -- everything here uses synthetic fixtures in a temporary
directory.

Covers (task numbering):
    1. a dataset with retained invalid rows reports only
       valid_for_modeling==true counts as modeled counts
    2. burned + unburned == valid_modeling_cells
    3. valid + excluded == total
    4. burned rate uses modeled burned / modeled valid count
    5. Step8E fails when Step8A modeled counts disagree with Step8B all_valid
    6. Step8B-D metrics are passed through unchanged
    7. Markdown clearly labels total, excluded, and modeled counts

Run:
    python -m unittest tests.test_step8e_report_population_accounting
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.run_step8_modeling as run_step8


# =============================================================================
# Synthetic fixtures (no canonical experiment artifacts touched)
# =============================================================================
def _make_ctx(tmp: Path) -> dict:
    step8a_dir = tmp / "step8a"
    step8b_dir = tmp / "step8b"
    step8c_dir = tmp / "step8c"
    step8d_dir = tmp / "step8d"
    step7e_dir = tmp / "step7e"
    gate_dir = tmp / "labels"
    for d in (step8a_dir, step8b_dir, step8c_dir, step8d_dir, step7e_dir, gate_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "experiment_id": "synthetic_test_exp",
        "region_key": "synthetic_region",
        "role": "mediterranean_transfer_wildfire",
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-01",
        "label_start_date": "2021-07-02",
        "label_end_date": "2021-08-01",
        "baseline_years": [2019, 2020],
        "gate_labels_dir": gate_dir,
        "step7e_output_dir": step7e_dir,
        "step8a_output_dir": step8a_dir,
        "step8b_output_dir": step8b_dir,
        "step8c_output_dir": step8c_dir,
        "step8d_output_dir": step8d_dir,
        "step8e_output_dir": tmp / "step8e",
    }


def _write_synthetic_step8a_dataset(
    path: Path, n_valid_burned: int = 10, n_valid_unburned: int = 20,
    n_excluded: int = 3, invalid_reason: str = "pre_label_burn_excluded",
) -> pd.DataFrame:
    rows = []
    for i in range(n_valid_burned):
        rows.append({"cell_id": f"b{i}", "burned": 1, "valid_for_modeling": True, "invalid_reason": None})
    for i in range(n_valid_unburned):
        rows.append({"cell_id": f"u{i}", "burned": 0, "valid_for_modeling": True, "invalid_reason": None})
    for i in range(n_excluded):
        # deliberately burned=1 for excluded rows too, mirroring the real
        # evia_2021 bug (excluded pre-label-burn cells are, by definition,
        # burned) -- proves the fix does NOT count them as modeled-burned.
        rows.append({"cell_id": f"x{i}", "burned": 1, "valid_for_modeling": False, "invalid_reason": invalid_reason})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _step8b_metrics(n_positives: int, n_negatives: int) -> dict:
    return {
        "population_metrics": {
            "all_valid": {
                "n_positives": n_positives, "n_negatives": n_negatives, "n_splits_used": 5,
                "overall_baseline": {"roc_auc": 0.70, "pr_auc": 0.55, "brier_score": 0.20},
                "overall_thermal": {"roc_auc": 0.75, "pr_auc": 0.60, "brier_score": 0.18},
                "delta_auc": 0.05, "delta_pr_auc": 0.05, "delta_brier": -0.02,
                "interpretation": "positive",
            },
        },
    }


def _step8c_metrics() -> dict:
    return {"bootstrap_ci_by_population": {"all_valid": {"available": True, "delta_auc_ci95": [0.01, 0.09]}}}


def _step8d_metrics() -> dict:
    return {"ablation_results": {"all_valid": {"ranking_order": ["fused_lst_only"]}}}


# =============================================================================
# 1-4: compute_step8a_dataset_section
# =============================================================================
class TestComputeStep8aDatasetSection(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.ctx = _make_ctx(self.tmp)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_retained_invalid_rows_excluded_from_modeled_counts(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=10, n_valid_unburned=20, n_excluded=3)
        results = {"step8a": {"parquet_path": str(parquet_path)}}

        section = run_step8.compute_step8a_dataset_section(results, self.ctx)

        self.assertEqual(section["total_500m_cells"], 33)
        self.assertEqual(section["excluded_modeling_cells"], 3)
        self.assertEqual(section["valid_modeling_cells"], 30)
        self.assertEqual(section["burned_cell_count"], 10)  # NOT 13 (the 3 excluded rows are burned=1)
        self.assertEqual(section["unburned_cell_count"], 20)

    def test_burned_plus_unburned_equals_valid_modeling_cells(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=7, n_valid_unburned=13, n_excluded=5)
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertEqual(
            section["burned_cell_count"] + section["unburned_cell_count"],
            section["valid_modeling_cells"],
        )

    def test_valid_plus_excluded_equals_total(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=7, n_valid_unburned=13, n_excluded=5)
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertEqual(
            section["valid_modeling_cells"] + section["excluded_modeling_cells"],
            section["total_500m_cells"],
        )

    def test_burned_rate_uses_modeled_counts_only(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=2774, n_valid_unburned=4954, n_excluded=16)
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertAlmostEqual(section["burned_rate"], 2774 / 7728, places=12)
        # Explicitly NOT the pre-exclusion rate (the original bug).
        self.assertNotAlmostEqual(section["burned_rate"], (2774 + 16) / (2774 + 16 + 4954), places=6)

    def test_invalid_reason_counts_and_exclusion_reason_derived_not_hardcoded(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(
            parquet_path, n_valid_burned=5, n_valid_unburned=5, n_excluded=4,
            invalid_reason="some_other_exclusion_reason",
        )
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertEqual(section["invalid_reason_counts"], {"some_other_exclusion_reason": 4})
        self.assertEqual(section["exclusion_reason"], "some_other_exclusion_reason")

    def test_no_excluded_rows_omits_exclusion_fields(self):
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=5, n_valid_unburned=5, n_excluded=0)
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertEqual(section["excluded_modeling_cells"], 0)
        self.assertNotIn("exclusion_reason", section)
        self.assertNotIn("invalid_reason_counts", section)

    def test_matches_evia_2021_expected_result(self):
        """Reproduces the exact evia_2021 bug/fix numbers from the task."""
        parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(parquet_path, n_valid_burned=2774, n_valid_unburned=4954, n_excluded=16)
        results = {"step8a": {"parquet_path": str(parquet_path)}}
        section = run_step8.compute_step8a_dataset_section(results, self.ctx)
        self.assertEqual(section["total_500m_cells"], 7744)
        self.assertEqual(section["excluded_modeling_cells"], 16)
        self.assertEqual(section["valid_modeling_cells"], 7728)
        self.assertEqual(section["burned_cell_count"], 2774)
        self.assertEqual(section["unburned_cell_count"], 4954)
        self.assertAlmostEqual(section["burned_rate"], 0.3589544513457557, places=12)

    def test_missing_dataset_fails_fast_no_fallback_to_raw_stats(self):
        results = {"step8a": {"parquet_path": str(self.tmp / "step8a" / "does_not_exist.parquet")}}
        with self.assertRaises(run_step8.Step8EReportError):
            run_step8.compute_step8a_dataset_section(results, self.ctx)


# =============================================================================
# 5. Step8B all_valid cross-check
# =============================================================================
class TestCrossCheckAgainstStep8B(unittest.TestCase):
    def test_matching_counts_pass(self):
        section = {"burned_cell_count": 2774, "unburned_cell_count": 4954}
        run_step8._cross_check_step8a_against_step8b(section, _step8b_metrics(2774, 4954))  # no raise

    def test_mismatched_counts_raise(self):
        section = {"burned_cell_count": 2789, "unburned_cell_count": 4955}
        with self.assertRaises(run_step8.Step8EReportError):
            run_step8._cross_check_step8a_against_step8b(section, _step8b_metrics(2774, 4954))

    def test_missing_step8b_all_valid_is_skipped_not_an_error(self):
        section = {"burned_cell_count": 2774, "unburned_cell_count": 4954}
        run_step8._cross_check_step8a_against_step8b(section, {})  # no raise


# =============================================================================
# 6-7. End-to-end write_final_report(): pass-through + Markdown wording
# =============================================================================
class TestWriteFinalReportEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.ctx = _make_ctx(self.tmp)
        self.parquet_path = self.tmp / "step8a" / "dataset.parquet"
        _write_synthetic_step8a_dataset(self.parquet_path, n_valid_burned=2774, n_valid_unburned=4954, n_excluded=16)

        (self.ctx["step8b_output_dir"] / "step8b_model_comparison_metrics.json").write_text(
            json.dumps(_step8b_metrics(2774, 4954)), encoding="utf-8",
        )
        (self.ctx["step8c_output_dir"] / "step8c_bootstrap_metrics.json").write_text(
            json.dumps(_step8c_metrics()), encoding="utf-8",
        )
        (self.ctx["step8d_output_dir"] / "step8d_ablation_metrics.json").write_text(
            json.dumps(_step8d_metrics()), encoding="utf-8",
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_step8b_c_d_metrics_passed_through_unchanged(self):
        results = {"step8a": {"parquet_path": str(self.parquet_path)}}
        run_step8.write_final_report(self.ctx, results)

        report = json.loads((self.ctx["step8e_output_dir"] / "final_step8_report.json").read_text())

        self.assertEqual(report["step8b_baseline_vs_fused_model"], _step8b_metrics(2774, 4954)["population_metrics"])
        self.assertEqual(report["step8c_bootstrap_uncertainty"], _step8c_metrics())
        self.assertEqual(report["step8d_thermal_ablation"], _step8d_metrics()["ablation_results"])
        self.assertIn("no_30m_burned_area_prediction_claim", report["claim_policy"])
        self.assertTrue(report["claim_policy"]["no_30m_burned_area_prediction_claim"])
        self.assertIsInstance(report["limitations"], list)
        self.assertGreater(len(report["limitations"]), 0)

    def test_step8a_dataset_section_is_correct_in_final_report(self):
        results = {"step8a": {"parquet_path": str(self.parquet_path)}}
        run_step8.write_final_report(self.ctx, results)
        report = json.loads((self.ctx["step8e_output_dir"] / "final_step8_report.json").read_text())
        ds = report["step8a_dataset"]
        self.assertEqual(ds["total_500m_cells"], 7744)
        self.assertEqual(ds["excluded_modeling_cells"], 16)
        self.assertEqual(ds["valid_modeling_cells"], 7728)
        self.assertEqual(ds["burned_cell_count"], 2774)
        self.assertEqual(ds["unburned_cell_count"], 4954)
        self.assertEqual(ds["burned_cell_count"] + ds["unburned_cell_count"], ds["valid_modeling_cells"])

    def test_markdown_labels_total_excluded_and_modeled_counts_distinctly(self):
        results = {"step8a": {"parquet_path": str(self.parquet_path)}}
        run_step8.write_final_report(self.ctx, results)
        md = (self.ctx["step8e_output_dir"] / "final_step8_report.md").read_text()

        self.assertIn("Total cells retained for provenance: 7744", md)
        self.assertIn("Excluded from modeling: 16", md)
        self.assertIn("Valid modeling cells: 7728", md)
        self.assertIn("Modeled burned: 2774", md)
        self.assertIn("Modeled unburned: 4954", md)
        self.assertIn("Modeled burned rate:", md)
        # The pre-exclusion (buggy) numbers must never appear labeled as modeled.
        self.assertNotIn("Modeled burned: 2789", md)
        self.assertNotIn("Modeled unburned: 4955", md)

    def test_write_final_report_fails_fast_on_step8b_mismatch(self):
        (self.ctx["step8b_output_dir"] / "step8b_model_comparison_metrics.json").write_text(
            json.dumps(_step8b_metrics(2789, 4955)), encoding="utf-8",  # pre-exclusion (wrong) counts
        )
        results = {"step8a": {"parquet_path": str(self.parquet_path)}}
        with self.assertRaises(run_step8.Step8EReportError):
            run_step8.write_final_report(self.ctx, results)


if __name__ == "__main__":
    unittest.main()
