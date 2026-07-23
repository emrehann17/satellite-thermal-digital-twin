"""
tests/test_step9e_report_fix.py

Focused tests for the Step9E metadata/provenance fix in
src/step9e_distribution_shift_audit.py:

    1. safe_wording is derived dynamically from Step9D's canonical
       overall_conclusion instead of being a single static sentence used
       for every pair.
    2. Step9B predictions (.parquet) and metrics (.json) provenance fields
       are kept distinct (the original bug copied the predictions path
       into the metrics field).
    3. A --report-only regeneration mode updates ONLY safe_wording/
       provenance/timestamp fields and fails fast if any numeric/scientific
       section would change.

This is a report-generation/provenance test only: no Step9A-D logic is
invoked, no model is trained/retrained, and no canonical experiment
artifact under outputs/cross_region/ is read or written -- every test
patches `cross_region_output_root` to point at a temporary directory and
uses fully synthetic fixtures there.

Covers (task numbering):
    1. transfer_not_supported wording
    2. partial_transfer_supported wording
    3. correct separation of predictions Parquet and metrics JSON paths
    4. metrics path cannot end in .parquet
    5. source/target mismatch fails
    6. report-only mode preserves numeric JSON sections
    7. report-only mode does not modify CSV/PNG or Step9A-D files
    8. Mugla-Evia resolves to the partial/asymmetric wording

Run:
    python -m unittest tests.test_step9e_report_fix
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.step9e_distribution_shift_audit as step9e


# =============================================================================
# Synthetic fixtures -- no canonical experiment artifacts touched. All Step9E
# path resolution is redirected under a temp dir via patching
# `cross_region_output_root` (imported by name into step9e's module
# namespace), so every resolve_step9b_*_path / resolve_step9d_report_path /
# step9e_output_dir call in the module under test resolves inside tmp_path.
# =============================================================================
def _fake_cross_region_root(tmp_path: Path):
    def _root(source_id: str, target_id: str) -> Path:
        return tmp_path / "cross_region" / f"{source_id}__{target_id}"
    return _root


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_bytes(path: Path, content: bytes = b"dummy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _step9b_metrics(source_id: str, target_id: str) -> dict:
    return {"source_experiment_id": source_id, "target_experiment_id": target_id, "results": []}


def _step9d_report(source_id: str, target_id: str, overall_conclusion: str) -> dict:
    return {
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        "overall_conclusion": overall_conclusion,
        "overall_conclusion_text": "synthetic conclusion text",
    }


def _synthetic_part_f_summary() -> dict:
    return {
        "primary_population": "burnable_tree_shrub_grass",
        "top_globally_shifted_features": [
            {"feature": "elevation", "smd": 0.9, "psi_source_to_target": 0.3,
             "psi_target_to_source": 0.31, "normalized_wasserstein_by_source_iqr": 0.4,
             "outside_source_support_fraction": 0.05, "shift_category": "high_shift"},
        ],
        "top_missingness_differences": [
            {"feature": "elevation", "source_missing_fraction": 0.0,
             "target_missing_fraction": 0.01, "missingness_gap": 0.01},
        ],
        "landcover_differences_primary_population": {},
        "features_with_relationship_direction_flip": [],
        "probability_collapse_below_threshold": [],
        "ranking_reversal_suspected": False,
        "diagnosis_categories": ["high_shift"],
        "likely_contributors_to_poor_cross_region_discrimination": ["synthetic contributor"],
        "note": "synthetic note",
    }


def _synthetic_payload(source_id: str, target_id: str, safe_wording: str, metrics_path: str) -> dict:
    """A full, self-consistent (except for the deliberately-wrong metrics
    path, when requested) distribution_shift_audit.json-shaped payload."""
    return {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "audit_type": "post_hoc_distribution_and_relationship_shift_diagnostic",
        "safe_wording": safe_wording,
        "interpretation_rules": ["rule one", "rule two"],
        "never_claims": ["causal explanation"],
        "numeric_audit_features": ["elevation", "current_lst_anomaly_mean"],
        "categorical_audit_features": ["landcover_dominant"],
        "never_audit_as_feature_columns": ["burned", "cell_id"],
        "primary_populations": ["burnable_tree_shrub_grass"],
        "secondary_populations": ["all_valid", "burnable_tree_shrub"],
        "populations_evaluated": ["burnable_tree_shrub_grass", "all_valid", "burnable_tree_shrub"],
        "part_a_numeric_feature_shift": [
            {"feature": "elevation", "population": "all_valid", "smd": 0.42, "ks_statistic": 0.31},
        ],
        "part_b_categorical_landcover_shift": {"all_valid": {"total_variation_distance": 0.12}},
        "part_c_label_conditional_relationships": [
            {"feature": "elevation", "population": "all_valid", "raw_auc": 0.61},
        ],
        "part_c_relationship_direction_flips": [
            {"feature": "elevation", "population": "all_valid", "relationship_flip_score": 1},
        ],
        "part_d_prediction_distribution_audit": [
            {"transfer_direction": "x_to_y", "population": "all_valid", "model": "thermal",
             "roc_auc": 0.55, "diagnostic_inverse_roc_auc": 0.71},
        ],
        "part_e_calibration_bins": [
            {"transfer_direction": "x_to_y", "population": "all_valid", "model": "thermal",
             "bin_index": 0, "predicted_probability_mean": 0.1},
        ],
        "part_f_summary": _synthetic_part_f_summary(),
        "step9b_metrics_source_path": metrics_path,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


class Step9EFixtureTestCase(unittest.TestCase):
    """Base class: builds a synthetic (source, target) pair tree under a
    TemporaryDirectory and patches cross_region_output_root for the
    duration of each test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self._patcher = patch.object(
            step9e, "cross_region_output_root", _fake_cross_region_root(self.tmp),
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmpdir.cleanup)

    def _pair_root(self, source_id: str, target_id: str) -> Path:
        return self.tmp / "cross_region" / f"{source_id}__{target_id}"

    def _write_step9b(self, source_id: str, target_id: str) -> None:
        root = self._pair_root(source_id, target_id)
        _write_bytes(root / "step9b" / "cross_region_transfer_predictions.parquet")
        _write_json(root / "step9b" / "cross_region_transfer_metrics.json", _step9b_metrics(source_id, target_id))

    def _write_step9d(self, source_id: str, target_id: str, overall_conclusion: str) -> None:
        root = self._pair_root(source_id, target_id)
        _write_json(
            root / "step9d" / "final_cross_region_report.json",
            _step9d_report(source_id, target_id, overall_conclusion),
        )


# =============================================================================
# 1-2. Wording templates
# =============================================================================
class TestSafeWordingTemplates(unittest.TestCase):
    def test_transfer_not_supported_wording(self):
        wording = step9e.resolve_safe_wording("transfer_not_supported")
        self.assertEqual(
            wording,
            "Thermal incremental cross-region transfer was not supported in the "
            "original Step9 evaluation. Step9E examines whether feature-distribution "
            "shift, probability-scale shift, or region-dependent feature-label "
            "relationships are consistent with this result.",
        )

    def test_partial_transfer_supported_wording(self):
        wording = step9e.resolve_safe_wording("partial_transfer_supported")
        self.assertEqual(
            wording,
            "The original Step9 evaluation showed asymmetric or partial cross-region "
            "support for the thermal predictor set. Step9E examines the "
            "feature-distribution, probability-scale, and feature-label relationship "
            "shifts associated with this mixed result.",
        )

    def test_transfer_supported_wording_and_bidirectional_alias(self):
        direct = step9e.resolve_safe_wording("transfer_supported")
        aliased = step9e.resolve_safe_wording("bidirectional_transfer_supported")
        self.assertEqual(direct, aliased)
        self.assertIn("showed cross-region support for the thermal predictor set", direct)

    def test_wording_never_implies_forbidden_claims(self):
        forbidden_phrases = [
            "causal", "corrected transfer", "operational wildfire",
            "universally generaliz", "statistically significant",
        ]
        for conclusion in ("transfer_not_supported", "partial_transfer_supported", "transfer_supported"):
            wording = step9e.resolve_safe_wording(conclusion).lower()
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase, wording)

    def test_unknown_conclusion_fails_fast(self):
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_safe_wording("some_unrecognized_value")

    def test_missing_conclusion_fails_fast(self):
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_safe_wording(None)


# =============================================================================
# 3-4. Provenance path separation + extension guard
# =============================================================================
class TestProvenancePathSeparation(Step9EFixtureTestCase):
    def test_predictions_and_metrics_paths_are_distinct_and_correctly_suffixed(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        step9b_metrics = _step9b_metrics(source_id, target_id)
        result = step9e.resolve_step9e_provenance_and_wording(source_id, target_id, step9b_metrics)

        self.assertTrue(result["step9b_predictions_source_path"].endswith(".parquet"))
        self.assertTrue(result["step9b_metrics_source_path"].endswith(".json"))
        self.assertNotEqual(result["step9b_predictions_source_path"], result["step9b_metrics_source_path"])
        self.assertIsNotNone(result["step9b_predictions_sha256"])
        self.assertIsNotNone(result["step9b_metrics_sha256"])
        # The two source files have different (dummy) content -> different hashes.
        self.assertNotEqual(result["step9b_predictions_sha256"], result["step9b_metrics_sha256"])

    def test_metrics_path_cannot_end_in_parquet(self):
        bogus_metrics_path = self.tmp / "cross_region" / "x__y" / "step9b" / "cross_region_transfer_predictions.parquet"
        with self.assertRaises(step9e.Step9EError):
            step9e._assert_extension("step9b_metrics_source_path", bogus_metrics_path, ".json")

    def test_predictions_path_cannot_end_in_json(self):
        bogus_predictions_path = self.tmp / "cross_region" / "x__y" / "step9b" / "cross_region_transfer_metrics.json"
        with self.assertRaises(step9e.Step9EError):
            step9e._assert_extension("step9b_predictions_source_path", bogus_predictions_path, ".parquet")

    def test_missing_predictions_file_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        root = self._pair_root(source_id, target_id)
        # metrics present, predictions missing
        _write_json(root / "step9b" / "cross_region_transfer_metrics.json", _step9b_metrics(source_id, target_id))
        self._write_step9d(source_id, target_id, "transfer_not_supported")
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_step9e_provenance_and_wording(
                source_id, target_id, _step9b_metrics(source_id, target_id),
            )

    def test_missing_metrics_file_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        root = self._pair_root(source_id, target_id)
        _write_bytes(root / "step9b" / "cross_region_transfer_predictions.parquet")
        self._write_step9d(source_id, target_id, "transfer_not_supported")
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_step9e_provenance_and_wording(
                source_id, target_id, _step9b_metrics(source_id, target_id),
            )


# =============================================================================
# 5. Source/target mismatch
# =============================================================================
class TestSourceTargetMismatch(Step9EFixtureTestCase):
    def test_step9b_metrics_mismatch_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        wrong_metrics = _step9b_metrics("someone_else", "another_experiment")
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_step9e_provenance_and_wording(source_id, target_id, wrong_metrics)

    def test_step9d_report_mismatch_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        self._write_step9b(source_id, target_id)
        # Step9D report claims a DIFFERENT pair.
        root = self._pair_root(source_id, target_id)
        _write_json(
            root / "step9d" / "final_cross_region_report.json",
            _step9d_report("someone_else", "another_experiment", "transfer_not_supported"),
        )
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_step9e_provenance_and_wording(
                source_id, target_id, _step9b_metrics(source_id, target_id),
            )

    def test_step9d_conclusion_missing_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        self._write_step9b(source_id, target_id)
        root = self._pair_root(source_id, target_id)
        report = _step9d_report(source_id, target_id, "transfer_not_supported")
        del report["overall_conclusion"]
        _write_json(root / "step9d" / "final_cross_region_report.json", report)
        with self.assertRaises(step9e.Step9EError):
            step9e.resolve_step9e_provenance_and_wording(
                source_id, target_id, _step9b_metrics(source_id, target_id),
            )


# =============================================================================
# 6-7. --report-only: numeric preservation + no CSV/PNG/Step9A-D mutation
# =============================================================================
class TestReportOnlyRegeneration(Step9EFixtureTestCase):
    def _seed_full_step9e_output(self, source_id: str, target_id: str, old_safe_wording: str) -> Path:
        root = self._pair_root(source_id, target_id)
        step9e_dir = root / "step9e"
        step9e_dir.mkdir(parents=True, exist_ok=True)

        # The ORIGINAL bug: step9b_metrics_source_path incorrectly points at
        # the .parquet predictions file.
        bogus_metrics_path = str(root / "step9b" / "cross_region_transfer_predictions.parquet")
        payload = _synthetic_payload(source_id, target_id, old_safe_wording, bogus_metrics_path)
        _write_json(step9e_dir / "distribution_shift_audit.json", payload)

        md_lines = [
            "# Step9E: Cross-Region Distribution-Shift and Relationship-Shift Audit",
            "", f"- source: `{source_id}`", f"- target: `{target_id}`", "",
            "> " + old_safe_wording, "",
        ]
        (step9e_dir / "distribution_shift_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

        # Files --report-only must NEVER touch.
        for name in (
            "numeric_feature_shift.csv", "categorical_landcover_shift.csv",
            "label_conditional_feature_relationships.csv", "relationship_direction_flips.csv",
            "prediction_distribution_audit.csv", "calibration_bins.csv",
            "feature_shift_heatmap.png", "top_shifted_feature_distributions.png",
        ):
            _write_bytes(step9e_dir / name, content=b"untouched-fixture-content")

        for stage in ("step9a", "step9b", "step9c", "step9d"):
            _write_bytes(root / stage / "sentinel.txt", content=f"{stage}-untouched".encode())

        return step9e_dir

    def test_report_only_preserves_numeric_sections_and_fixes_metadata(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        old_wording = "Cross-region discrimination was not supported in the original Step9 evaluation."
        step9e_dir = self._seed_full_step9e_output(source_id, target_id, old_wording)
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "partial_transfer_supported")

        old_payload = json.loads((step9e_dir / "distribution_shift_audit.json").read_text())

        result = step9e.regenerate_report_only(source_id, target_id, force=True)

        # Numeric/scientific sections byte-for-byte (value-)identical.
        for key in (
            "part_a_numeric_feature_shift", "part_b_categorical_landcover_shift",
            "part_c_label_conditional_relationships", "part_c_relationship_direction_flips",
            "part_d_prediction_distribution_audit", "part_e_calibration_bins", "part_f_summary",
            "numeric_audit_features", "categorical_audit_features", "primary_populations",
            "secondary_populations", "populations_evaluated",
        ):
            self.assertEqual(result[key], old_payload[key], f"section {key!r} must be unchanged")

        # Metadata IS updated and now correct.
        self.assertNotEqual(result["safe_wording"], old_wording)
        self.assertEqual(
            result["safe_wording"], step9e.resolve_safe_wording("partial_transfer_supported"),
        )
        self.assertTrue(result["step9b_predictions_source_path"].endswith(".parquet"))
        self.assertTrue(result["step9b_metrics_source_path"].endswith(".json"))
        self.assertNotEqual(result["step9b_predictions_source_path"], result["step9b_metrics_source_path"])

        # Written to disk correctly.
        on_disk = json.loads((step9e_dir / "distribution_shift_audit.json").read_text())
        self.assertEqual(on_disk["safe_wording"], result["safe_wording"])

    def test_report_only_does_not_touch_csv_png_or_step9a_d_files(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        step9e_dir = self._seed_full_step9e_output(source_id, target_id, "stale wording")
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        untouched_files = [
            step9e_dir / "numeric_feature_shift.csv",
            step9e_dir / "categorical_landcover_shift.csv",
            step9e_dir / "label_conditional_feature_relationships.csv",
            step9e_dir / "relationship_direction_flips.csv",
            step9e_dir / "prediction_distribution_audit.csv",
            step9e_dir / "calibration_bins.csv",
            step9e_dir / "feature_shift_heatmap.png",
            step9e_dir / "top_shifted_feature_distributions.png",
        ]
        root = self._pair_root(source_id, target_id)
        untouched_files += [root / stage / "sentinel.txt" for stage in ("step9a", "step9b", "step9c", "step9d")]
        before = {p: p.read_bytes() for p in untouched_files}

        step9e.regenerate_report_only(source_id, target_id, force=True)

        for p, content_before in before.items():
            self.assertEqual(p.read_bytes(), content_before, f"{p} was modified by --report-only")

    def test_report_only_rewrites_markdown_only_when_stale(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        old_wording = "Cross-region discrimination was not supported in the original Step9 evaluation."
        step9e_dir = self._seed_full_step9e_output(source_id, target_id, old_wording)
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        step9e.regenerate_report_only(source_id, target_id, force=True)
        new_md = (step9e_dir / "distribution_shift_summary.md").read_text()
        self.assertNotIn(old_wording, new_md)
        self.assertIn(step9e.resolve_safe_wording("transfer_not_supported"), new_md)

    def test_report_only_leaves_already_fresh_markdown_untouched(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        correct_wording = step9e.resolve_safe_wording("transfer_not_supported")
        step9e_dir = self._seed_full_step9e_output(source_id, target_id, correct_wording)
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        md_before = (step9e_dir / "distribution_shift_summary.md").read_bytes()
        step9e.regenerate_report_only(source_id, target_id, force=True)
        md_after = (step9e_dir / "distribution_shift_summary.md").read_bytes()
        self.assertEqual(md_before, md_after)

    def test_report_only_without_existing_output_fails_fast(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")
        with self.assertRaises(step9e.Step9EError):
            step9e.regenerate_report_only(source_id, target_id, force=True)

    def test_report_only_without_force_is_a_noop_skip(self):
        source_id, target_id = "synthetic_source", "synthetic_target"
        old_wording = "stale wording that should remain untouched without --force"
        step9e_dir = self._seed_full_step9e_output(source_id, target_id, old_wording)
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "transfer_not_supported")

        result = step9e.regenerate_report_only(source_id, target_id, force=False)
        self.assertEqual(result["safe_wording"], old_wording)
        on_disk = json.loads((step9e_dir / "distribution_shift_audit.json").read_text())
        self.assertEqual(on_disk["safe_wording"], old_wording)


class TestAssertNumericSectionsUnchanged(unittest.TestCase):
    def test_metadata_only_diff_passes(self):
        old = _synthetic_payload("a", "b", "old wording", "wrong/path.parquet")
        new = dict(old)
        new["safe_wording"] = "new wording"
        new["step9b_metrics_source_path"] = "correct/path.json"
        new["created_at"] = "2026-02-02T00:00:00+00:00"
        step9e.assert_numeric_sections_unchanged(old, new)  # must not raise

    def test_numeric_field_diff_raises(self):
        old = _synthetic_payload("a", "b", "old wording", "wrong/path.parquet")
        new = json.loads(json.dumps(old))
        new["part_a_numeric_feature_shift"][0]["smd"] = 999.0
        with self.assertRaises(step9e.Step9EError):
            step9e.assert_numeric_sections_unchanged(old, new)

    def test_part_f_summary_diff_raises(self):
        old = _synthetic_payload("a", "b", "old wording", "wrong/path.parquet")
        new = json.loads(json.dumps(old))
        new["part_f_summary"]["diagnosis_categories"] = ["low_shift"]
        with self.assertRaises(step9e.Step9EError):
            step9e.assert_numeric_sections_unchanged(old, new)


# =============================================================================
# 8. Mugla <-> Evia resolves to the partial/asymmetric wording (synthetic)
# =============================================================================
class TestMuglaEviaPartialWording(Step9EFixtureTestCase):
    def test_mugla_evia_pair_resolves_partial_asymmetric_wording(self):
        source_id, target_id = "mugla_2021", "evia_2021"
        self._write_step9b(source_id, target_id)
        self._write_step9d(source_id, target_id, "partial_transfer_supported")

        result = step9e.resolve_step9e_provenance_and_wording(
            source_id, target_id, _step9b_metrics(source_id, target_id),
        )
        self.assertEqual(result["step9d_overall_conclusion"], "partial_transfer_supported")
        self.assertEqual(
            result["safe_wording"],
            "The original Step9 evaluation showed asymmetric or partial cross-region "
            "support for the thermal predictor set. Step9E examines the "
            "feature-distribution, probability-scale, and feature-label relationship "
            "shifts associated with this mixed result.",
        )


if __name__ == "__main__":
    unittest.main()
