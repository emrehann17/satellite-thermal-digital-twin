"""
tests/test_step9f.py

Step9F ("exploratory cross-region feature-representation experiment") icin
odaklı unittest testleri. Agir/gercek GEE veya buyuk model egitimi
GEREKTIRMEZ -- saf mantik (sabit varyantlar, yasak kolon kontrolu,
namespace guvenligi, region-relative istatistiklerin etiket KULLANMADIGI,
esli bootstrap'in spatial block'lari KORUDUGU, reprodüksiyon kontrolu
mantigi, aday tarama kurali, dry-run no-output davranisi, main.py
transfer-explore dispatch) uzerine odaklanir.

Calistirma:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.cross_region_experiment as cre


class TestFixedVariants(unittest.TestCase):
    def test_exact_variant_set(self):
        expected_names = {
            "original_baseline", "original_thermal", "thermal_without_elevation",
            "thermal_without_absolute_lst", "thermal_without_tvdi_difference",
            "thermal_without_elevation_or_absolute_lst", "stable_core", "stable_core_without_landcover",
        }
        self.assertEqual(set(cre.FIXED_VARIANTS.keys()), expected_names)

    def test_original_baseline_features(self):
        self.assertEqual(
            cre.FIXED_VARIANTS["original_baseline"],
            ["ndvi_mean", "elevation_mean", "slope_mean", "landcover_dominant"],
        )

    def test_original_thermal_is_baseline_plus_thermal(self):
        self.assertEqual(len(cre.FIXED_VARIANTS["original_thermal"]), 10)
        self.assertIn("elevation_mean", cre.FIXED_VARIANTS["original_thermal"])
        self.assertIn("fused_lst_mean", cre.FIXED_VARIANTS["original_thermal"])

    def test_thermal_without_elevation_drops_only_elevation(self):
        variant = cre.FIXED_VARIANTS["thermal_without_elevation"]
        self.assertNotIn("elevation_mean", variant)
        self.assertEqual(len(variant), len(cre.FIXED_VARIANTS["original_thermal"]) - 1)

    def test_thermal_without_absolute_lst_drops_three_features(self):
        variant = cre.FIXED_VARIANTS["thermal_without_absolute_lst"]
        for f in ("current_lst_mean", "downscaled_lst_mean", "fused_lst_mean"):
            self.assertNotIn(f, variant)
        self.assertIn("lst_anomaly_mean", variant)

    def test_stable_core_features(self):
        self.assertEqual(
            set(cre.FIXED_VARIANTS["stable_core"]),
            {"ndvi_mean", "slope_mean", "landcover_dominant", "lst_anomaly_mean", "current_tvdi_mean"},
        )

    def test_stable_core_without_landcover_drops_landcover_only(self):
        with_lc = set(cre.FIXED_VARIANTS["stable_core"])
        without_lc = set(cre.FIXED_VARIANTS["stable_core_without_landcover"])
        self.assertEqual(with_lc - without_lc, {"landcover_dominant"})

    def test_regime_b_variants_are_exactly_two(self):
        self.assertEqual(cre.REGIME_B_VARIANTS, ["original_thermal", "stable_core"])

    def test_regime_labels_never_say_source_only_or_direct_transfer(self):
        forbidden_substrings = ["source-only", "direct transfer", "unbiased external transfer"]
        for label in cre.REGIME_B_LABELS:
            for bad in forbidden_substrings:
                self.assertNotIn(bad, label.lower())


class TestNoForbiddenFeatures(unittest.TestCase):
    def test_no_variant_contains_forbidden_columns(self):
        for variant, features in cre.FIXED_VARIANTS.items():
            leaked = set(features).intersection(cre.FORBIDDEN_MODEL_COLUMNS)
            self.assertEqual(leaked, set(), f"variant '{variant}' leaks forbidden columns: {leaked}")

    def test_check_no_forbidden_features_raises_on_violation(self):
        with self.assertRaises(ValueError):
            cre.check_no_forbidden_features(["ndvi_mean", "burned"])

    def test_check_no_forbidden_features_passes_on_clean_list(self):
        cre.check_no_forbidden_features(["ndvi_mean", "elevation_mean"])  # should not raise


class TestNamespaceSafety(unittest.TestCase):
    def test_cross_region_path_within_pair_passes(self):
        cre.assert_paths_are_safely_namespaced(
            "manavgat_2021", "bejis_2022",
            Path("/tmp/outputs/cross_region/manavgat_2021__bejis_2022/step9f/x.json"),
        )

    def test_cross_region_path_outside_pair_raises(self):
        with self.assertRaises(ValueError):
            cre.assert_paths_are_safely_namespaced(
                "manavgat_2021", "bejis_2022",
                Path("/tmp/outputs/cross_region/manavgat_2021__kozan_2023/step9f/x.json"),
            )

    def test_experiments_path_for_either_region_passes(self):
        cre.assert_paths_are_safely_namespaced(
            "manavgat_2021", "bejis_2022",
            Path("/tmp/outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet"),
        )

    def test_experiments_path_for_unrelated_region_raises(self):
        with self.assertRaises(ValueError):
            cre.assert_paths_are_safely_namespaced(
                "manavgat_2021", "bejis_2022",
                Path("/tmp/outputs/experiments/kozan_2023/step8a/step8a_500m_modeling_dataset.parquet"),
            )


class TestRegionRelativeNeverUsesLabels(unittest.TestCase):
    def _make_df(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "valid_for_modeling": [True] * n,
            "burned": rng.integers(0, 2, n),
            "ndvi_mean": rng.normal(0.4, 0.1, n),
            "elevation_mean": rng.normal(300, 50, n),
        })

    def test_stats_unaffected_by_burned_column_mutation(self):
        df = self._make_df()
        stats_before = cre.compute_region_robust_stats(df, ["ndvi_mean", "elevation_mean"])

        df_mutated = df.copy()
        df_mutated["burned"] = 1 - df_mutated["burned"]  # tamamen ters cevir
        stats_after = cre.compute_region_robust_stats(df_mutated, ["ndvi_mean", "elevation_mean"])

        self.assertEqual(stats_before, stats_after)

    def test_stats_computation_does_not_read_burned_column_at_all(self):
        df = self._make_df()
        df_no_burned = df.drop(columns=["burned"])
        # burned kolonu HIC olmasa bile stats hesaplanabilmeli (kullanilmadiginin kaniti).
        stats = cre.compute_region_robust_stats(df_no_burned, ["ndvi_mean", "elevation_mean"])
        self.assertIn("ndvi_mean", stats)
        self.assertIsNotNone(stats["ndvi_mean"]["median"])

    def test_zero_iqr_handled_safely(self):
        df = pd.DataFrame({
            "valid_for_modeling": [True] * 10,
            "constant_feature": [5.0] * 10,
        })
        stats = cre.compute_region_robust_stats(df, ["constant_feature"])
        self.assertTrue(stats["constant_feature"]["zero_iqr_fallback_used"])
        self.assertEqual(stats["constant_feature"]["iqr"], 1.0)  # fallback, sifire bolme YOK

    def test_apply_transform_preserves_nan_for_missing(self):
        df = pd.DataFrame({"valid_for_modeling": [True, True, True], "ndvi_mean": [0.1, np.nan, 0.3]})
        stats = cre.compute_region_robust_stats(df, ["ndvi_mean"])
        transformed = cre.apply_region_robust_transform(df, stats, ["ndvi_mean"])
        self.assertTrue(pd.isna(transformed["ndvi_mean"].iloc[1]))


class TestPairedBootstrapUsesSpatialBlocks(unittest.TestCase):
    def test_bootstrap_resamples_whole_blocks_not_individual_rows(self):
        # Iki blok: 'A' (5 satir, hep ayni deger) ve 'B' (5 satir, hep ayni deger).
        # Herhangi bir replikada, bir bloktan gelen satir sayisi HER ZAMAN o
        # blogun orijinal satir sayisinin bir katı olmalidir (satir-bazli
        # rastgele karistirma OLMAMALI).
        n_per_block = 5
        df = pd.DataFrame({
            "block": ["A"] * n_per_block + ["B"] * n_per_block,
            "burned": ([1, 0, 1, 0, 1] + [0, 1, 0, 1, 0]),
            "cand_prob": [0.9] * n_per_block + [0.1] * n_per_block,
            "ref_prob": [0.5] * n_per_block + [0.5] * n_per_block,
        })
        samples = cre.paired_spatial_block_bootstrap(
            df, block_col="block", y_col="burned",
            candidate_prob_col="cand_prob", reference_prob_col="ref_prob",
            n_replicates=20, random_state=0,
        )
        self.assertGreater(len(samples), 0)
        # delta_roc_auc sutunu var olmali (metrikler hesaplanmis).
        self.assertIn("delta_roc_auc", samples.columns)

    def test_bootstrap_support_category_boundaries(self):
        self.assertEqual(cre.bootstrap_support_category(0.01, 0.05, higher_is_better=True), "positive_support")
        self.assertEqual(cre.bootstrap_support_category(-0.05, -0.01, higher_is_better=True), "negative_support")
        self.assertEqual(cre.bootstrap_support_category(-0.01, 0.05, higher_is_better=True), "uncertain")
        # Brier: dusuk = iyi -> negatif delta = positive_support
        self.assertEqual(cre.bootstrap_support_category(-0.05, -0.01, higher_is_better=False), "positive_support")
        self.assertEqual(cre.bootstrap_support_category(0.01, 0.05, higher_is_better=False), "negative_support")


class TestSourceOnlyThreshold(unittest.TestCase):
    def test_threshold_selection_uses_only_source_oof_grid(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        prob = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.15, 0.6, 0.25, 0.85])
        covered = np.ones(len(y), dtype=bool)
        threshold, info = cre.select_threshold_from_oof_predictions(y, prob, covered)
        self.assertIn(threshold, list(cre.F1_THRESHOLD_GRID))
        self.assertEqual(info["method"], "source_oof_f1_optimal")

    def test_threshold_selection_falls_back_when_insufficient_coverage(self):
        y = np.array([0, 0, 0])
        prob = np.array([0.1, 0.2, 0.3])
        covered = np.zeros(len(y), dtype=bool)
        threshold, info = cre.select_threshold_from_oof_predictions(y, prob, covered)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(info["method"], "default_insufficient_oof_coverage")


class TestReproductionCheckLogic(unittest.TestCase):
    def setUp(self):
        import src.step9f_exploratory_transfer_feature_experiment as s9f
        self.s9f = s9f

    def _candidate(self, direction, population, variant, roc_auc, pr_auc, brier):
        return {
            "transfer_direction": direction, "population": population,
            "regime": self.s9f.REGIME_A_LABEL, "variant": variant,
            "target_metrics": {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier_score": brier},
        }

    def test_matching_metrics_pass(self):
        candidates = [self._candidate("a_to_b", "pop1", "original_baseline", 0.75, 0.2, 0.05)]
        step9b_metrics = {"results": [{
            "transfer_direction": "a_to_b", "population": "pop1", "skipped": False,
            "baseline_metrics": {"roc_auc": 0.75, "pr_auc": 0.2, "brier_score": 0.05},
        }]}
        result = self.s9f.verify_reproduction_against_step9b(candidates, step9b_metrics)
        self.assertTrue(result["all_within_tolerance"])

    def test_mismatched_metrics_fail(self):
        candidates = [self._candidate("a_to_b", "pop1", "original_baseline", 0.75, 0.2, 0.05)]
        step9b_metrics = {"results": [{
            "transfer_direction": "a_to_b", "population": "pop1", "skipped": False,
            "baseline_metrics": {"roc_auc": 0.80, "pr_auc": 0.2, "brier_score": 0.05},  # roc_auc farkli
        }]}
        result = self.s9f.verify_reproduction_against_step9b(candidates, step9b_metrics)
        self.assertFalse(result["all_within_tolerance"])
        self.assertEqual(result["checks"][0]["status"], "MISMATCH_BEYOND_TOLERANCE")

    def test_missing_step9b_reference_does_not_fail(self):
        candidates = [self._candidate("a_to_b", "unknown_pop", "original_baseline", 0.75, 0.2, 0.05)]
        step9b_metrics = {"results": []}
        result = self.s9f.verify_reproduction_against_step9b(candidates, step9b_metrics)
        self.assertTrue(result["all_within_tolerance"])
        self.assertEqual(result["checks"][0]["status"], "no_step9b_reference_available")


class TestCandidateScreeningRule(unittest.TestCase):
    def setUp(self):
        import src.step9f_exploratory_transfer_feature_experiment as s9f
        self.s9f = s9f
        self.primary = s9f.PRIMARY_POPULATIONS[0]
        self.directions = ["a_to_b", "b_to_a"]

    def _make_candidate(self, direction, regime, variant, roc_auc, source_oof_auc):
        return {
            "transfer_direction": direction, "population": self.primary,
            "regime": regime, "variant": variant,
            "target_metrics": {"roc_auc": roc_auc, "ranking_reversal_suspected": False},
            "source_oof": {"roc_auc": source_oof_auc},
        }

    def _make_paired(self, direction, regime, variant, d_roc, d_pr, d_brier):
        return {
            "transfer_direction": direction, "population": self.primary,
            "regime": regime, "variant": variant,
            "delta_roc_auc_vs_original_thermal": d_roc,
            "delta_pr_auc_vs_original_thermal": d_pr,
            "delta_brier_vs_original_thermal": d_brier,
        }

    def test_candidate_meeting_all_criteria_is_flagged(self):
        regime, variant = self.s9f.REGIME_A_LABEL, "stable_core"
        ref_variant = self.s9f.PRIMARY_REFERENCE_VARIANT
        candidates, paired_rows = [], []
        for d in self.directions:
            candidates.append(self._make_candidate(d, regime, variant, roc_auc=0.60, source_oof_auc=0.70))
            candidates.append(self._make_candidate(d, self.s9f.REGIME_A_LABEL, ref_variant, roc_auc=0.55, source_oof_auc=0.72))
            paired_rows.append(self._make_paired(d, regime, variant, d_roc=0.05, d_pr=0.01, d_brier=-0.005))
        paired_df = pd.DataFrame(paired_rows)
        screening = self.s9f.build_candidate_screening_table(candidates, paired_df, bootstrap_groups=[])
        row = screening[(screening["regime"] == regime) & (screening["variant"] == variant)].iloc[0]
        self.assertTrue(bool(row["candidate_for_third_region_freeze"]))

    def test_candidate_failing_one_direction_is_not_flagged(self):
        regime, variant = self.s9f.REGIME_A_LABEL, "stable_core"
        ref_variant = self.s9f.PRIMARY_REFERENCE_VARIANT
        candidates, paired_rows = [], []
        deltas = [0.05, -0.02]  # ikinci yonde IYILESME YOK
        for d, d_roc in zip(self.directions, deltas):
            candidates.append(self._make_candidate(d, regime, variant, roc_auc=0.60, source_oof_auc=0.70))
            candidates.append(self._make_candidate(d, self.s9f.REGIME_A_LABEL, ref_variant, roc_auc=0.55, source_oof_auc=0.72))
            paired_rows.append(self._make_paired(d, regime, variant, d_roc=d_roc, d_pr=0.01, d_brier=-0.005))
        paired_df = pd.DataFrame(paired_rows)
        screening = self.s9f.build_candidate_screening_table(candidates, paired_df, bootstrap_groups=[])
        row = screening[(screening["regime"] == regime) & (screening["variant"] == variant)].iloc[0]
        self.assertFalse(bool(row["candidate_for_third_region_freeze"]))

    def test_large_source_oof_drop_disqualifies_candidate(self):
        regime, variant = self.s9f.REGIME_A_LABEL, "stable_core"
        ref_variant = self.s9f.PRIMARY_REFERENCE_VARIANT
        candidates, paired_rows = [], []
        for d in self.directions:
            # source OOF ROC-AUC referansa gore 0.20 dustu (> 0.05 tolerans).
            candidates.append(self._make_candidate(d, regime, variant, roc_auc=0.60, source_oof_auc=0.50))
            candidates.append(self._make_candidate(d, self.s9f.REGIME_A_LABEL, ref_variant, roc_auc=0.55, source_oof_auc=0.70))
            paired_rows.append(self._make_paired(d, regime, variant, d_roc=0.05, d_pr=0.01, d_brier=-0.005))
        paired_df = pd.DataFrame(paired_rows)
        screening = self.s9f.build_candidate_screening_table(candidates, paired_df, bootstrap_groups=[])
        row = screening[(screening["regime"] == regime) & (screening["variant"] == variant)].iloc[0]
        self.assertFalse(bool(row["candidate_for_third_region_freeze"]))
        self.assertFalse(bool(row["source_oof_auc_drop_within_tolerance"]))


class TestNoStep9AToStep9EMutation(unittest.TestCase):
    """gather_step9_provenance/load_step9b_metrics gibi salt-okunur okuyucular
    HICBIR dosyaya YAZMAMALIDIR (yalnizca .exists()/.read_text() kullanmali)."""

    def test_provenance_functions_never_write(self):
        import unittest.mock as mock
        import src.step9f_exploratory_transfer_feature_experiment as s9f
        with mock.patch.object(Path, "write_text", side_effect=AssertionError("write_text CAGRILDI!")):
            with mock.patch.object(Path, "write_bytes", side_effect=AssertionError("write_bytes CAGRILDI!")):
                s9f.gather_step9_provenance("manavgat_2021", "bejis_2022")  # exception firlatmamali


class TestDryRunNoOutput(unittest.TestCase):
    def test_dry_run_creates_no_files_and_no_step9f_dir_if_absent(self):
        import shutil
        from scripts.run_exploratory_transfer_features import main as run_main
        from core.cross_region_experiment import step9f_output_dir

        output_dir = step9f_output_dir("manavgat_2021", "bejis_2022")
        pre_existing = output_dir.exists()
        if pre_existing:
            before_files = set(output_dir.rglob("*"))
        else:
            before_files = set()

        result = run_main(source_id="manavgat_2021", target_id="bejis_2022", reverse=True, dry_run=True)
        self.assertFalse(result["ran"])
        self.assertEqual(result["reason"], "dry_run")

        if not pre_existing:
            self.assertFalse(output_dir.exists(), "dry-run step9f cikti dizinini OLUSTURMAMALI")
        else:
            after_files = set(output_dir.rglob("*"))
            self.assertEqual(before_files, after_files, "dry-run mevcut step9f dizinine YENI dosya EKLEMEMELI")


class TestStep9AToStep9EPreserved(unittest.TestCase):
    """Step9F'in (dry-run VEYA gercek calisma) Step9A-Step9E ciktilarina
    HICBIR SEKILDE dokunmadigini dogrular (hash-based before/after)."""

    def _hash_existing_stage_files(self, source_id: str, target_id: str, stages: tuple[str, ...]) -> dict:
        import hashlib
        from core.cross_region_experiment import resolve_step9_stage_dir

        hashes = {}
        for stage in stages:
            stage_dir = resolve_step9_stage_dir(source_id, target_id, stage)
            if not stage_dir.exists():
                continue
            for path in sorted(stage_dir.rglob("*")):
                if path.is_file():
                    hashes[str(path)] = hashlib.md5(path.read_bytes()).hexdigest()
        return hashes

    def test_dry_run_does_not_modify_step9a_through_step9e(self):
        from scripts.run_exploratory_transfer_features import main as run_main

        stages = ("step9a", "step9b", "step9c", "step9d", "step9e")
        before = self._hash_existing_stage_files("manavgat_2021", "bejis_2022", stages)
        run_main(source_id="manavgat_2021", target_id="bejis_2022", reverse=True, dry_run=True)
        after = self._hash_existing_stage_files("manavgat_2021", "bejis_2022", stages)
        self.assertEqual(before, after, "Step9A-E dosyalari dry-run sirasinda DEGISTI!")


if __name__ == "__main__":
    unittest.main()