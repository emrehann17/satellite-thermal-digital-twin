"""
tests/test_step10.py

Step10 ("unsupervised self-calibrated cross-region transfer") icin odakli
unittest testleri. Cogu test SAF MANTIK uzerine (GEE/buyuk model egitimi
gerektirmez); birkac entegrasyon testi (raw reprodüksiyon, within-region
hizalama) KUCUK sentetik veriyle GERCEK model fit'i kullanir.

Calistirma:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.step10_shared as s10


class TestRegionwiseZScore(unittest.TestCase):
    def _df(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "f1": rng.normal(10, 3, n), "f2": rng.normal(-5, 1.5, n),
            "valid_for_modeling": True,
        })

    def test_transformed_observed_values_have_mean_0_std_1(self):
        df = self._df()
        stats = s10.compute_regionwise_zscore_stats(df, ["f1", "f2"])
        transformed = s10.apply_regionwise_zscore(df, stats, ["f1", "f2"])
        self.assertAlmostEqual(transformed["f1"].mean(), 0.0, places=8)
        self.assertAlmostEqual(transformed["f1"].std(ddof=0), 1.0, places=8)
        self.assertAlmostEqual(transformed["f2"].mean(), 0.0, places=8)
        self.assertAlmostEqual(transformed["f2"].std(ddof=0), 1.0, places=8)

    def test_missing_values_become_zero(self):
        df = self._df()
        df.loc[0:4, "f1"] = np.nan
        stats = s10.compute_regionwise_zscore_stats(df, ["f1"])
        transformed = s10.apply_regionwise_zscore(df, stats, ["f1"])
        self.assertTrue((transformed["f1"].iloc[0:5] == 0.0).all())

    def test_zero_variance_guard_is_deterministic(self):
        df = pd.DataFrame({"const": [5.0] * 50, "valid_for_modeling": True})
        stats = s10.compute_regionwise_zscore_stats(df, ["const"])
        self.assertTrue(stats["const"]["constant_feature_guard_used"])
        self.assertEqual(stats["const"]["std"], 1.0)
        transformed = s10.apply_regionwise_zscore(df, stats, ["const"])
        self.assertTrue((transformed["const"] == 0.0).all())
        # Deterministic: ayni girdiyle tekrar cagirinca AYNI sonuc.
        stats2 = s10.compute_regionwise_zscore_stats(df, ["const"])
        self.assertEqual(stats, stats2)

    def test_ddof_zero_used(self):
        df = self._df(n=5, seed=1)
        stats = s10.compute_regionwise_zscore_stats(df, ["f1"])
        expected_std = float(df["f1"].std(ddof=0))
        self.assertAlmostEqual(stats["f1"]["std"], expected_std, places=10)

    def test_function_signatures_have_no_label_parameter(self):
        import inspect
        for fn in (s10.compute_regionwise_zscore_stats, s10.apply_regionwise_zscore):
            params = list(inspect.signature(fn).parameters.keys())
            self.assertNotIn("y", params)
            self.assertNotIn("target_y", params)
            self.assertNotIn("burned", params)


class TestCORAL(unittest.TestCase):
    def test_source_covariance_moves_toward_target_covariance(self):
        rng = np.random.default_rng(42)
        # Kaynak: dusuk korelasyon; hedef: yuksek korelasyon -- CORAL sonrasi
        # kaynagin kovaryansi HEDEFE (yaklasik) esit olmali.
        Xs = rng.normal(0, 1, (500, 3))
        A_true = np.array([[1, 0.8, 0.2], [0, 0.6, 0.1], [0, 0, 0.9]])
        Xt = rng.normal(0, 1, (500, 3)) @ A_true

        coral_fit = s10.fit_coral_alignment(Xs, Xt, lambda_=1e-5)
        Xs_coral = s10.apply_coral(Xs, coral_fit)

        cov_before = np.cov(Xs, rowvar=False)
        cov_after = np.cov(Xs_coral, rowvar=False)
        cov_target = np.cov(Xt, rowvar=False)

        dist_before = np.linalg.norm(cov_before - cov_target)
        dist_after = np.linalg.norm(cov_after - cov_target)
        self.assertLess(dist_after, dist_before)
        # CORAL, kaynagin kovaryansini HEDEFE YAKINSAR (tam esitlik degil, ridge nedeniyle).
        self.assertLess(dist_after, 0.5)

    def test_transform_is_finite_and_real(self):
        rng = np.random.default_rng(1)
        Xs = rng.normal(0, 1, (100, 4))
        Xt = rng.normal(0, 1, (80, 4))
        coral_fit = s10.fit_coral_alignment(Xs, Xt)
        Xs_coral = s10.apply_coral(Xs, coral_fit)
        self.assertTrue(np.isfinite(Xs_coral).all())
        self.assertFalse(np.iscomplexobj(Xs_coral))

    def test_target_is_never_transformed_by_coral(self):
        # Spec: Xt_coral = Xt_z (hedef DEGISMEZ) -- apply_coral yalnizca
        # KAYNAK icin cagrilir; bu, apply_coral'in imzasinin kendisinde
        # (yalnizca TEK bir X matrisi alir) yapisal olarak garantidir.
        import inspect
        params = list(inspect.signature(s10.apply_coral).parameters.keys())
        self.assertEqual(params, ["Xs_z_numeric", "coral_fit"])

    def test_eigenvalue_floor_prevents_non_finite_output_on_near_singular_input(self):
        rng = np.random.default_rng(2)
        # Neredeyse tekil (rank-deficient) bir kovaryans durumu.
        base = rng.normal(0, 1, (50, 1))
        Xs = np.hstack([base, base * 1.0000001, rng.normal(0, 1, (50, 1))])
        Xt = rng.normal(0, 1, (50, 3))
        coral_fit = s10.fit_coral_alignment(Xs, Xt)
        Xs_coral = s10.apply_coral(Xs, coral_fit)
        self.assertTrue(np.isfinite(Xs_coral).all())

    def test_function_signatures_have_no_label_parameter(self):
        import inspect
        for fn in (s10.fit_coral_alignment, s10.apply_coral):
            params = list(inspect.signature(fn).parameters.keys())
            self.assertNotIn("y", params)
            self.assertNotIn("burned", params)


class TestNWayPairedBootstrap(unittest.TestCase):
    def _df(self, n_blocks=10, per_block=6, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for b in range(n_blocks):
            y = rng.integers(0, 2, per_block)
            probA = rng.uniform(0, 1, per_block)
            for i in range(per_block):
                rows.append({"block": f"b{b}", "burned": int(y[i]), "probA": probA[i], "probB": 1.0 - probA[i]})
        return pd.DataFrame(rows)

    def test_paired_series_use_identical_sampled_blocks(self):
        # probB = 1 - probA HER SATIRDA -- ayni resample edilmis satirlar
        # kullanilirsa, roc_auc(probB) HER REPLIKADA TAM OLARAK
        # 1 - roc_auc(probA) olmalidir (AUC(-x) = 1 - AUC(x) ozdesligi).
        df = self._df()
        result = s10.run_n_way_paired_bootstrap(
            df, block_col="block", y_col="burned", prob_columns={"A": "probA", "B": "probB"},
            n_replicates=50, random_state=7,
        )
        replicates = result["replicates_df"]
        self.assertGreater(len(replicates), 0)
        for _, row in replicates.iterrows():
            self.assertAlmostEqual(row["roc_auc__A"] + row["roc_auc__B"], 1.0, places=8)

    def test_invalid_single_class_replicates_counted_and_excluded(self):
        # Tum satirlar burned=1 -> HER replika tek-sinif -> TUMU gecersiz.
        df = pd.DataFrame({
            "block": ["b0"] * 5 + ["b1"] * 5, "burned": [1] * 10,
            "probA": np.linspace(0.1, 0.9, 10), "probB": np.linspace(0.9, 0.1, 10),
        })
        result = s10.run_n_way_paired_bootstrap(
            df, block_col="block", y_col="burned", prob_columns={"A": "probA", "B": "probB"},
            n_replicates=20, random_state=1,
        )
        self.assertEqual(result["n_valid"], 0)
        self.assertEqual(result["n_invalid_single_class"], 20)
        self.assertTrue(result["replicates_df"].empty)

    def test_requested_vs_valid_vs_invalid_counts_are_consistent(self):
        df = self._df(n_blocks=5, per_block=4, seed=3)
        result = s10.run_n_way_paired_bootstrap(
            df, block_col="block", y_col="burned", prob_columns={"A": "probA"},
            n_replicates=30, random_state=2,
        )
        self.assertEqual(result["n_requested"], 30)
        self.assertEqual(result["n_valid"] + result["n_invalid_single_class"], 30)

    def test_bootstrap_unstable_flag_threshold(self):
        self.assertTrue(s10.is_bootstrap_unstable(899))
        self.assertFalse(s10.is_bootstrap_unstable(900))
        self.assertFalse(s10.is_bootstrap_unstable(1000))

    def test_percentile_ci_boundaries(self):
        values = pd.Series(np.linspace(-1, 1, 1000))
        lo, hi, mean = s10.percentile_ci(values)
        self.assertAlmostEqual(lo, -0.95, places=1)
        self.assertAlmostEqual(hi, 0.95, places=1)


class TestForbiddenFeaturesAndFirewall(unittest.TestCase):
    def test_no_feature_list_contains_forbidden_columns(self):
        for model_family, features in s10.FEATURE_LISTS.items():
            leaked = set(features).intersection(s10.FORBIDDEN_MODEL_COLUMNS)
            self.assertEqual(leaked, set(), f"{model_family} feature listesi yasak kolon iceriyor: {leaked}")

    def test_check_no_forbidden_features_raises(self):
        with self.assertRaises(s10.Step10Error):
            s10.check_no_forbidden_features(["ndvi_mean", "burned"])

    def test_assert_label_blind_passes_without_burned(self):
        df = pd.DataFrame({"ndvi_mean": [0.1, 0.2], "cell_id": ["a", "b"]})
        s10.assert_label_blind(df)  # exception firlatmamali

    def test_assert_label_blind_raises_with_burned(self):
        df = pd.DataFrame({"ndvi_mean": [0.1, 0.2], "burned": [0, 1]})
        with self.assertRaises(s10.Step10Error):
            s10.assert_label_blind(df)

    def test_adaptation_methods_are_exactly_three(self):
        self.assertEqual(
            s10.ADAPTATION_METHODS,
            ("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore"),
        )

    def test_regionwise_zscore_metadata_classification_is_not_source_only(self):
        label = s10.REGIONWISE_ZSCORE_METADATA_CLASS
        self.assertEqual(label, "unsupervised_target_covariate_adaptation")
        for bad in ("source-only", "direct transfer", "unbiased external transfer"):
            self.assertNotIn(bad, label.lower())


class TestNoInversionInSourceCode(unittest.TestCase):
    """Step10'un (Step9E/9F'in aksine) resmi tahminlerde/metriklerinde
    'inverse'/1-p ters cevirme mantigi ICERMEDIGINI dogrular (kod-metni
    kontrolu -- Step10 spec'i diagnostic inverse AUC dahi ISTEMEZ)."""

    def test_step10b_source_has_no_probability_inversion(self):
        src = (_PROJECT_ROOT / "src" / "step10b_label_blind_adaptation.py").read_text(encoding="utf-8")
        self.assertNotIn("1.0 - prob", src)
        self.assertNotIn("1 - prob", src)
        self.assertNotIn("1.0 - target_prob", src)

    def test_step10c_source_has_no_probability_inversion(self):
        src = (_PROJECT_ROOT / "src" / "step10c_paired_evaluation_bootstrap.py").read_text(encoding="utf-8")
        self.assertNotIn("1.0 - prob", src)
        self.assertNotIn("1 - prob", src)


# =============================================================================
# Entegrasyon testleri (kucuk sentetik veri, GERCEK model fit'i kullanir)
# =============================================================================
def _make_synthetic_experiment_df(n=150, seed=0, n_pos=40, with_label=True):
    rng = np.random.default_rng(seed)
    burnable_pool = [8, 9, 10]
    df = pd.DataFrame({
        "cell_id": [f"cell_{seed}_{i}" for i in range(n)],
        "spatial_block_id": rng.integers(0, 12, n),
        "valid_for_modeling": True,
        "burnable_tree_shrub_grass": True,
        "burnable_tree_shrub": True,
        "ndvi_mean": rng.normal(0.4, 0.1, n), "elevation_mean": rng.normal(300, 50, n),
        "slope_mean": rng.uniform(0, 20, n), "landcover_dominant": rng.choice(burnable_pool, n),
        "lst_anomaly_mean": rng.normal(0, 1, n), "current_lst_mean": rng.normal(305, 5, n),
        "current_tvdi_mean": rng.uniform(0, 1, n), "tvdi_difference_mean": rng.normal(0, 0.1, n),
        "downscaled_lst_mean": rng.normal(305, 5, n), "fused_lst_mean": rng.normal(305, 5, n),
    })
    if with_label:
        y = np.zeros(n, dtype=int)
        y[:n_pos] = 1
        rng.shuffle(y)
        df["burned"] = y
    return df


class TestTargetLabelIndependenceIntegration(unittest.TestCase):
    def setUp(self):
        import src.step10b_label_blind_adaptation as s10b
        self.s10b = s10b
        self.source_df = _make_synthetic_experiment_df(n=150, seed=1, n_pos=40, with_label=True)
        self.target_full_with_label = _make_synthetic_experiment_df(n=100, seed=2, n_pos=32, with_label=True)
        self.target_X = self.s10b.strip_target_to_label_blind(self.target_full_with_label)

    def test_target_X_never_contains_burned(self):
        self.assertNotIn("burned", self.target_X.columns)

    def test_prediction_generation_works_without_target_label(self):
        preds, stats = self.s10b.generate_predictions_for_direction(
            self.source_df, self.target_X, "src_exp", "tgt_exp", random_state=42,
        )
        self.assertFalse(preds.empty)
        self.assertNotIn("burned", preds.columns)
        self.assertEqual(set(preds["adaptation_method"].unique()), set(("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore")))
        self.assertEqual(set(preds["model_family"].unique()), {"baseline", "thermal"})

    def test_permuting_target_y_does_not_change_predictions(self):
        preds1, _ = self.s10b.generate_predictions_for_direction(
            self.source_df, self.target_X, "src_exp", "tgt_exp", random_state=42,
        )
        permuted_full = self.target_full_with_label.copy()
        rng = np.random.default_rng(999)
        permuted_full["burned"] = rng.permutation(permuted_full["burned"].to_numpy())
        target_X_permuted = self.s10b.strip_target_to_label_blind(permuted_full)

        preds2, _ = self.s10b.generate_predictions_for_direction(
            self.source_df, target_X_permuted, "src_exp", "tgt_exp", random_state=42,
        )
        sort_cols = ["model_family", "adaptation_method", "target_cell_id"]
        pd.testing.assert_frame_equal(
            preds1.sort_values(sort_cols).reset_index(drop=True),
            preds2.sort_values(sort_cols).reset_index(drop=True),
        )

    def test_strip_target_to_label_blind_raises_if_burned_somehow_remains(self):
        # assert_label_blind, strip fonksiyonunun SONUNDA da cagrilir --
        # kasitli olarak bozulmus bir kolon kumesiyle test edilemez (fonksiyon
        # kendi ic mantigiyla burned'i zaten cikarir), ancak assert_label_blind'in
        # KENDISI dogrudan test edilmistir (bkz. TestForbiddenFeaturesAndFirewall).
        result = self.s10b.strip_target_to_label_blind(self.target_full_with_label)
        self.assertNotIn("burned", result.columns)


class TestPreregistrationImmutability(unittest.TestCase):
    def setUp(self):
        import tempfile
        import src.step10a_preregistration_and_audit as s10a
        self.s10a = s10a
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_first_creation_writes_manifest_with_analysis_id(self):
        import unittest.mock as mock
        with mock.patch.object(self.s10a, "step10_output_dir", return_value=self.tmp_dir):
            result = self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=False, dry_run=False)
        manifest_path = self.tmp_dir / "step10_preregistration.json"
        self.assertTrue(manifest_path.exists())
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(result["analysis_id"], on_disk["analysis_id"])

    def test_reuse_is_idempotent_same_analysis_id_and_bytes(self):
        import unittest.mock as mock
        with mock.patch.object(self.s10a, "step10_output_dir", return_value=self.tmp_dir):
            r1 = self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=False, dry_run=False)
            before = (self.tmp_dir / "step10_preregistration.json").read_bytes()
            r2 = self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=True, dry_run=False)
            after = (self.tmp_dir / "step10_preregistration.json").read_bytes()
        self.assertEqual(r1["analysis_id"], r2["analysis_id"])
        self.assertEqual(before, after, "--force PREREGISTRATION dosyasini DEGISTIRMEMELI.")

    def test_changed_scientific_config_fails_even_with_force(self):
        import unittest.mock as mock
        with mock.patch.object(self.s10a, "step10_output_dir", return_value=self.tmp_dir):
            self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=False, dry_run=False)
            altered_config = self.s10a.build_scientific_config("manavgat_2021", "bejis_2022")
            altered_config["random_state"] = 999999  # KASITLI bilimsel degisiklik
            with mock.patch.object(self.s10a, "build_scientific_config", return_value=altered_config):
                with self.assertRaises(self.s10a.Step10Error):
                    self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=True, dry_run=False)

    def test_dry_run_writes_nothing(self):
        import unittest.mock as mock
        with mock.patch.object(self.s10a, "step10_output_dir", return_value=self.tmp_dir):
            self.s10a.main(source_id="manavgat_2021", target_id="bejis_2022", force=False, dry_run=True)
        self.assertEqual(list(self.tmp_dir.glob("*")), [])


class TestWithinRegionAlignmentFailFast(unittest.TestCase):
    def test_missing_oof_rows_fail_fast(self):
        import tempfile
        import unittest.mock as mock
        import src.step10c_paired_evaluation_bootstrap as s10c

        predictions_df = pd.DataFrame({
            "direction": ["src_to_tgt"] * 4, "source_experiment": ["src"] * 4, "target_experiment": ["tgt"] * 4,
            "population": ["burnable_tree_shrub_grass"] * 4, "target_cell_id": ["c0", "c1", "c2", "c3"],
            "target_spatial_block_id": [0, 0, 1, 1], "model_family": ["baseline"] * 4,
            "adaptation_method": ["raw_source_only"] * 4, "prediction_probability": [0.1, 0.2, 0.3, 0.4],
        })
        target_full = pd.DataFrame({
            "cell_id": ["c0", "c1", "c2", "c3"], "spatial_block_id": [0, 0, 1, 1],
            "burned": [0, 1, 0, 1], "valid_for_modeling": True, "burnable_tree_shrub_grass": True,
        })
        # OOF KASITLI olarak eksik (yalnizca c0,c1 var; c2,c3 YOK).
        oof_df = pd.DataFrame({
            "cell_id": ["c0", "c1"], "spatial_block_id": [0, 0], "population": ["burnable_tree_shrub_grass"] * 2,
            "y_prob_baseline": [0.15, 0.25], "y_prob_thermal": [0.15, 0.25],
        })

        with tempfile.TemporaryDirectory() as td:
            oof_path = Path(td) / "oof.parquet"
            oof_df.to_parquet(oof_path)
            with mock.patch.object(s10c, "load_step8a_dataset", return_value=target_full), \
                 mock.patch.object(s10c, "resolve_step8b_predictions_path", return_value=oof_path):
                with self.assertRaises(s10c.Step10Error):
                    s10c.build_aligned_direction_frame(predictions_df, "src_to_tgt", "src", "tgt")

    def test_duplicate_cell_id_in_oof_fails_fast(self):
        import tempfile
        import unittest.mock as mock
        import src.step10c_paired_evaluation_bootstrap as s10c

        predictions_df = pd.DataFrame({
            "direction": ["src_to_tgt"] * 2, "source_experiment": ["src"] * 2, "target_experiment": ["tgt"] * 2,
            "population": ["burnable_tree_shrub_grass"] * 2, "target_cell_id": ["c0", "c1"],
            "target_spatial_block_id": [0, 0], "model_family": ["baseline"] * 2,
            "adaptation_method": ["raw_source_only"] * 2, "prediction_probability": [0.1, 0.2],
        })
        target_full = pd.DataFrame({
            "cell_id": ["c0", "c1"], "spatial_block_id": [0, 0], "burned": [0, 1],
            "valid_for_modeling": True, "burnable_tree_shrub_grass": True,
        })
        # OOF'ta cell_id BENZERSIZ degil (c0 iki kez).
        oof_df = pd.DataFrame({
            "cell_id": ["c0", "c0", "c1"], "spatial_block_id": [0, 0, 0], "population": ["burnable_tree_shrub_grass"] * 3,
            "y_prob_baseline": [0.1, 0.9, 0.2], "y_prob_thermal": [0.1, 0.9, 0.2],
        })
        with tempfile.TemporaryDirectory() as td:
            oof_path = Path(td) / "oof.parquet"
            oof_df.to_parquet(oof_path)
            with mock.patch.object(s10c, "load_step8a_dataset", return_value=target_full), \
                 mock.patch.object(s10c, "resolve_step8b_predictions_path", return_value=oof_path):
                with self.assertRaises(s10c.Step10Error):
                    s10c.build_aligned_direction_frame(predictions_df, "src_to_tgt", "src", "tgt")


class TestReproductionCheckLogic(unittest.TestCase):
    """Kategori 1 (raw reprodüksiyon): verify_raw_reproduction'in FAIL-FAST
    mantigini, gercek model egitimi olmadan, KUCUK/tasnif edilmis
    point_metrics + gecici Step9B metrics.json ile test eder. Gercek
    uctan-uca sayisal reprodüksiyon (Step9B ile TAM 0.0 fark), bu oturumda
    src/step10c_paired_evaluation_bootstrap.py ile GERCEK sentetik veri
    uzerinde AYRICA dogrulanmistir (bkz. degisiklik ozeti)."""

    def setUp(self):
        import tempfile
        import src.step10c_paired_evaluation_bootstrap as s10c
        self.s10c = s10c
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_step9b_metrics(self, direction, population, roc_auc, pr_auc, brier_score=0.20):
        # The frozen Step9B `results[]` rows carry all three metrics named in
        # REQUIRED_STEP9_RAW_METRICS -- roc_auc, pr_auc AND brier_score --
        # inside each metric block. `_validated_step9_transfer_metrics`
        # refuses to extract a reference that is missing any of them rather
        # than substituting a placeholder, so the fixture must supply the
        # real schema. Only roc_auc/pr_auc take part in the 1e-6 reproduction
        # comparison; brier_score is carried through for provenance.
        payload = {"results": [{
            "transfer_direction": direction, "population": population, "skipped": False,
            "baseline_metrics": {
                "roc_auc": roc_auc, "pr_auc": pr_auc, "brier_score": brier_score,
            },
            "thermal_metrics": {
                "roc_auc": roc_auc + 0.05, "pr_auc": pr_auc + 0.01,
                "brier_score": brier_score - 0.02,
            },
        }]}
        path = self.tmp_dir / "cross_region_transfer_metrics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_matching_raw_metrics_pass(self):
        import unittest.mock as mock
        direction = "src_to_tgt"
        point_metrics = {
            "raw_source_only": {
                "baseline": {"roc_auc": 0.60, "pr_auc": 0.10},
                "thermal": {"roc_auc": 0.65, "pr_auc": 0.11},
            },
        }
        self._write_step9b_metrics(direction, s10.PRIMARY_POPULATION, 0.60, 0.10)
        with mock.patch.object(self.s10c, "resolve_step9b_metrics_path", return_value=self.tmp_dir / "cross_region_transfer_metrics.json"):
            result = self.s10c.verify_raw_reproduction(point_metrics, "src", "tgt", direction)
        self.assertTrue(result["all_within_tolerance"])

    def test_mismatched_raw_metrics_raise_and_block_adapted_report(self):
        import unittest.mock as mock
        direction = "src_to_tgt"
        point_metrics = {
            "raw_source_only": {
                "baseline": {"roc_auc": 0.75, "pr_auc": 0.10},  # 0.60'tan FARKLI -- reprodüksiyon BASARISIZ
                "thermal": {"roc_auc": 0.65, "pr_auc": 0.11},
            },
        }
        self._write_step9b_metrics(direction, s10.PRIMARY_POPULATION, 0.60, 0.10)
        with mock.patch.object(self.s10c, "resolve_step9b_metrics_path", return_value=self.tmp_dir / "cross_region_transfer_metrics.json"):
            with self.assertRaises(self.s10c.Step10Error):
                self.s10c.verify_raw_reproduction(point_metrics, "src", "tgt", direction)

    def test_missing_step9b_metrics_file_raises(self):
        import unittest.mock as mock
        with mock.patch.object(self.s10c, "resolve_step9b_metrics_path", return_value=self.tmp_dir / "does_not_exist.json"):
            with self.assertRaises(self.s10c.Step10Error):
                self.s10c.verify_raw_reproduction({"raw_source_only": {"baseline": {}, "thermal": {}}}, "src", "tgt", "src_to_tgt")

    def test_reproduction_uses_original_pair_path_not_swapped_per_direction(self):
        """KRITIK regresyon testi: Step9B'nin cikti dizini HER ZAMAN orijinal
        (source_id, target_id) ciftine gore sabittir -- ters yon icin
        source/target ID'leri path cozumlemesinde YER DEGISTIRMEMELIDIR."""
        import unittest.mock as mock
        calls = []

        def _fake_resolver(a, b):
            calls.append((a, b))
            return self.tmp_dir / "cross_region_transfer_metrics.json"

        point_metrics = {"raw_source_only": {"baseline": {"roc_auc": 0.6, "pr_auc": 0.1}, "thermal": {"roc_auc": 0.65, "pr_auc": 0.11}}}
        self._write_step9b_metrics("tgt_to_src", s10.PRIMARY_POPULATION, 0.6, 0.1)
        with mock.patch.object(self.s10c, "resolve_step9b_metrics_path", side_effect=_fake_resolver):
            self.s10c.verify_raw_reproduction(point_metrics, "src", "tgt", "tgt_to_src")
        # original_source_id/original_target_id ("src","tgt") HER ZAMAN AYNI SIRAYLA cagrilmali.
        self.assertEqual(calls, [("src", "tgt")])

    def test_both_directions_resolve_the_same_pair_path_only_the_slice_differs(self):
        """Ayni pair dosyasi her iki yon icin de cozulur; degisen tek sey,
        dosya ICINDEN secilen `transfer_direction` satiridir. Fiziksel yol
        yon basina ASLA swap edilmez."""
        import unittest.mock as mock

        payload = {"results": [
            {
                "transfer_direction": "src_to_tgt", "population": s10.PRIMARY_POPULATION,
                "skipped": False,
                "baseline_metrics": {"roc_auc": 0.60, "pr_auc": 0.10, "brier_score": 0.20},
                "thermal_metrics": {"roc_auc": 0.65, "pr_auc": 0.11, "brier_score": 0.18},
            },
            {
                "transfer_direction": "tgt_to_src", "population": s10.PRIMARY_POPULATION,
                "skipped": False,
                "baseline_metrics": {"roc_auc": 0.40, "pr_auc": 0.05, "brier_score": 0.30},
                "thermal_metrics": {"roc_auc": 0.45, "pr_auc": 0.06, "brier_score": 0.28},
            },
        ]}
        path = self.tmp_dir / "cross_region_transfer_metrics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        calls = []

        def _fake_resolver(a, b):
            calls.append((a, b))
            return path

        expected = {
            "src_to_tgt": (0.60, 0.65),
            "tgt_to_src": (0.40, 0.45),
        }
        with mock.patch.object(self.s10c, "resolve_step9b_metrics_path", side_effect=_fake_resolver):
            for direction, (baseline_roc, thermal_roc) in expected.items():
                reference = self.s10c.resolve_step9_raw_reference("src", "tgt", direction)
                self.assertEqual(reference["metrics"]["baseline"]["roc_auc"], baseline_roc)
                self.assertEqual(reference["metrics"]["thermal"]["roc_auc"], thermal_roc)
                # Pair identity is the ORIGINAL one for both directions.
                self.assertEqual(reference["root_source_experiment_id"], "src")
                self.assertEqual(reference["root_target_experiment_id"], "tgt")

        # Same physical path, same argument order, for both directions.
        self.assertEqual(calls, [("src", "tgt"), ("src", "tgt")])


class TestStep10ReportOnlyQA(unittest.TestCase):
    """Report-only regression coverage against copied frozen scientific inputs."""

    def setUp(self):
        import shutil
        import tempfile
        import src.step10d_final_report as s10d

        self.s10d = s10d
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name)
        self.real_output = _PROJECT_ROOT / "outputs" / "cross_region" / "manavgat_2021__bejis_2022" / "step10"
        for name in s10d.PROTECTED_INPUT_FILENAMES:
            shutil.copy2(self.real_output / name, self.output_dir / name)
        self.analysis_id = json.loads((self.output_dir / "step10_preregistration.json").read_text())["analysis_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_output(self):
        import unittest.mock as mock
        return mock.patch.object(self.s10d, "step10_output_dir", return_value=self.output_dir)

    def _build(self):
        with self._patch_output():
            return self.s10d.build_final_report("manavgat_2021", "bejis_2022", self.analysis_id)

    def test_chance_status_logic_all_three_cases(self):
        classify = self.s10d.classify_chance_status
        self.assertEqual(classify(0.2, 0.49), "bootstrap_supported_below_chance")
        self.assertEqual(classify(0.51, 0.8), "bootstrap_supported_above_chance")
        self.assertEqual(classify(0.4, 0.6), "chance_level_not_excluded")

    def test_paired_support_logic_all_three_cases(self):
        classify = self.s10d.classify_paired_difference_support
        self.assertEqual(classify(0.01, 0.2), "bootstrap_supported_positive")
        self.assertEqual(classify(-0.2, -0.01), "bootstrap_supported_negative")
        self.assertEqual(classify(-0.01, 0.01), "uncertain_interval_includes_zero")

    def test_tables_are_complete_and_brier_is_unavailable(self):
        report = self._build()
        self.assertEqual(len(report["target_performance"]), 12)
        self.assertEqual(len(report["paired_adaptation_differences"]), 24)
        self.assertEqual(len(report["within_transfer_decomposition"]), 16)
        self.assertTrue(all(row["brier"] is None for row in report["target_performance"]))

    def test_frozen_interpretations_are_preserved(self):
        report = self._build()
        per_direction = report["per_direction_interpretation"]
        forward = per_direction["manavgat_2021_to_bejis_2022"]["by_model_family"]
        reverse = per_direction["bejis_2022_to_manavgat_2021"]["by_model_family"]
        self.assertEqual(forward["thermal"]["covariate_recovery_zscore"], "bootstrap_supported_positive")
        self.assertEqual(reverse["thermal"]["covariate_recovery_zscore"], "uncertain_interval_includes_zero")
        self.assertEqual(forward["thermal"]["coral_vs_zscore"], "bootstrap_supported_positive")
        self.assertEqual(reverse["thermal"]["coral_vs_zscore"], "bootstrap_supported_positive")
        self.assertEqual(forward["baseline"]["coral_vs_zscore"], "uncertain_interval_includes_zero")
        self.assertEqual(reverse["baseline"]["coral_vs_zscore"], "uncertain_interval_includes_zero")

    def test_thermal_chance_status_is_method_specific(self):
        report = self._build()
        per_direction = report["per_direction_interpretation"]
        forward = per_direction["manavgat_2021_to_bejis_2022"]["by_model_family"]["thermal"]["separate_questions"]
        reverse = per_direction["bejis_2022_to_manavgat_2021"]["by_model_family"]["thermal"]["separate_questions"]
        self.assertEqual(forward["regionwise_zscore_performance_relative_to_chance"], "chance_level_not_excluded")
        self.assertEqual(forward["coral_after_regionwise_zscore_performance_relative_to_chance"], "chance_level_not_excluded")
        self.assertEqual(reverse["regionwise_zscore_performance_relative_to_chance"], "bootstrap_supported_below_chance")
        self.assertEqual(reverse["coral_after_regionwise_zscore_performance_relative_to_chance"], "bootstrap_supported_above_chance")

    def test_mixed_analysis_ids_fail_fast(self):
        path = self.output_dir / "step10_adaptation_statistics.json"
        payload = json.loads(path.read_text())
        payload["analysis_id"] = "f" * 64
        path.write_text(json.dumps(payload))
        with self._patch_output(), self.assertRaises(s10.Step10Error):
            self.s10d.build_final_report("manavgat_2021", "bejis_2022", self.analysis_id)

    def test_missing_frozen_input_fails_fast(self):
        (self.output_dir / "step10_metrics.csv").unlink()
        with self._patch_output(), self.assertRaises(s10.Step10Error):
            self.s10d.run_step10d(
                "manavgat_2021", "bejis_2022", self.analysis_id,
                report_only_generation=True,
            )

    def test_prediction_target_label_firewall(self):
        self.assertEqual(
            self.s10d.find_prohibited_prediction_columns(
                ["target_experiment", "target_cell_id", "BurnDate", "y_true", "outcome"]
            ),
            ["BurnDate", "outcome", "y_true"],
        )
        with self._patch_output():
            qa = self.s10d.inspect_predictions_for_qa(self.output_dir / "step10_predictions.parquet")
        self.assertFalse(qa["target_label_present_in_predictions"])

    def test_only_final_reports_change_and_protected_hashes_match(self):
        before_names = {path.name for path in self.output_dir.iterdir()}
        before_hashes = self.s10d.protected_input_hashes(self.output_dir)
        with self._patch_output():
            report = self.s10d.run_step10d(
                "manavgat_2021", "bejis_2022", self.analysis_id,
                force=True, report_only_generation=True,
            )
        after_hashes = self.s10d.protected_input_hashes(self.output_dir)
        after_names = {path.name for path in self.output_dir.iterdir()}
        self.assertEqual(before_hashes, after_hashes)
        self.assertEqual(after_names - before_names, set(self.s10d.REPORT_FILENAMES))
        self.assertEqual(report["report_only_generation"]["scientific_stages_called"], [])

    def test_report_output_is_deterministic(self):
        with self._patch_output():
            self.s10d.run_step10d("manavgat_2021", "bejis_2022", self.analysis_id, force=True, report_only_generation=True)
            first = [(self.output_dir / name).read_bytes() for name in self.s10d.REPORT_FILENAMES]
            self.s10d.run_step10d("manavgat_2021", "bejis_2022", self.analysis_id, force=True, report_only_generation=True)
            second = [(self.output_dir / name).read_bytes() for name in self.s10d.REPORT_FILENAMES]
        self.assertEqual(first, second)

    def test_report_only_dispatch_calls_step10d_not_scientific_stages(self):
        import unittest.mock as mock
        import core.step10_shared as shared
        import scripts.run_step10_self_calibrated_transfer as runner
        import src.step10a_preregistration_and_audit as s10a
        import src.step10b_label_blind_adaptation as s10b
        import src.step10c_paired_evaluation_bootstrap as s10c

        fake_report = {"analysis_id": self.analysis_id}
        with mock.patch.object(shared, "step10_output_dir", return_value=self.output_dir), \
             mock.patch.object(self.s10d, "run_step10d", return_value=fake_report) as step10d, \
             mock.patch.object(s10a, "main") as step10a, \
             mock.patch.object(s10b, "run_step10b") as step10b, \
             mock.patch.object(s10c, "run_step10c") as step10c:
            result = runner.main("manavgat_2021", "bejis_2022", reverse=True, report_only=True)
        step10d.assert_called_once()
        step10a.assert_not_called()
        step10b.assert_not_called()
        step10c.assert_not_called()
        self.assertTrue(result["report_only"])

    def test_report_only_dry_run_writes_nothing(self):
        import unittest.mock as mock
        import scripts.run_step10_self_calibrated_transfer as runner

        before = {path.name: path.read_bytes() for path in self.output_dir.iterdir()}
        with mock.patch.object(self.s10d, "step10_output_dir", return_value=self.output_dir):
            result = runner.main(
                "manavgat_2021", "bejis_2022", reverse=True,
                dry_run=True, report_only=True,
            )
        after = {path.name: path.read_bytes() for path in self.output_dir.iterdir()}
        self.assertEqual(before, after)
        self.assertFalse(result["ran"])

    def test_direct_runner_parser_accepts_report_only(self):
        from scripts.run_step10_self_calibrated_transfer import parse_args
        args = parse_args(["--source", "manavgat_2021", "--target", "bejis_2022", "--reverse", "--report-only"])
        self.assertTrue(args.report_only)


if __name__ == "__main__":
    unittest.main()