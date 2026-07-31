"""
tests/test_pipeline_orchestrator.py

core/pipeline_orchestrator.py için odaklı unittest testleri.

Bu testler GEE kimlik doğrulaması, gerçek raster/model verisi veya ağ
erişimi GEREKTİRMEZ -- yalnızca saf orkestrasyon mantığını (asama sırası
doğrulama, namespace güvenlik kontrolü, dry-run'ın hiçbir side-effect
üretmediği) test eder.

Çalıştırma:
    python -m unittest discover -s tests
    (pytest kuruluysa: python -m pytest tests)
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.pipeline_orchestrator as orch


class TestStageOrdering(unittest.TestCase):
    def test_stage_order_is_the_documented_sequence(self):
        self.assertEqual(orch.STAGE_ORDER, ["gate", "predictors", "scene-provenance", "step7", "seam-audit", "seam-localization", "step8"])

    def test_full_range_returns_all_stages(self):
        self.assertEqual(orch.validate_stage_range("gate", "step8"), orch.STAGE_ORDER)

    def test_single_stage_range(self):
        self.assertEqual(orch.validate_stage_range("predictors", "predictors"), ["predictors"])

    def test_partial_range(self):
        self.assertEqual(orch.validate_stage_range("predictors", "step8"), ["predictors", "scene-provenance", "step7", "seam-audit", "seam-localization", "step8"])

    def test_reversed_range_raises(self):
        with self.assertRaises(SystemExit):
            orch.validate_stage_range("step8", "gate")

    def test_unknown_from_stage_raises(self):
        with self.assertRaises(SystemExit):
            orch.validate_stage_range("not_a_stage", "step8")

    def test_unknown_to_stage_raises(self):
        with self.assertRaises(SystemExit):
            orch.validate_stage_range("gate", "not_a_stage")


class TestNamespaceSafety(unittest.TestCase):
    def test_kozan_context_is_exempt(self):
        # is_kozan=True -> legacy paylasilan yollar KASITLI olarak izinlidir.
        ctx = {
            "is_kozan": True,
            "experiment_id": "kozan_2023",
            "output_root": Path("/tmp/does_not_matter"),
            "data_root": Path("/completely/different/legacy/path"),
        }
        # Hicbir exception beklenmiyor.
        orch._assert_context_is_safely_namespaced(ctx)

    def test_non_kozan_context_within_output_root_passes(self):
        output_root = Path("/tmp/outputs/experiments/manavgat_2021")
        ctx = {
            "is_kozan": False,
            "experiment_id": "manavgat_2021",
            "output_root": output_root,
            "data_root": output_root / "data",
            "step5_output_dir": output_root / "step5",
            "step5c_output_dir": output_root / "step5c",
            "gate_labels_dir": output_root / "validation" / "labels",
            "step7a_output_dir": output_root / "step7a",
            "step8a_output_dir": output_root / "step8a",
        }
        orch._assert_context_is_safely_namespaced(ctx)

    def test_non_kozan_context_leaking_to_legacy_path_raises(self):
        output_root = Path("/tmp/outputs/experiments/manavgat_2021")
        ctx = {
            "is_kozan": False,
            "experiment_id": "manavgat_2021",
            "output_root": output_root,
            "data_root": output_root / "data",
            # KASITLI ihlal: step5_output_dir, output_root disinda (legacy
            # paylasilan Kozan yoluna benzer bir sizinti).
            "step5_output_dir": Path("/tmp/outputs/step5"),
        }
        with self.assertRaises(SystemExit):
            orch._assert_context_is_safely_namespaced(ctx)

    def test_non_kozan_context_wrong_output_root_pattern_raises(self):
        ctx = {
            "is_kozan": False,
            "experiment_id": "manavgat_2021",
            # outputs/experiments/<experiment_id> deseninde DEGIL.
            "output_root": Path("/tmp/outputs/shared/manavgat_2021"),
        }
        with self.assertRaises(SystemExit):
            orch._assert_context_is_safely_namespaced(ctx)


class TestDescribeExperimentPlan(unittest.TestCase):
    def test_unknown_experiment_raises_value_error(self):
        with self.assertRaises(ValueError):
            orch.describe_experiment_plan(
                "definitely_not_a_registered_experiment", "gate", "step8",
                "local-only", False,
            )

    def test_disabled_experiment_raises_value_error(self):
        # zamora_2022, core/regions.py kaydında enabled=False.
        with self.assertRaises(ValueError):
            orch.describe_experiment_plan("zamora_2022", "gate", "step8", "local-only", False)

    def test_invalid_predictor_mode_raises(self):
        with self.assertRaises(SystemExit):
            orch.describe_experiment_plan("manavgat_2021", "gate", "step8", "not-a-mode", False)

    def test_invalid_stage_range_raises_before_touching_context(self):
        with self.assertRaises(SystemExit):
            orch.describe_experiment_plan("manavgat_2021", "step8", "gate", "local-only", False)

    def test_known_enabled_experiment_produces_namespaced_plan(self):
        plan = orch.describe_experiment_plan("manavgat_2021", "gate", "step8", "local-only", False)
        self.assertEqual(plan["experiment_id"], "manavgat_2021")
        self.assertEqual(plan["stages"], orch.STAGE_ORDER)
        self.assertFalse(plan["is_kozan"])
        self.assertIn("experiments", Path(plan["output_root"]).parts)
        self.assertIn("manavgat_2021", Path(plan["output_root"]).parts)
        self.assertEqual(Path(plan["seam_audit_output_dir"]).parts[-2:], ("seam_audit", "v2"))


class TestDryRunNoExecution(unittest.TestCase):
    """dry_run=True verildiginde hicbir dosyanin OLUSTURULMADIGINI dogrular."""

    def _snapshot_experiment_dirs(self) -> set[Path]:
        from core.paths import PROJECT_ROOT
        root = PROJECT_ROOT / "outputs" / "experiments"
        if not root.exists():
            return set()
        return set(root.rglob("*"))

    def test_dry_run_creates_no_new_files(self):
        before = self._snapshot_experiment_dirs()
        orch.run_experiment_plan(
            experiment_id="manavgat_2021", from_stage="gate", to_stage="step8",
            predictor_mode="local-only", export_labels=False, dry_run=True, force=False,
        )
        after = self._snapshot_experiment_dirs()
        new_files = {p for p in after if p not in before and p.is_file()}
        self.assertEqual(new_files, set(), f"dry-run beklenmedik dosyalar olusturdu: {new_files}")

    def test_dry_run_stage_results_report_not_ran(self):
        result = orch.run_experiment_plan(
            experiment_id="manavgat_2021", from_stage="gate", to_stage="gate",
            predictor_mode="local-only", export_labels=False, dry_run=True, force=False,
        )
        gate_result = result["stage_results"]["gate"]
        self.assertFalse(gate_result.get("ran"))
        self.assertEqual(gate_result.get("reason"), "dry_run")


class TestStep8RobustnessDispatch(unittest.TestCase):
    def test_orchestrator_reuses_thin_runner(self):
        with patch(
            "scripts.run_step8_large_block_robustness.main",
            return_value={"ran": False, "dry_run": True},
        ) as mocked:
            result = orch.run_step8_robustness_stage(
                ["manavgat_2021", "bejis_2022"], [10, 20], dry_run=True, force=False
            )
        self.assertFalse(result["ran"])
        mocked.assert_called_once_with(
            experiments=["manavgat_2021", "bejis_2022"],
            block_sizes_cells=[10, 20],
            dry_run=True,
            force=False,
        )


class TestStep8BigBlockRobustnessDispatch(unittest.TestCase):
    def test_orchestrator_reuses_thin_runner(self):
        with patch(
            "scripts.run_step8_big_block_robustness.main",
            return_value={"ran": False, "dry_run": True},
        ) as mocked:
            result = orch.run_step8_big_block_robustness_stage(
                "mugla_2021", [10, 20], dry_run=True, force=False
            )
        self.assertFalse(result["ran"])
        mocked.assert_called_once_with(
            experiment="mugla_2021", block_sizes=[10, 20], dry_run=True, force=False,
            regenerate_reports_only=False, output_root=None,
        )

    def test_orchestrator_accepts_arbitrary_experiment_id(self):
        with patch(
            "scripts.run_step8_big_block_robustness.main",
            return_value={"ran": False, "dry_run": True},
        ) as mocked:
            orch.run_step8_big_block_robustness_stage(
                "some_future_experiment", [10, 20], dry_run=True, force=False
            )
        mocked.assert_called_once_with(
            experiment="some_future_experiment", block_sizes=[10, 20], dry_run=True, force=False,
            regenerate_reports_only=False, output_root=None,
        )


if __name__ == "__main__":
    unittest.main()


class TestMarginalAoADispatch(unittest.TestCase):
    """The marginal AoA stage is a thin dispatch: it must forward every
    parameter -- including the output_root/experiments_root injection points
    -- to the runner unchanged, and add no scientific logic of its own."""

    def test_orchestrator_forwards_exact_kwargs(self):
        with patch(
            "scripts.run_marginal_area_of_applicability.main",
            return_value={"ran": False, "dry_run": True},
        ) as mocked:
            result = orch.run_marginal_aoa_stage(
                experiments=["a_experiment", "b_experiment"],
                all_enabled=False, dry_run=True, force=False,
            )
        self.assertFalse(result["ran"])
        mocked.assert_called_once_with(
            experiments=["a_experiment", "b_experiment"],
            all_enabled=False, dry_run=True, force=False,
            output_root=None, experiments_root=None,
        )

    def test_orchestrator_carries_injection_roots(self):
        with patch(
            "scripts.run_marginal_area_of_applicability.main",
            return_value={"ran": True},
        ) as mocked:
            orch.run_marginal_aoa_stage(
                experiments=None, all_enabled=True, dry_run=False, force=True,
                output_root="/tmp/injected_out", experiments_root="/tmp/injected_exp",
            )
        mocked.assert_called_once_with(
            experiments=None, all_enabled=True, dry_run=False, force=True,
            output_root="/tmp/injected_out", experiments_root="/tmp/injected_exp",
        )

    def test_orchestrator_accepts_arbitrary_experiment_ids(self):
        with patch(
            "scripts.run_marginal_area_of_applicability.main",
            return_value={"ran": False},
        ) as mocked:
            orch.run_marginal_aoa_stage(
                experiments=["some_future_experiment"], all_enabled=False,
                dry_run=True, force=False,
            )
        self.assertEqual(
            mocked.call_args.kwargs["experiments"], ["some_future_experiment"]
        )


class TestMarginalAoARunnerPassthrough(unittest.TestCase):
    def test_runner_forwards_injection_roots_to_run_analysis(self):
        from scripts import run_marginal_area_of_applicability as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(
                experiments=["a_experiment", "b_experiment"], all_enabled=False,
                dry_run=True, force=False,
                output_root="/tmp/out_root", experiments_root="/tmp/exp_root",
            )
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["experiments"], ["a_experiment", "b_experiment"])
        self.assertTrue(kwargs["dry_run"])
        self.assertEqual(str(kwargs["output_root"]), "/tmp/out_root")
        self.assertEqual(str(kwargs["experiments_root"]), "/tmp/exp_root")

    def test_runner_defaults_both_roots_to_none(self):
        from scripts import run_marginal_area_of_applicability as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(experiments=["a_experiment", "b_experiment"], dry_run=True)
        kwargs = mocked.call_args.kwargs
        self.assertIsNone(kwargs["output_root"])
        self.assertIsNone(kwargs["experiments_root"])


class TestWindowClosureSensitivityDispatch(unittest.TestCase):
    """Thin dispatch: every parameter, including the injection roots, must
    reach the runner unchanged."""

    def test_orchestrator_forwards_exact_kwargs(self):
        with patch(
            "scripts.run_window_closure_sensitivity.main",
            return_value={"ran": False, "dry_run": True},
        ) as mocked:
            result = orch.run_window_closure_sensitivity_stage(
                experiment_id="some_future_experiment", shifts=[0, 7, 14],
                from_stage="plan", to_stage="compare", dry_run=True,
                force=False, resume=False,
            )
        self.assertFalse(result["ran"])
        mocked.assert_called_once_with(
            experiment_id="some_future_experiment", shifts=[0, 7, 14],
            from_stage="plan", to_stage="compare",
            dry_run=True, force=False, resume=False,
            output_root=None, experiments_root=None,
        )

    def test_orchestrator_forwards_the_local_downstream_stage(self):
        with patch(
            "scripts.run_window_closure_sensitivity.main",
            return_value={"ran": True, "stages_run": ["local-downstream"]},
        ) as mocked:
            orch.run_window_closure_sensitivity_stage(
                experiment_id="some_future_experiment", shifts=[0, 7, 14],
                from_stage="local-downstream", to_stage="local-downstream",
                dry_run=False, force=False, resume=True,
            )
        mocked.assert_called_once_with(
            experiment_id="some_future_experiment", shifts=[0, 7, 14],
            from_stage="local-downstream", to_stage="local-downstream",
            dry_run=False, force=False, resume=True,
            output_root=None, experiments_root=None,
        )

    def test_orchestrator_carries_injection_roots(self):
        with patch(
            "scripts.run_window_closure_sensitivity.main",
            return_value={"ran": True},
        ) as mocked:
            orch.run_window_closure_sensitivity_stage(
                experiment_id="e", shifts=[0, 7], from_stage="model",
                to_stage="compare", dry_run=False, force=True, resume=True,
                output_root="/tmp/injected_out", experiments_root="/tmp/injected_exp",
            )
        mocked.assert_called_once_with(
            experiment_id="e", shifts=[0, 7], from_stage="model", to_stage="compare",
            dry_run=False, force=True, resume=True,
            output_root="/tmp/injected_out", experiments_root="/tmp/injected_exp",
        )


class TestWindowClosureRunnerPassthrough(unittest.TestCase):
    def test_runner_forwards_injection_roots(self):
        from scripts import run_window_closure_sensitivity as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(
                experiment_id="e", shifts=[0, 7, 14], dry_run=True,
                output_root="/tmp/out_root", experiments_root="/tmp/exp_root",
            )
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["experiment_id"], "e")
        self.assertEqual(tuple(kwargs["shifts"]), (0, 7, 14))
        self.assertTrue(kwargs["dry_run"])
        self.assertEqual(str(kwargs["output_root"]), "/tmp/out_root")
        self.assertEqual(str(kwargs["experiments_root"]), "/tmp/exp_root")

    def test_runner_defaults_shifts_and_roots(self):
        from scripts import run_window_closure_sensitivity as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(experiment_id="e", dry_run=True)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(tuple(kwargs["shifts"]), (0, 7, 14))
        self.assertIsNone(kwargs["output_root"])
        self.assertIsNone(kwargs["experiments_root"])
