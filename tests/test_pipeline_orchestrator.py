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
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.pipeline_orchestrator as orch


class TestStageOrdering(unittest.TestCase):
    def test_stage_order_is_the_documented_sequence(self):
        self.assertEqual(orch.STAGE_ORDER, ["gate", "predictors", "step7", "step8"])

    def test_full_range_returns_all_stages(self):
        self.assertEqual(orch.validate_stage_range("gate", "step8"), orch.STAGE_ORDER)

    def test_single_stage_range(self):
        self.assertEqual(orch.validate_stage_range("predictors", "predictors"), ["predictors"])

    def test_partial_range(self):
        self.assertEqual(orch.validate_stage_range("predictors", "step8"), ["predictors", "step7", "step8"])

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


if __name__ == "__main__":
    unittest.main()