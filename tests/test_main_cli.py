"""
tests/test_main_cli.py

scripts/main.py'nin argparse yapısı için odaklı unittest testleri. Hiçbir
alt-komutu GERÇEKTEN ÇALIŞTIRMAZ (yalnızca parser/dispatch davranışını test
eder); GEE/ağ erişimi gerektirmez.

Çalıştırma:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.main import (
    build_parser, cmd_experiment, cmd_legacy, cmd_self_cal_transfer, cmd_shift_audit,
    cmd_step8_robustness, cmd_transfer, cmd_transfer_explore,
    cmd_step10, cmd_large_block_robustness, cmd_concept_shift,
    cmd_step8_big_block_robustness, cmd_marginal_aoa,
    cmd_marginal_aoa_completion, cmd_window_closure_sensitivity,
    cmd_few_shot_recovery,
    cmd_mugla_subsampling,
)


class TestParserStructure(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_bare_invocation_has_no_command(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.command)

    def test_experiment_subcommand_parses(self):
        args = self.parser.parse_args([
            "experiment", "--experiment", "manavgat_2021",
            "--from-stage", "predictors", "--to-stage", "step8",
            "--predictor-mode", "local-only", "--dry-run",
        ])
        self.assertEqual(args.command, "experiment")
        self.assertEqual(args.experiment, "manavgat_2021")
        self.assertEqual(args.from_stage, "predictors")
        self.assertEqual(args.to_stage, "step8")
        self.assertEqual(args.predictor_mode, "local-only")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)
        self.assertFalse(args.export_labels)
        self.assertIsNone(args.seam_products)
        self.assertIsNone(args.seam_scales)
        self.assertIs(args.func, cmd_experiment)

    def test_seam_audit_stage_and_overrides_parse(self):
        args = self.parser.parse_args([
            "experiment", "--experiment", "mugla_2021",
            "--from-stage", "seam-audit", "--to-stage", "seam-audit",
            "--predictor-mode", "local-only", "--dry-run",
            "--seam-products", "current_lst,fused_lst",
            "--seam-scales", "native,modeling_500m",
        ])
        self.assertEqual(args.from_stage, "seam-audit")
        self.assertEqual(args.seam_products, "current_lst,fused_lst")
        self.assertEqual(args.seam_scales, "native,modeling_500m")

    def test_experiment_missing_required_arg_raises_systemexit(self):
        with self.assertRaises(SystemExit):
            # --predictor-mode eksik -> argparse SystemExit firlatir.
            self.parser.parse_args([
                "experiment", "--experiment", "manavgat_2021",
                "--from-stage", "gate", "--to-stage", "step8",
            ])

    def test_experiment_invalid_stage_choice_raises_systemexit(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "experiment", "--experiment", "manavgat_2021",
                "--from-stage", "not_a_stage", "--to-stage", "step8",
                "--predictor-mode", "local-only",
            ])

    def test_experiment_invalid_predictor_mode_raises_systemexit(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "experiment", "--experiment", "manavgat_2021",
                "--from-stage", "gate", "--to-stage", "step8",
                "--predictor-mode", "not-a-mode",
            ])

    def test_transfer_subcommand_parses(self):
        args = self.parser.parse_args([
            "transfer", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--force",
        ])
        self.assertEqual(args.command, "transfer")
        self.assertEqual(args.source, "manavgat_2021")
        self.assertEqual(args.target, "bejis_2022")
        self.assertTrue(args.reverse)
        self.assertTrue(args.force)
        self.assertIs(args.func, cmd_transfer)

    def test_shift_audit_subcommand_parses(self):
        args = self.parser.parse_args([
            "shift-audit", "--source", "manavgat_2021", "--target", "bejis_2022", "--dry-run",
        ])
        self.assertEqual(args.command, "shift-audit")
        self.assertTrue(args.dry_run)
        self.assertIs(args.func, cmd_shift_audit)

    def test_transfer_explore_subcommand_parses(self):
        args = self.parser.parse_args([
            "transfer-explore", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--force", "--bootstrap-replicates", "500", "--seed", "7",
        ])
        self.assertEqual(args.command, "transfer-explore")
        self.assertEqual(args.source, "manavgat_2021")
        self.assertEqual(args.target, "bejis_2022")
        self.assertTrue(args.reverse)
        self.assertTrue(args.force)
        self.assertEqual(args.bootstrap_replicates, 500)
        self.assertEqual(args.seed, 7)
        self.assertIs(args.func, cmd_transfer_explore)

    def test_transfer_explore_default_bootstrap_and_seed(self):
        args = self.parser.parse_args([
            "transfer-explore", "--source", "manavgat_2021", "--target", "bejis_2022", "--dry-run",
        ])
        self.assertEqual(args.bootstrap_replicates, 1000)
        self.assertIsNone(args.seed)

    def test_self_cal_transfer_subcommand_parses(self):
        args = self.parser.parse_args([
            "self-cal-transfer", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--force", "--bootstrap-replicates", "500", "--seed", "7",
        ])
        self.assertEqual(args.command, "self-cal-transfer")
        self.assertEqual(args.source, "manavgat_2021")
        self.assertEqual(args.target, "bejis_2022")
        self.assertTrue(args.reverse)
        self.assertTrue(args.force)
        self.assertFalse(args.report_only)
        self.assertEqual(args.bootstrap_replicates, 500)
        self.assertEqual(args.seed, 7)
        self.assertIs(args.func, cmd_self_cal_transfer)

    def test_self_cal_transfer_report_only_parses(self):
        args = self.parser.parse_args([
            "self-cal-transfer", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--report-only",
        ])
        self.assertTrue(args.report_only)
        self.assertTrue(args.reverse)
        self.assertIs(args.func, cmd_self_cal_transfer)

    def test_self_cal_transfer_default_bootstrap_and_seed(self):
        args = self.parser.parse_args([
            "self-cal-transfer", "--source", "manavgat_2021", "--target", "bejis_2022", "--dry-run",
        ])
        self.assertEqual(args.bootstrap_replicates, 1000)
        self.assertEqual(args.seed, 42)
        self.assertFalse(args.reverse)

    def test_self_cal_transfer_dry_run_dispatches_without_error(self):
        args = self.parser.parse_args([
            "self-cal-transfer", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--dry-run",
        ])
        exit_code = cmd_self_cal_transfer(args)
        self.assertEqual(exit_code, 0)

    def test_self_cal_transfer_dry_run_creates_no_predictions(self):
        from core.step10_shared import step10_output_dir
        output_dir = step10_output_dir("manavgat_2021", "bejis_2022")
        predictions_path = output_dir / "step10_predictions.parquet"
        existed_before = predictions_path.exists()
        args = self.parser.parse_args([
            "self-cal-transfer", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--dry-run",
        ])
        cmd_self_cal_transfer(args)
        if not existed_before:
            self.assertFalse(predictions_path.exists())

    def test_step8_robustness_subcommand_parses_frozen_plan(self):
        args = self.parser.parse_args([
            "step8-robustness",
            "--experiments", "manavgat_2021", "bejis_2022",
            "--block-sizes-cells", "10", "20",
            "--dry-run",
        ])
        self.assertEqual(args.experiments, ["manavgat_2021", "bejis_2022"])
        self.assertEqual(args.block_sizes_cells, [10, 20])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)
        self.assertIs(args.func, cmd_step8_robustness)

    def test_step8_robustness_cli_dispatches_through_orchestrator(self):
        args = self.parser.parse_args([
            "step8-robustness",
            "--experiments", "manavgat_2021", "bejis_2022",
            "--block-sizes-cells", "10", "20",
            "--dry-run",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch,
            "run_step8_robustness_stage",
            return_value={"ran": False},
        ) as mocked:
            self.assertEqual(cmd_step8_robustness(args), 0)
        mocked.assert_called_once_with(
            experiments=["manavgat_2021", "bejis_2022"],
            block_sizes_cells=[10, 20],
            dry_run=True,
            force=False,
        )

    # =========================================================================
    # New user-facing commands: step10, large-block-robustness, concept-shift
    # =========================================================================
    def test_new_commands_present_in_help(self):
        # every new command must be registered as a subparser choice
        subparsers_action = next(
            a for a in self.parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
        )
        choices = set(subparsers_action.choices.keys())
        for command in ("step10", "large-block-robustness", "step8-big-block-robustness", "concept-shift"):
            self.assertIn(command, choices)
        # backward compatibility: existing commands remain
        for command in ("experiment", "transfer", "shift-audit", "transfer-explore",
                        "self-cal-transfer", "step8-robustness", "legacy"):
            self.assertIn(command, choices)

    def test_step10_subcommand_parses(self):
        args = self.parser.parse_args([
            "step10", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--report-only", "--dry-run",
        ])
        self.assertEqual(args.command, "step10")
        self.assertEqual(args.source, "manavgat_2021")
        self.assertEqual(args.target, "bejis_2022")
        self.assertTrue(args.reverse)
        self.assertTrue(args.report_only)
        self.assertTrue(args.dry_run)
        self.assertIs(args.func, cmd_step10)

    def test_step10_delegates_and_forwards_flags(self):
        args = self.parser.parse_args([
            "step10", "--source", "manavgat_2021", "--target", "bejis_2022",
            "--reverse", "--dry-run",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_step10_stage",
            return_value={"ran": False},
        ) as mocked:
            self.assertEqual(cmd_step10(args), 0)
        mocked.assert_called_once_with(
            source_id="manavgat_2021", target_id="bejis_2022", reverse=True,
            dry_run=True, force=False, report_only=False,
            bootstrap_replicates=1000, seed=42,
        )

    def test_large_block_robustness_subcommand_parses(self):
        args = self.parser.parse_args(["large-block-robustness", "--dry-run"])
        self.assertEqual(args.command, "large-block-robustness")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)
        self.assertFalse(args.run_large_block_fit)
        self.assertIs(args.func, cmd_large_block_robustness)

    def test_large_block_fit_is_explicit_and_forwarded(self):
        # default: fit is NOT requested
        args_default = self.parser.parse_args(["large-block-robustness", "--dry-run"])
        self.assertFalse(args_default.run_large_block_fit)
        # explicit fit flag is forwarded to the runner
        args_fit = self.parser.parse_args(["large-block-robustness", "--run-large-block-fit", "--force"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_large_block_robustness_stage",
            return_value={"ran": False},
        ) as mocked:
            self.assertEqual(cmd_large_block_robustness(args_fit), 0)
        mocked.assert_called_once_with(
            dry_run=False, force=True, run_large_block_fit=True,
        )

    def test_step8_big_block_robustness_subcommand_parses(self):
        args = self.parser.parse_args([
            "step8-big-block-robustness",
            "--experiment", "mugla_2021",
            "--block-sizes", "10", "20",
            "--dry-run",
        ])
        self.assertEqual(args.command, "step8-big-block-robustness")
        self.assertEqual(args.experiment, "mugla_2021")
        self.assertEqual(args.block_sizes, [10, 20])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)
        self.assertIs(args.func, cmd_step8_big_block_robustness)

    def test_step8_big_block_robustness_accepts_arbitrary_experiment(self):
        # no AOI is hard-coded into the CLI -- any experiment_id parses.
        args = self.parser.parse_args([
            "step8-big-block-robustness",
            "--experiment", "some_future_experiment",
            "--dry-run",
        ])
        self.assertEqual(args.experiment, "some_future_experiment")
        self.assertEqual(args.block_sizes, [10, 20])  # default

    def test_step8_big_block_robustness_cli_dispatches_through_orchestrator(self):
        args = self.parser.parse_args([
            "step8-big-block-robustness",
            "--experiment", "mugla_2021",
            "--block-sizes", "10", "20",
            "--dry-run",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch,
            "run_step8_big_block_robustness_stage",
            return_value={"ran": False},
        ) as mocked:
            self.assertEqual(cmd_step8_big_block_robustness(args), 0)
        mocked.assert_called_once_with(
            experiment="mugla_2021", block_sizes=[10, 20], dry_run=True, force=False,
            regenerate_reports_only=False, output_root=None,
        )

    def test_concept_shift_subcommand_parses(self):
        # --source/--target are REQUIRED: the pair is never implicit.
        args = self.parser.parse_args([
            "concept-shift",
            "--source", "mugla_2021",
            "--target", "manavgat_2021",
            "--dry-run",
        ])
        self.assertEqual(args.command, "concept-shift")
        self.assertEqual(args.source, "mugla_2021")
        self.assertEqual(args.target, "manavgat_2021")
        self.assertFalse(args.integration_only)
        self.assertTrue(args.dry_run)
        self.assertIs(args.func, cmd_concept_shift)

    def test_concept_shift_default_runs_numeric_analysis(self):
        args = self.parser.parse_args([
            "concept-shift",
            "--source", "mugla_2021",
            "--target", "manavgat_2021",
            "--dry-run",
        ])
        self.assertEqual(args.source, "mugla_2021")
        self.assertEqual(args.target, "manavgat_2021")
        with patch.object(
            sys.modules["scripts.main"].orch, "run_concept_shift_stage",
            return_value={"ran": False},
        ) as numeric, patch.object(
            sys.modules["scripts.main"].orch, "run_concept_shift_integration_stage",
            return_value={"ran": False},
        ) as integration:
            self.assertEqual(cmd_concept_shift(args), 0)
        numeric.assert_called_once_with(
            source_id="mugla_2021", target_id="manavgat_2021",
            dry_run=True, force=False, output_root=None,
        )
        integration.assert_not_called()

    def test_concept_shift_integration_only_is_report_only(self):
        args = self.parser.parse_args([
            "concept-shift",
            "--source", "mugla_2021",
            "--target", "manavgat_2021",
            "--integration-only", "--force",
        ])
        self.assertEqual(args.source, "mugla_2021")
        self.assertEqual(args.target, "manavgat_2021")
        self.assertTrue(args.integration_only)
        with patch.object(
            sys.modules["scripts.main"].orch, "run_concept_shift_integration_stage",
            return_value={"ran": True},
        ) as integration, patch.object(
            sys.modules["scripts.main"].orch, "run_concept_shift_stage",
            return_value={"ran": True},
        ) as numeric:
            self.assertEqual(cmd_concept_shift(args), 0)
        integration.assert_called_once_with(
            source_id="mugla_2021", target_id="manavgat_2021",
            dry_run=False, force=True,
        )
        numeric.assert_not_called()

    # --- marginal-aoa ---
    def test_marginal_aoa_subcommand_parses_arbitrary_experiment_ids(self):
        """No AOI is hard-coded in the CLI: any future experiment_id parses."""
        args = self.parser.parse_args([
            "marginal-aoa",
            "--experiments", "some_future_experiment", "another_future_experiment",
            "--dry-run",
        ])
        self.assertEqual(args.command, "marginal-aoa")
        self.assertEqual(
            args.experiments, ["some_future_experiment", "another_future_experiment"]
        )
        self.assertFalse(args.all_enabled)
        self.assertTrue(args.dry_run)
        self.assertFalse(args.force)
        self.assertIs(args.func, cmd_marginal_aoa)

    def test_marginal_aoa_all_enabled_selector_parses(self):
        args = self.parser.parse_args(["marginal-aoa", "--all-enabled", "--force"])
        self.assertTrue(args.all_enabled)
        self.assertIsNone(args.experiments)
        self.assertTrue(args.force)

    def test_marginal_aoa_selectors_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "marginal-aoa", "--experiments", "a_experiment", "--all-enabled",
            ])

    def test_marginal_aoa_requires_a_selector(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["marginal-aoa", "--dry-run"])

    def test_marginal_aoa_cli_dispatches_through_orchestrator(self):
        args = self.parser.parse_args([
            "marginal-aoa", "--experiments", "a_experiment", "b_experiment", "--dry-run",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_marginal_aoa_stage",
            return_value={"ran": False, "dry_run": True, "directed_pair_count": 2},
        ) as mocked:
            self.assertEqual(cmd_marginal_aoa(args), 0)
        mocked.assert_called_once_with(
            experiments=["a_experiment", "b_experiment"],
            all_enabled=False, dry_run=True, force=False,
        )

    # --- window-closure-sensitivity (RETIRED 2026-08-10) ---
    # The command published into outputs/diagnostics/window_closure_sensitivity/,
    # which was retired when all five AOIs were unified under
    # window_closure_region. It stays registered for provenance but must never
    # reach the orchestrator, the backend, or the retired output root again.
    def test_retired_window_closure_subcommand_is_still_registered(self):
        """Kept visible so the historical invocation stays self-documenting."""
        args = self.parser.parse_args([
            "window-closure-sensitivity",
            "--experiment", "some_future_experiment",
            "--shifts", "0", "7", "14",
            "--from-stage", "plan", "--to-stage", "compare",
            "--dry-run",
        ])
        self.assertEqual(args.command, "window-closure-sensitivity")
        self.assertIs(args.func, cmd_window_closure_sensitivity)

    def test_retired_window_closure_parses_without_an_experiment(self):
        """A retired command must answer with its message, not a usage error."""
        args = self.parser.parse_args(["window-closure-sensitivity"])
        self.assertIsNone(args.experiment)
        self.assertIs(args.func, cmd_window_closure_sensitivity)

    def test_retired_window_closure_refuses_and_never_dispatches(self):
        args = self.parser.parse_args([
            "window-closure-sensitivity",
            "--experiment", "some_future_experiment", "--dry-run",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_window_closure_sensitivity_stage",
        ) as mocked:
            self.assertEqual(cmd_window_closure_sensitivity(args), 2)
        mocked.assert_not_called()

    def test_retired_window_closure_refuses_every_stage_and_flag_combination(self):
        """No stage range, force or resume can get past the retirement guard."""
        invocations = [
            ["window-closure-sensitivity", "--experiment", "e", "--dry-run"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "plan", "--to-stage", "plan"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "prelabel-export", "--to-stage", "prelabel-export"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "predictor-export", "--to-stage", "predictor-export"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "model", "--to-stage", "model"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "compare", "--to-stage", "compare"],
            ["window-closure-sensitivity", "--experiment", "e",
             "--from-stage", "plan", "--to-stage", "plan", "--force", "--resume"],
            ["window-closure-sensitivity"],
        ]
        for argv in invocations:
            with self.subTest(argv=argv):
                args = self.parser.parse_args(argv)
                with patch.object(
                    sys.modules["scripts.main"].orch,
                    "run_window_closure_sensitivity_stage",
                ) as mocked:
                    self.assertEqual(cmd_window_closure_sensitivity(args), 2)
                mocked.assert_not_called()

    def test_retired_window_closure_message_names_the_replacement(self):
        args = self.parser.parse_args(["window-closure-sensitivity"])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(cmd_window_closure_sensitivity(args), 2)
        message = stderr.getvalue()
        self.assertIn("window-closure-sensitivity is retired", message)
        self.assertIn("window-closure-region", message)
        self.assertIn("manavgat_2021", message)
        self.assertIn("read-only reference", message)

    def test_retired_window_closure_rejects_an_unknown_stage(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "window-closure-sensitivity", "--experiment", "e",
                "--from-stage", "not_a_stage",
            ])

    def test_marginal_aoa_completion_subcommand_parses(self):
        args = self.parser.parse_args(["marginal-aoa-completion", "--dry-run"])
        self.assertEqual(args.command, "marginal-aoa-completion")
        self.assertEqual(args.from_stage, "plan")
        self.assertEqual(args.to_stage, "compare")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.resume)
        self.assertIsNone(args.experiments)
        self.assertIs(args.func, cmd_marginal_aoa_completion)

    def test_marginal_aoa_completion_supports_every_stage(self):
        from src.marginal_aoa_completion import STAGES

        for stage in STAGES:
            args = self.parser.parse_args([
                "marginal-aoa-completion", "--from-stage", stage,
                "--to-stage", stage, "--dry-run",
            ])
            self.assertEqual(args.from_stage, stage)
            self.assertEqual(args.to_stage, stage)

    def test_marginal_aoa_completion_rejects_an_unknown_stage(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "marginal-aoa-completion", "--from-stage", "not-a-stage",
            ])

    def test_marginal_aoa_completion_forwards_every_flag(self):
        args = self.parser.parse_args([
            "marginal-aoa-completion",
            "--experiments", "a", "b",
            "--from-stage", "weighted-predictor-space",
            "--to-stage", "compare",
            "--resume",
            "--output-root", "/tmp/out",
            "--experiments-root", "/tmp/exp",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_marginal_aoa_completion_stage",
            return_value={"ran": True, "stages_executed": ["compare"],
                          "analysis_id": "c" * 64},
        ) as mocked:
            self.assertEqual(cmd_marginal_aoa_completion(args), 0)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["experiments"], ["a", "b"])
        self.assertEqual(kwargs["from_stage"], "weighted-predictor-space")
        self.assertEqual(kwargs["to_stage"], "compare")
        self.assertTrue(kwargs["resume"])
        self.assertFalse(kwargs["dry_run"])
        self.assertEqual(kwargs["output_root"], "/tmp/out")
        self.assertEqual(kwargs["experiments_root"], "/tmp/exp")

    def test_marginal_aoa_completion_dry_run_is_forwarded(self):
        args = self.parser.parse_args(["marginal-aoa-completion", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_marginal_aoa_completion_stage",
            return_value={"ran": False, "dry_run": True, "stages_executed": [],
                          "analysis_id": "d" * 64, "files_written": []},
        ) as mocked:
            self.assertEqual(cmd_marginal_aoa_completion(args), 0)
        self.assertTrue(mocked.call_args.kwargs["dry_run"])

    def test_marginal_aoa_completion_reports_failure_as_exit_one(self):
        args = self.parser.parse_args(["marginal-aoa-completion", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_marginal_aoa_completion_stage",
            side_effect=SystemExit("contract violation"),
        ):
            self.assertEqual(cmd_marginal_aoa_completion(args), 1)

    def test_few_shot_recovery_subcommand_parses(self):
        args = self.parser.parse_args(["few-shot-recovery", "--dry-run"])
        self.assertEqual(args.command, "few-shot-recovery")
        self.assertEqual(args.from_stage, "plan")
        self.assertEqual(args.to_stage, "summarize")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.resume)
        self.assertFalse(args.force)
        self.assertIsNone(args.experiments)
        self.assertIs(args.func, cmd_few_shot_recovery)

    def test_few_shot_recovery_supports_every_stage(self):
        from src.few_shot_recovery import STAGES

        for stage in STAGES:
            args = self.parser.parse_args([
                "few-shot-recovery", "--from-stage", stage,
                "--to-stage", stage, "--dry-run",
            ])
            self.assertEqual(args.from_stage, stage)
            self.assertEqual(args.to_stage, stage)

    def test_few_shot_recovery_rejects_an_unknown_stage(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "few-shot-recovery", "--from-stage", "not-a-stage",
            ])

    def test_few_shot_recovery_forwards_every_flag(self):
        args = self.parser.parse_args([
            "few-shot-recovery",
            "--experiments", "a", "b",
            "--from-stage", "fit",
            "--to-stage", "summarize",
            "--resume", "--force",
            "--output-root", "/tmp/out",
            "--experiments-root", "/tmp/exp",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_few_shot_recovery_stage",
            return_value={"ran": True, "stages_executed": ["summarize"],
                          "analysis_id": "e" * 64},
        ) as mocked:
            self.assertEqual(cmd_few_shot_recovery(args), 0)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["experiments"], ["a", "b"])
        self.assertEqual(kwargs["from_stage"], "fit")
        self.assertEqual(kwargs["to_stage"], "summarize")
        self.assertTrue(kwargs["resume"])
        self.assertTrue(kwargs["force"])
        self.assertFalse(kwargs["dry_run"])
        self.assertEqual(kwargs["output_root"], "/tmp/out")
        self.assertEqual(kwargs["experiments_root"], "/tmp/exp")

    def test_few_shot_recovery_dry_run_is_forwarded(self):
        args = self.parser.parse_args(["few-shot-recovery", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_few_shot_recovery_stage",
            return_value={"ran": False, "dry_run": True, "stages_executed": [],
                          "analysis_id": "f" * 64, "files_written": []},
        ) as mocked:
            self.assertEqual(cmd_few_shot_recovery(args), 0)
        self.assertTrue(mocked.call_args.kwargs["dry_run"])

    def test_few_shot_recovery_reports_failure_as_exit_one(self):
        args = self.parser.parse_args(["few-shot-recovery", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_few_shot_recovery_stage",
            side_effect=SystemExit("canonical Step8A hash mismatch"),
        ):
            self.assertEqual(cmd_few_shot_recovery(args), 1)

    def test_mugla_subsampling_subcommand_parses(self):
        args = self.parser.parse_args(["mugla-subsampling", "--dry-run"])
        self.assertEqual(args.command, "mugla-subsampling")
        self.assertEqual(args.from_stage, "plan")
        self.assertEqual(args.to_stage, "summarize")
        self.assertTrue(args.dry_run)
        self.assertFalse(args.resume)
        self.assertFalse(args.force)
        self.assertIsNone(args.experiments)
        self.assertIs(args.func, cmd_mugla_subsampling)

    def test_mugla_subsampling_supports_every_stage(self):
        from src.mugla_subsampling import STAGES

        self.assertEqual(tuple(STAGES), ("plan", "fit", "summarize"))
        for stage in STAGES:
            args = self.parser.parse_args([
                "mugla-subsampling", "--from-stage", stage,
                "--to-stage", stage, "--dry-run",
            ])
            self.assertEqual(args.from_stage, stage)
            self.assertEqual(args.to_stage, stage)

    def test_mugla_subsampling_rejects_an_unknown_stage(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "mugla-subsampling", "--from-stage", "not-a-stage",
            ])

    def test_mugla_subsampling_forwards_every_flag(self):
        args = self.parser.parse_args([
            "mugla-subsampling",
            "--experiments", "a", "b",
            "--from-stage", "fit",
            "--to-stage", "summarize",
            "--resume", "--force",
            "--output-root", "/tmp/out",
            "--experiments-root", "/tmp/exp",
        ])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_mugla_subsampling_stage",
            return_value={"ran": True, "stages_executed": ["summarize"],
                          "analysis_id": "a" * 64},
        ) as mocked:
            self.assertEqual(cmd_mugla_subsampling(args), 0)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["experiments"], ["a", "b"])
        self.assertEqual(kwargs["from_stage"], "fit")
        self.assertEqual(kwargs["to_stage"], "summarize")
        self.assertTrue(kwargs["resume"])
        self.assertTrue(kwargs["force"])
        self.assertFalse(kwargs["dry_run"])
        self.assertEqual(kwargs["output_root"], "/tmp/out")
        self.assertEqual(kwargs["experiments_root"], "/tmp/exp")

    def test_mugla_subsampling_dry_run_is_forwarded(self):
        args = self.parser.parse_args(["mugla-subsampling", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_mugla_subsampling_stage",
            return_value={"ran": False, "dry_run": True, "stages_executed": [],
                          "analysis_id": "b" * 64, "files_written": []},
        ) as mocked:
            self.assertEqual(cmd_mugla_subsampling(args), 0)
        self.assertTrue(mocked.call_args.kwargs["dry_run"])

    def test_mugla_subsampling_reports_failure_as_exit_one(self):
        args = self.parser.parse_args(["mugla-subsampling", "--dry-run"])
        with patch.object(
            sys.modules["scripts.main"].orch, "run_mugla_subsampling_stage",
            side_effect=SystemExit("canonical Step8A hash mismatch"),
        ):
            self.assertEqual(cmd_mugla_subsampling(args), 1)

    def test_legacy_subcommand_defaults_to_kozan(self):
        args = self.parser.parse_args(["legacy", "--dry-run"])
        self.assertEqual(args.command, "legacy")
        self.assertEqual(args.experiment, "kozan_2023")
        self.assertIs(args.func, cmd_legacy)

    def test_legacy_subcommand_accepts_explicit_experiment(self):
        args = self.parser.parse_args(["legacy", "--experiment", "manavgat_2021", "--dry-run"])
        self.assertEqual(args.experiment, "manavgat_2021")


class TestLegacyGuard(unittest.TestCase):
    """cmd_legacy, kozan_2023 disindaki deneyleri CALISTIRMADAN reddetmelidir."""

    def test_non_kozan_experiment_rejected_without_running_anything(self):
        parser = build_parser()
        args = parser.parse_args(["legacy", "--experiment", "manavgat_2021", "--dry-run"])
        exit_code = cmd_legacy(args)
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
