"""
tests/test_main_cli.py

scripts/main.py'nin argparse yapısı için odaklı unittest testleri. Hiçbir
alt-komutu GERÇEKTEN ÇALIŞTIRMAZ (yalnızca parser/dispatch davranışını test
eder); GEE/ağ erişimi gerektirmez.

Çalıştırma:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.main import build_parser, cmd_experiment, cmd_legacy, cmd_shift_audit, cmd_transfer


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
        self.assertIs(args.func, cmd_experiment)

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