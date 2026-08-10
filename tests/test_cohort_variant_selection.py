"""
tests/test_cohort_variant_selection.py

Targeted tests for the frozen canonical/legacy variant decision and every
selection path that must honour it.

Frozen decision under test:
    evia_2021_extended  = canonical
    evia_2021           = legacy_superseded, superseded_by=evia_2021_extended
    kozan_2023          = canonical, role=negative_control (NOT legacy)
    legacy outputs are retained -- never deleted, never overwritten

Covers:
    A  every registry record declares a valid variant_status
    B  the Evia variant pair and the kozan negative control specifically
    C  the generic helpers (get_variant_status /
       assert_not_superseded_experiments / list_canonical_enabled_experiments)
       including fail-closed behaviour on unknown/missing/contradictory status
    D  --all-enabled discovery excludes legacy Evia (superseded_by reason) and
       the negative control (negative_control reason), while an experiment
       that simply has not reached Step8A keeps its missing-input reason
    E  explicit --experiments selection of legacy Evia fails closed
    F  multi-AOI transfer synthesis and Step9G multi-AOI univariate comparison
       both reject legacy Evia and both accept the canonical variants
    G  the prevalence audit JSON: four distinct definitions, correct
       arithmetic, and the two population scopes recorded separately
    H  nothing in this test run writes into a legacy output directory

Read-only: no Earth Engine call, no export, no gate, no model, no dry-run,
no bootstrap. No file under outputs/ is written.

Run:
    python -m pytest tests/test_cohort_variant_selection.py -q
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.regions as regions
import src.burned_pattern_audit as bpa

CANONICAL_EVIA = "evia_2021_extended"
LEGACY_EVIA = "evia_2021"
NEGATIVE_CONTROL = "kozan_2023"
MONTIFERRU = "montiferru_2021"

PREVALENCE_JSON = _PROJECT_ROOT / "docs" / "evia_2021_prevalence_audit.json"

# The four prevalence definitions are DELIBERATELY distinct; quoting one
# without its definition is the failure mode this audit exists to catch.
EXPECTED_PREVALENCE = {
    LEGACY_EVIA: {
        "raw_aoi_grid": (2789, 7744),
        "final_all_valid_model_population": (2774, 7728),
        "raw_burnable_tree_shrub_grass_mask": (2674, 3957),
        "final_downstream_primary_burnable_tree_shrub_grass_population": (2663, 3946),
    },
    CANONICAL_EVIA: {
        "raw_aoi_grid": (2803, 22925),
        "final_all_valid_model_population": (2788, 22906),
        "raw_burnable_tree_shrub_grass_mask": (2675, 9309),
        "final_downstream_primary_burnable_tree_shrub_grass_population": (2664, 9298),
    },
}

FROZEN_VERIFICATION_HASHES = {
    LEGACY_EVIA: {
        (
            "outputs/experiments/evia_2021/"
            "step8a/step8a_dataset_stats.json"
        ): (
            "a228a887c3b1e6c455bc785c1311ded2"
            "e8a7bdc6a8fafa615a7d33c94fbb9ab8"
        ),
        (
            "outputs/experiments/evia_2021/"
            "step8a/step8a_500m_modeling_dataset.parquet"
        ): (
            "dd543544b6cd1b38c943ba54805c4832"
            "342294276c868ae3a2d6f61d8b329456"
        ),
        (
            "outputs/experiments/evia_2021/"
            "step8b/step8b_model_comparison_metrics.json"
        ): (
            "289c842875571bc370a23933efdb9cf6"
            "720cb0f9233762539a5479a95b9856bc"
        ),
        (
            "outputs/experiments/evia_2021/"
            "step8b/step8b_delta_auc_by_population.csv"
        ): (
            "b1a1a49d8a9f1b1ce8cc0ef36788ba3"
            "65cdcfe9e59943a5fceb80c72ca7329bd"
        ),
    },
    CANONICAL_EVIA: {
        (
            "outputs/experiments/evia_2021_extended/"
            "step8a/step8a_dataset_stats.json"
        ): (
            "8ba510dacff226db389a406f736372fed"
            "f511e83edc95a8f4cec332a72aef74f"
        ),
        (
            "outputs/experiments/evia_2021_extended/"
            "step8a/step8a_500m_modeling_dataset.parquet"
        ): (
            "bdce859cf482f575d0f273174b157f47"
            "efd61779953fdd23d9486c5face5e553"
        ),
        (
            "outputs/experiments/evia_2021_extended/"
            "step8b/step8b_model_comparison_metrics.json"
        ): (
            "7a5b2b97a4bfe32202a982b5f411cbd0"
            "1bbcf8075970e8790ccebf27498eeef1"
        ),
        (
            "outputs/experiments/evia_2021_extended/"
            "step8b/step8b_delta_auc_by_population.csv"
        ): (
            "ba49312afda6cd5a6898a71699380014"
            "8ce40244fed790958aea7571b85a4ce9"
        ),
    },
}

# =============================================================================
# A. Every registry record declares a valid variant_status
# =============================================================================
class TestRegistryVariantContract(unittest.TestCase):
    def test_every_experiment_has_a_valid_variant_status(self):
        for experiment_id in regions.EXPERIMENTS:
            with self.subTest(experiment_id=experiment_id):
                status = regions.get_variant_status(experiment_id)
                self.assertIn(status, regions.ALLOWED_VARIANT_STATUSES)

    def test_only_two_statuses_are_allowed(self):
        self.assertEqual(
            set(regions.ALLOWED_VARIANT_STATUSES),
            {"canonical", "legacy_superseded"},
        )

    def test_superseded_records_point_at_a_registered_canonical_successor(self):
        for experiment_id, record in regions.EXPERIMENTS.items():
            if record.get("variant_status") != regions.VARIANT_STATUS_LEGACY_SUPERSEDED:
                continue
            with self.subTest(experiment_id=experiment_id):
                successor = record["superseded_by"]
                self.assertIn(successor, regions.EXPERIMENTS)
                self.assertNotEqual(successor, experiment_id)
                self.assertEqual(regions.get_variant_status(successor),
                                 regions.VARIANT_STATUS_CANONICAL)

    def test_canonical_records_never_carry_superseded_by(self):
        for experiment_id, record in regions.EXPERIMENTS.items():
            if record.get("variant_status") == regions.VARIANT_STATUS_CANONICAL:
                with self.subTest(experiment_id=experiment_id):
                    self.assertNotIn("superseded_by", record)


# =============================================================================
# B. The specific frozen decisions
# =============================================================================
class TestFrozenVariantDecision(unittest.TestCase):
    def test_extended_evia_is_canonical(self):
        self.assertEqual(regions.get_variant_status(CANONICAL_EVIA),
                         regions.VARIANT_STATUS_CANONICAL)

    def test_legacy_evia_is_superseded_by_the_extended_variant(self):
        self.assertEqual(regions.get_variant_status(LEGACY_EVIA),
                         regions.VARIANT_STATUS_LEGACY_SUPERSEDED)
        self.assertEqual(regions.EXPERIMENTS[LEGACY_EVIA]["superseded_by"], CANONICAL_EVIA)

    def test_kozan_is_canonical_negative_control_not_legacy(self):
        self.assertEqual(regions.get_variant_status(NEGATIVE_CONTROL),
                         regions.VARIANT_STATUS_CANONICAL)
        self.assertEqual(regions.EXPERIMENTS[NEGATIVE_CONTROL]["role"], "negative_control")
        self.assertNotIn("superseded_by", regions.EXPERIMENTS[NEGATIVE_CONTROL])

    def test_enabled_flags_are_unchanged_by_the_variant_decision(self):
        # The patch must not disable anything -- variant_status and enabled
        # are independent axes.
        for experiment_id in (CANONICAL_EVIA, LEGACY_EVIA, NEGATIVE_CONTROL, MONTIFERRU):
            with self.subTest(experiment_id=experiment_id):
                self.assertTrue(regions.EXPERIMENTS[experiment_id]["enabled"])


# =============================================================================
# C. Generic helpers, including fail-closed paths
# =============================================================================
class TestVariantHelpers(unittest.TestCase):
    def test_list_canonical_enabled_experiments_drops_legacy_keeps_control(self):
        canonical = regions.list_canonical_enabled_experiments()
        self.assertNotIn(LEGACY_EVIA, canonical)
        self.assertIn(CANONICAL_EVIA, canonical)
        # kozan is canonical: role filtering, not variant filtering, is what
        # removes it from natural-vegetation cohorts.
        self.assertIn(NEGATIVE_CONTROL, canonical)

    def test_list_canonical_enabled_experiments_role_filter(self):
        controls = regions.list_canonical_enabled_experiments(role="negative_control")
        self.assertEqual(set(controls), {NEGATIVE_CONTROL})
        wildfires = regions.list_canonical_enabled_experiments(
            role="mediterranean_transfer_wildfire")
        self.assertIn(CANONICAL_EVIA, wildfires)
        self.assertNotIn(LEGACY_EVIA, wildfires)
        self.assertNotIn(NEGATIVE_CONTROL, wildfires)

    def test_assert_not_superseded_accepts_canonical_ids_unchanged(self):
        ids = [CANONICAL_EVIA, MONTIFERRU, NEGATIVE_CONTROL]
        self.assertEqual(regions.assert_not_superseded_experiments(ids, context="t"),
                         tuple(ids))

    def test_assert_not_superseded_rejects_legacy_and_names_successor(self):
        with self.assertRaises(regions.VariantStatusError) as ctx:
            regions.assert_not_superseded_experiments(
                [CANONICAL_EVIA, LEGACY_EVIA], context="unit-test cohort")
        message = str(ctx.exception)
        self.assertIn(LEGACY_EVIA, message)
        self.assertIn(CANONICAL_EVIA, message)
        self.assertIn("unit-test cohort", message)

    def test_variant_error_is_a_value_error_subclass(self):
        # Existing callers already catching ValueError keep working.
        self.assertTrue(issubclass(regions.VariantStatusError, ValueError))

    def test_unknown_experiment_id_still_raises(self):
        with self.assertRaises(ValueError):
            regions.get_variant_status("no_such_experiment_id")

    def test_missing_status_fails_closed(self):
        with self.assertRaises(regions.VariantStatusError):
            regions.validate_variant_record({"enabled": True}, "synthetic")

    def test_unknown_status_fails_closed(self):
        with self.assertRaises(regions.VariantStatusError):
            regions.validate_variant_record(
                {"variant_status": "provisional"}, "synthetic")

    def test_superseded_without_pointer_fails_closed(self):
        with self.assertRaises(regions.VariantStatusError):
            regions.validate_variant_record(
                {"variant_status": "legacy_superseded"}, "synthetic")

    def test_superseded_pointing_at_itself_fails_closed(self):
        with self.assertRaises(regions.VariantStatusError):
            regions.validate_variant_record(
                {"variant_status": "legacy_superseded", "superseded_by": "synthetic"},
                "synthetic")

    def test_canonical_carrying_a_successor_fails_closed(self):
        with self.assertRaises(regions.VariantStatusError):
            regions.validate_variant_record(
                {"variant_status": "canonical", "superseded_by": CANONICAL_EVIA},
                "synthetic")

    def test_superseded_target_must_be_canonical(self):
        source_id = "synthetic_legacy_source"
        target_id = "synthetic_legacy_target"

        synthetic_records = {
            source_id: {
                "variant_status": "legacy_superseded",
                "superseded_by": target_id,
            },
            target_id: {
                "variant_status": "legacy_superseded",
                "superseded_by": CANONICAL_EVIA,
            },
        }

        with patch.dict(
            regions.EXPERIMENTS,
            synthetic_records,
            clear=False,
        ):
            with self.assertRaisesRegex(
                regions.VariantStatusError,
                "must reference a canonical experiment",
            ):
                regions.validate_variant_record(
                    regions.EXPERIMENTS[source_id],
                    source_id,
                )


# =============================================================================
# D. --all-enabled discovery resolver
# =============================================================================
class TestAllEnabledResolution(unittest.TestCase):
    """Uses the REAL registry with only `canonical_step8a_path` faked, so the
    exclusion reasons under test are the ones production discovery emits."""

    def _resolve(self, present_ids):
        def fake_path(experiment_id):
            root = Path("/nonexistent") / experiment_id / "step8a"
            path = root / "step8a_500m_modeling_dataset.parquet"
            return _ExistingPath(path) if experiment_id in present_ids else path

        with patch.object(bpa, "canonical_step8a_path", side_effect=fake_path):
            return bpa.resolve_experiments(all_enabled=True)

    def test_legacy_evia_is_excluded_with_the_superseded_reason(self):
        resolution = self._resolve({CANONICAL_EVIA, MONTIFERRU})
        self.assertNotIn(LEGACY_EVIA, resolution.resolved_ids)
        self.assertEqual(resolution.excluded[LEGACY_EVIA],
                         f"superseded_by_{CANONICAL_EVIA}")

    def test_negative_control_is_excluded_with_the_role_reason(self):
        resolution = self._resolve({CANONICAL_EVIA, NEGATIVE_CONTROL})
        self.assertNotIn(NEGATIVE_CONTROL, resolution.resolved_ids)
        self.assertEqual(resolution.excluded[NEGATIVE_CONTROL], "negative_control")

    def test_canonical_extended_evia_is_resolved(self):
        resolution = self._resolve({CANONICAL_EVIA})
        self.assertIn(CANONICAL_EVIA, resolution.resolved_ids)
        self.assertNotIn(CANONICAL_EVIA, resolution.excluded)

    def test_missing_step8a_keeps_the_missing_input_reason_not_legacy(self):
        # Montiferru has not reached Step8A: it must be reported as a missing
        # input, NEVER mislabelled as a legacy/superseded variant.
        resolution = self._resolve({CANONICAL_EVIA})
        reason = resolution.excluded[MONTIFERRU]
        self.assertTrue(reason.startswith("missing_canonical_step8a_dataset:"), reason)
        self.assertNotIn("superseded", reason)
        self.assertNotIn("negative_control", reason)

    def test_legacy_evia_reported_as_legacy_even_when_step8a_exists(self):
        # evia_2021 DOES have frozen Step8A output on disk; the variant reason
        # must win over the input-presence check.
        resolution = self._resolve({CANONICAL_EVIA, LEGACY_EVIA})
        self.assertEqual(resolution.excluded[LEGACY_EVIA],
                         f"superseded_by_{CANONICAL_EVIA}")

    def test_every_enabled_experiment_is_either_resolved_or_explained(self):
        resolution = self._resolve({CANONICAL_EVIA})
        enabled = set(regions.list_experiments(include_disabled=False))
        accounted = set(resolution.resolved_ids) | set(resolution.excluded)
        self.assertEqual(enabled, accounted)

    def test_domain_classifier_audit_shares_this_resolver(self):
        # No parallel variant filter may exist downstream.
        import src.domain_classifier_audit as dca

        self.assertIs(dca.resolve_experiments_generic, bpa.resolve_experiments)


class _ExistingPath(type(Path())):
    """Path subclass whose `is_file()` is always True, so resolver tests can
    simulate a present Step8A dataset without creating any file."""

    def is_file(self) -> bool:  # noqa: D102
        return True


# =============================================================================
# E. Explicit --experiments selection
# =============================================================================
class TestExplicitSelection(unittest.TestCase):
    def test_explicit_legacy_evia_fails_closed(self):
        with self.assertRaises(bpa.BurnedPatternAuditError) as ctx:
            bpa.resolve_experiments(experiments=[CANONICAL_EVIA, LEGACY_EVIA])
        message = str(ctx.exception)
        self.assertIn(LEGACY_EVIA, message)
        self.assertIn(f"superseded_by='{CANONICAL_EVIA}'", message)

    def test_explicit_legacy_evia_is_not_silently_dropped(self):
        # The failure must be an exception, never a quietly shortened cohort.
        try:
            resolution = bpa.resolve_experiments(experiments=[LEGACY_EVIA])
        except bpa.BurnedPatternAuditError:
            return
        self.fail(f"legacy experiment was silently accepted: {resolution.resolved_ids}")

    def test_explicit_negative_control_behaviour_is_unchanged(self):
        # kozan_2023 is canonical: explicit selection must still pass the
        # variant guard and fail (if at all) only on missing input.
        regions.assert_not_superseded_experiments([NEGATIVE_CONTROL], context="t")


# =============================================================================
# F. New cross-region selection entry points
# =============================================================================
class TestCrossRegionEntryPoints(unittest.TestCase):
    def test_transfer_synthesis_rejects_legacy_evia(self):
        import core.pipeline_orchestrator as orch

        with self.assertRaises(SystemExit) as ctx:
            orch.run_multi_aoi_transfer_synthesis_stage(
                aois=["manavgat_2021", LEGACY_EVIA], dry_run=True, force=False)
        self.assertIn(LEGACY_EVIA, str(ctx.exception))
        self.assertIn(CANONICAL_EVIA, str(ctx.exception))

    def test_transfer_synthesis_guard_runs_before_any_input_resolution(self):
        # The guard must reject on identity alone -- never depend on whether
        # frozen Step8/Step9/Step10 inputs happen to exist.
        import core.pipeline_orchestrator as orch

        with patch("src.multi_aoi_transfer_synthesis.build.build_synthesis",
                   side_effect=AssertionError("build must not be reached")):
            with self.assertRaises(SystemExit):
                orch.run_multi_aoi_transfer_synthesis_stage(
                    aois=[LEGACY_EVIA, CANONICAL_EVIA], dry_run=True, force=False)

    def test_step9g_comparison_rejects_legacy_evia(self):
        from src.step9g_multi_aoi_comparison.build import ComparisonError, resolve_experiments

        with self.assertRaises(ComparisonError) as ctx:
            resolve_experiments(["manavgat_2021", LEGACY_EVIA])
        self.assertIn(LEGACY_EVIA, str(ctx.exception))
        self.assertIn(CANONICAL_EVIA, str(ctx.exception))

    def test_step9g_comparison_accepts_canonical_extended_and_montiferru(self):
        from src.step9g_multi_aoi_comparison.build import resolve_experiments

        selected = resolve_experiments(["manavgat_2021", CANONICAL_EVIA, MONTIFERRU])
        self.assertEqual(selected, ("manavgat_2021", CANONICAL_EVIA, MONTIFERRU))

    def test_no_canonical_cohort_is_hardcoded_in_the_guarded_entry_points(self):
        # The guard must be generic: no fixed five-AOI list in either module,
        # and it must come from core.regions rather than a local re-implementation.
        generic_guards = ("assert_not_superseded_experiments", "validate_variant_record")
        for relative in ("core/pipeline_orchestrator.py",
                         "src/step9g_multi_aoi_comparison/build.py"):
            source = (_PROJECT_ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(module=relative):
                self.assertTrue(any(guard in source for guard in generic_guards),
                                f"{relative} does not use a core.regions variant guard")
                # No cohort AOI may be named by the guard. kozan_2023 is
                # deliberately NOT checked here: pipeline_orchestrator has a
                # pre-existing LEGACY_EXPERIMENT_ID for the legacy Kozan
                # pipeline, which is unrelated to variant selection.
                cohort_ids = [
                    e for e in regions.EXPERIMENTS
                    if regions.EXPERIMENTS[e].get("role") != "negative_control"
                ]
                for experiment_id in cohort_ids:
                    self.assertNotIn(f'"{experiment_id}"', source,
                                     f"{relative} hard-codes experiment_id '{experiment_id}'")

    def test_window_closure_contract_is_untouched_and_still_excludes_legacy(self):
        from src.multi_region_window_closure import contract

        source = (_PROJECT_ROOT / "src" / "multi_region_window_closure"
                  / "contract.py").read_text(encoding="utf-8")
        self.assertNotIn("assert_not_superseded_experiments", source)
        self.assertNotIn(LEGACY_EVIA, getattr(contract, "AOIS", ()) or ())


# =============================================================================
# G. Prevalence audit record
# =============================================================================
class TestPrevalenceAuditJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(PREVALENCE_JSON.read_text(encoding="utf-8"))

    def test_variant_decision_is_recorded(self):
        self.assertEqual(self.doc["canonical_variant"], CANONICAL_EVIA)
        self.assertEqual(self.doc["legacy_variant"], LEGACY_EVIA)
        self.assertEqual(self.doc["legacy_status"], "legacy_superseded")
        self.assertEqual(self.doc["superseded_by"], CANONICAL_EVIA)
        self.assertEqual(self.doc["contract_difference"], "AOI geometry only")

    def test_json_matches_the_live_registry(self):
        self.assertEqual(self.doc["legacy_status"], regions.get_variant_status(LEGACY_EVIA))
        self.assertEqual(self.doc["superseded_by"], regions.get_superseded_by(LEGACY_EVIA))

    def test_both_geometry_hash_definitions_are_recorded_correctly(self):
        from scripts.run_label_gate_only import (
            _static_region_aoi_provenance,
        )

        canonical_hashes = []
        bbox_tuple_hashes = []

        for experiment_id in (LEGACY_EVIA, CANONICAL_EVIA):
            variant = self.doc["variants"][experiment_id]
            record = regions.get_experiment(experiment_id)

            expected = _static_region_aoi_provenance(
                record["region_key"]
            )
            self.assertIsNotNone(expected)

            self.assertEqual(
                variant["canonical_geometry_hash"],
                expected["geometry_hash"],
            )
            self.assertEqual(
                len(variant["bbox_tuple_sha256"]),
                64,
            )

            canonical_hashes.append(
                variant["canonical_geometry_hash"]
            )
            bbox_tuple_hashes.append(
                variant["bbox_tuple_sha256"]
            )

        self.assertEqual(len(set(canonical_hashes)), 2)
        self.assertEqual(len(set(bbox_tuple_hashes)), 2)

    def test_all_four_prevalence_definitions_are_present_per_variant(self):
        for variant_id, expected in EXPECTED_PREVALENCE.items():
            with self.subTest(variant=variant_id):
                self.assertEqual(set(self.doc["variants"][variant_id]["prevalence"]),
                                 set(expected))

    def test_every_prevalence_row_has_correct_arithmetic(self):
        for variant_id, expected in EXPECTED_PREVALENCE.items():
            rows = self.doc["variants"][variant_id]["prevalence"]
            for name, (numerator, denominator) in expected.items():
                with self.subTest(variant=variant_id, definition=name):
                    row = rows[name]
                    self.assertEqual(row["numerator"], numerator)
                    self.assertEqual(row["denominator"], denominator)
                    self.assertAlmostEqual(row["computed_fraction"],
                                           numerator / denominator, places=10)

    def test_every_prevalence_row_carries_definition_source_and_hash(self):
        for variant_id in EXPECTED_PREVALENCE:
            variant = self.doc["variants"][variant_id]
            artifact_hashes = {a["sha256"] for a in variant["source_artifacts"].values()}
            for name, row in variant["prevalence"].items():
                with self.subTest(variant=variant_id, definition=name):
                    self.assertTrue(row["definition"].strip())
                    self.assertTrue(row["filter"].strip())
                    self.assertTrue(row["source_path"].startswith("outputs/experiments/"))
                    self.assertEqual(len(row["source_sha256"]), 64)
                    self.assertIn(row["source_sha256"], artifact_hashes)
                    self.assertEqual(len(row["source_json_fields"]), 2)

    def test_recorded_verification_hashes_match_the_files_on_disk(self):
        import hashlib

        for variant_id in EXPECTED_PREVALENCE:
            artifacts = self.doc["variants"][variant_id][
                "verification_artifacts"
            ]

            self.assertEqual(len(artifacts), 4)

            for artifact in artifacts.values():
                path = _PROJECT_ROOT / artifact["path"]
                with self.subTest(path=artifact["path"]):
                    self.assertTrue(path.is_file(), path)
                    actual = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    self.assertEqual(
                        actual,
                        artifact["sha256"],
                    )

    def test_verification_hashes_equal_the_frozen_audit_values(self):
        for variant_id, expected in (
            FROZEN_VERIFICATION_HASHES.items()
        ):
            artifacts = self.doc["variants"][variant_id][
                "verification_artifacts"
            ]

            actual = {
                artifact["path"]: artifact["sha256"]
                for artifact in artifacts.values()
            }

            self.assertEqual(actual, expected)

    def test_the_four_definitions_are_genuinely_distinct(self):
        # If any two coincided the audit would not prove anything.
        for variant_id in EXPECTED_PREVALENCE:
            rows = self.doc["variants"][variant_id]["prevalence"]
            with self.subTest(variant=variant_id):
                pairs = {(r["numerator"], r["denominator"]) for r in rows.values()}
                self.assertEqual(len(pairs), 4)

    def test_the_two_population_scopes_are_recorded_separately(self):
        note = self.doc["population_scope_note"]
        self.assertEqual(note["step8b_module_primary"], "all_valid")
        self.assertEqual(note["downstream_primary"], "burnable_tree_shrub_grass")
        self.assertIn("reselect", note["interpretation"])
        self.assertNotEqual(note["step8b_module_primary"], note["downstream_primary"])

    def test_scope_note_matches_the_live_constants(self):
        from core.config import STEP8B_PRIMARY_POPULATION
        from src.step9a_audit_cross_region_inputs import PRIMARY_POPULATIONS

        note = self.doc["population_scope_note"]
        self.assertEqual(note["step8b_module_primary"], STEP8B_PRIMARY_POPULATION)
        self.assertEqual(note["downstream_primary"], PRIMARY_POPULATIONS[0])

    def test_shared_contract_covers_predictors_labels_grid_and_model(self):
        shared = self.doc["shared_contract"]
        for key in ("predictor_start_date", "predictor_end_date", "label_start_date",
                    "label_end_date", "baseline_years", "label_source", "grid",
                    "baseline_features", "thermal_features_added", "model",
                    "exclusion_rules"):
            self.assertIn(key, shared)

    def test_shared_contract_matches_both_registry_records(self):
        shared = self.doc["shared_contract"]
        for experiment_id in (LEGACY_EVIA, CANONICAL_EVIA):
            record = regions.get_experiment(experiment_id)
            for key in ("predictor_start_date", "predictor_end_date",
                        "label_start_date", "label_end_date", "baseline_years",
                        "role", "country", "exclude_pre_label_burns",
                        "pre_label_burn_window"):
                with self.subTest(experiment_id=experiment_id, key=key):
                    self.assertEqual(shared[key], record[key])

    def test_legacy_retention_policy_is_recorded(self):
        retention = self.doc["legacy_output_retention"]
        self.assertFalse(retention["deleted"])
        self.assertFalse(retention["overwritten"])
        self.assertEqual(retention["legacy_output_root"], f"outputs/experiments/{LEGACY_EVIA}")


# =============================================================================
# H. Legacy outputs are never written
# =============================================================================
class TestLegacyOutputsUntouched(unittest.TestCase):
    LEGACY_ROOT = _PROJECT_ROOT / "outputs" / "experiments" / LEGACY_EVIA

    def _snapshot(self):
        if not self.LEGACY_ROOT.exists():
            return {}
        return {p: p.stat().st_mtime_ns for p in self.LEGACY_ROOT.rglob("*") if p.is_file()}

    def test_variant_selection_paths_write_nothing_into_the_legacy_namespace(self):
        before = self._snapshot()

        regions.list_canonical_enabled_experiments()
        regions.get_variant_status(LEGACY_EVIA)
        with self.assertRaises(regions.VariantStatusError):
            regions.assert_not_superseded_experiments([LEGACY_EVIA], context="t")
        with self.assertRaises(bpa.BurnedPatternAuditError):
            bpa.resolve_experiments(experiments=[LEGACY_EVIA])

        after = self._snapshot()
        self.assertEqual(before, after, "legacy Evia outputs were modified")

    def test_prevalence_audit_lives_outside_every_output_namespace(self):
        self.assertEqual(PREVALENCE_JSON.parent, _PROJECT_ROOT / "docs")
        self.assertNotIn("outputs", PREVALENCE_JSON.parts)


if __name__ == "__main__":
    unittest.main()
