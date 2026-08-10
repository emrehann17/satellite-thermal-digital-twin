"""Targeted tests for the three Muğla pre-run blockers.

1. the registry-driven pre-label exclusion binding (documents + accounting),
2. the cohort removal accounting published in `cohort_inventory.csv`,
3. the fixed MODIS month-filter clipping published in `regional_summary.csv`.

Nothing here reaches Earth Engine, the production downstream chain, a model
fit or a bootstrap: every fixture is a small in-memory/tmp_path artefact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.step6b_burned_landcover_gate as step6b  # noqa: E402
import src.step8a_prepare_500m_modeling_dataset as step8a  # noqa: E402
import src.window_closure_sensitivity as wcs  # noqa: E402
from src.multi_region_window_closure.contract import (  # noqa: E402
    ACTUAL_AOIS, SHIFTED_VARIANTS, VARIANTS, MultiRegionWindowClosureError,
)
from src.multi_region_window_closure.production import (  # noqa: E402
    assert_modis_clipping_matches_windows, derive_cohort_accounting,
    modis_clipping_summary_fields, resolve_modis_clipping,
)


# =============================================================================
# Fixtures
# =============================================================================
def _write_manifest(directory: Path, experiment_id: str, cell_ids: list[str]) -> None:
    """The Step6B gate exclusion manifest, under its production file names."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": cell_ids}).to_parquet(
        directory / step6b.PRE_LABEL_EXCLUSION_MANIFEST_PARQUET, index=False,
    )
    (directory / step6b.PRE_LABEL_EXCLUSION_MANIFEST_METADATA).write_text(
        json.dumps({"experiment_id": experiment_id, "excluded_cell_count": len(cell_ids)}),
        encoding="utf-8",
    )


def _context(gate_dir: Path, *, active: bool) -> dict:
    return {
        wcs.PRELABEL_EXCLUSION_POLICY_FIELD: active,
        "gate_labels_dir": gate_dir,
    }


def _binding(tmp_path: Path, *, active: bool, cells: list[str] | None = None,
             experiment_id: str = "synthetic_experiment") -> dict:
    """A resolved binding whose canonical namespace is `tmp_path`."""
    gate_dir = (
        wcs.canonical_experiment_root(ACTUAL_AOIS[0], tmp_path) / "validation" / "labels"
    )
    if cells is not None:
        _write_manifest(gate_dir, experiment_id, cells)
    return wcs.prelabel_exclusion_binding(
        ACTUAL_AOIS[0], _context(gate_dir, active=active), tmp_path,
    )


def _frame(excluded: list[str], total: int = 6) -> pd.DataFrame:
    cells = [f"r{i}_c{i}" for i in range(total)]
    excluded_mask = np.array([cell in set(excluded) for cell in cells])
    return pd.DataFrame({
        "cell_id": cells,
        "pre_label_burn_excluded": excluded_mask,
        "analysis_eligible": ~excluded_mask,
        "burned": (np.arange(total) % 2).astype("int64"),
    })


def _cohort_metadata(final: int = 100, initial: int = 160) -> dict:
    """A metadata document whose per-variant arithmetic closes exactly."""
    per_variant = {
        "canonical": {"removed_missing_required_feature_union": 20, "removed_variant_only_keys": 20},
        "close_7d_earlier": {"removed_missing_required_feature_union": 30, "removed_variant_only_keys": 10},
        "close_14d_earlier": {"removed_missing_required_feature_union": 25, "removed_variant_only_keys": 15},
    }
    document: dict = {
        "final_common_cohort_rows": final,
        "initial_rows_by_variant": {variant: initial for variant in VARIANTS},
        "removed_label_mismatch": 0,
        "removed_static_invariance_failure": 0,
        "removed_not_valid_for_modeling": {variant: 5 for variant in VARIANTS},
        "removed_outside_primary_population": {variant: 10 for variant in VARIANTS},
        "removed_prelabel_censor": {variant: 5 for variant in VARIANTS},
    }
    for field in ("removed_missing_required_feature_union", "removed_variant_only_keys"):
        document[field] = {variant: per_variant[variant][field] for variant in VARIANTS}
    return document


def _export_metadata(clipped: int, effective: int) -> dict:
    return {
        "artifact_inventory": [
            {
                "artifact_id": role, "role": role,
                "date_semantics": {
                    "duration_days": clipped + effective,
                    wcs.MODIS_CLIPPING_TRANSPARENCY_KEY: {
                        "calendar_month_filter": "6-9",
                        "clipped_day_count": clipped,
                        "effective_included_day_count": effective,
                    },
                },
            }
            for role in wcs.MODIS_ROLE_FILENAMES
        ],
    }


def _seed_export_provenance(production_root: Path, aoi: str, clipping: dict[str, int],
                            *, downstream: dict[str, int] | None = None) -> None:
    for variant, clipped in clipping.items():
        variant_root = production_root / aoi / "variants" / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / "predictor_export_metadata.json").write_text(
            json.dumps(_export_metadata(clipped, 60 - clipped)), encoding="utf-8",
        )
        if downstream is not None and variant in downstream:
            (variant_root / "local_downstream_metadata.json").write_text(
                json.dumps({"modis_clipped_day_count": downstream[variant]}),
                encoding="utf-8",
            )


# =============================================================================
# 1-6. Pre-label exclusion manifest binding
# =============================================================================
def test_file_name_constants_mirror_the_production_gate():
    """The mirrored names must equal the Step6B/Step8A production constants."""
    assert (
        wcs.PRELABEL_EXCLUSION_FILENAMES[wcs.PRELABEL_EXCLUSION_ROLE_MANIFEST]
        == step6b.PRE_LABEL_EXCLUSION_MANIFEST_PARQUET
        == step8a.PRE_LABEL_EXCLUSION_MANIFEST_FILENAME
    )
    assert (
        wcs.PRELABEL_EXCLUSION_FILENAMES[wcs.PRELABEL_EXCLUSION_ROLE_METADATA]
        == step6b.PRE_LABEL_EXCLUSION_MANIFEST_METADATA
        == step8a.PRE_LABEL_EXCLUSION_MANIFEST_METADATA_FILENAME
    )
    assert set(wcs.PRELABEL_EXCLUSION_AUDIT_COLUMNS) == set(
        wcs.STEP8A_OPTIONAL_AUDIT_COLUMNS
    )


def test_1_valid_binding_resolves_and_passes(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1"])
    assert binding["exclude_pre_label_burns"] is True
    assert binding["binding_ready"] is True
    assert binding["missing_required_documents"] == []
    for role in wcs.PRELABEL_EXCLUSION_REQUIRED_ROLES:
        record = binding["documents"][role]
        assert record["exists"] is True
        assert len(record["sha256"]) == 64
    assert wcs.assert_prelabel_exclusion_binding(binding, "test") is binding


def test_2_missing_binding_fails_closed(tmp_path):
    binding = _binding(tmp_path, active=True, cells=None)
    assert binding["binding_ready"] is False
    assert binding["missing_required_documents"] == sorted(
        wcs.PRELABEL_EXCLUSION_REQUIRED_ROLES
    )
    with pytest.raises(wcs.WindowClosureError, match="PRELABEL_EXCLUSION_BINDING_MISSING"):
        wcs.assert_prelabel_exclusion_binding(binding, "test")


def test_2b_a_partial_document_set_still_fails_closed(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1"])
    Path(binding["documents"][wcs.PRELABEL_EXCLUSION_ROLE_METADATA]["path"]).unlink()
    reloaded = _binding(tmp_path, active=True)
    assert reloaded["missing_required_documents"] == [
        wcs.PRELABEL_EXCLUSION_ROLE_METADATA
    ]
    with pytest.raises(wcs.WindowClosureError, match="PRELABEL_EXCLUSION_BINDING_MISSING"):
        wcs.assert_prelabel_exclusion_binding(reloaded, "test")


def test_3_manifest_and_audit_column_disagreement_fails(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1"])
    # The manifest excludes r1_c1; the dataset claims nothing was excluded.
    frame = _frame(excluded=[])
    with pytest.raises(
        wcs.WindowClosureError, match="PRELABEL_EXCLUSION_MANIFEST_DISAGREEMENT",
    ):
        wcs.assert_prelabel_exclusion_accounting(
            frame, tmp_path / "absent_stats.json", binding, "close_7d_earlier",
        )


def test_3b_agreement_reconciles_and_reports_real_counts(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1", "r2_c2", "not_in_variant"])
    frame = _frame(excluded=["r1_c1", "r2_c2"])
    record = wcs.assert_prelabel_exclusion_accounting(
        frame, tmp_path / "absent_stats.json", binding, "close_7d_earlier",
    )
    assert record["pre_label_burn_excluded_count"] == 2
    assert record["analysis_eligible_count"] == 4
    assert record["manifest_cell_count"] == 3
    assert record["manifest_cells_in_variant"] == 2
    assert record["accounting_reconciled"] is True


def test_4_inverse_relation_violation_fails(tmp_path):
    binding = _binding(tmp_path, active=True, cells=[])
    frame = _frame(excluded=[])
    frame.loc[0, "analysis_eligible"] = False  # neither eligible nor excluded
    with pytest.raises(wcs.WindowClosureError, match="analysis_eligible"):
        wcs.assert_prelabel_exclusion_accounting(
            frame, tmp_path / "absent_stats.json", binding, "close_7d_earlier",
        )


def test_4b_one_audit_column_without_the_other_fails(tmp_path):
    binding = _binding(tmp_path, active=True, cells=[])
    frame = _frame(excluded=[]).drop(columns=["analysis_eligible"])
    with pytest.raises(wcs.WindowClosureError, match="PRELABEL_EXCLUSION_AUDIT_MISSING"):
        wcs.assert_prelabel_exclusion_accounting(
            frame, tmp_path / "absent_stats.json", binding, "close_7d_earlier",
        )


def test_4c_stats_counters_must_agree_with_the_dataset(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1"])
    frame = _frame(excluded=["r1_c1"])
    stats = tmp_path / "step8a_dataset_stats.json"
    stats.write_text(
        json.dumps({"pre_label_burn_excluded_count": 99, "analysis_eligible_count": 5}),
        encoding="utf-8",
    )
    with pytest.raises(
        wcs.WindowClosureError, match="PRELABEL_EXCLUSION_ACCOUNTING_MISMATCH",
    ):
        wcs.assert_prelabel_exclusion_accounting(frame, stats, binding, "close_7d_earlier")

    stats.write_text(
        json.dumps({"pre_label_burn_excluded_count": 1, "analysis_eligible_count": 5}),
        encoding="utf-8",
    )
    record = wcs.assert_prelabel_exclusion_accounting(
        frame, stats, binding, "close_7d_earlier",
    )
    assert record["stats_counters"]["pre_label_burn_excluded_count"] == 1


def test_5_canonical_bytes_are_never_touched_by_the_binding(tmp_path):
    binding = _binding(tmp_path, active=True, cells=["r1_c1"])
    manifest = Path(binding["documents"][wcs.PRELABEL_EXCLUSION_ROLE_MANIFEST]["path"])
    before = manifest.read_bytes()
    frame = _frame(excluded=["r1_c1"])
    wcs.assert_prelabel_exclusion_accounting(
        frame, tmp_path / "absent_stats.json", binding, "close_7d_earlier",
    )
    assert manifest.read_bytes() == before
    assert binding["documents"][wcs.PRELABEL_EXCLUSION_ROLE_MANIFEST]["access"] == "read_only"


def test_6_an_experiment_without_the_policy_is_unaffected(tmp_path):
    """Bejís-shaped compatibility: no policy, no documents, no accounting."""
    binding = _binding(tmp_path, active=False, cells=None)
    assert binding["exclude_pre_label_burns"] is False
    assert binding["binding_ready"] is True
    assert binding["missing_required_documents"] == []
    wcs.assert_prelabel_exclusion_binding(binding, "test")

    record = wcs.assert_prelabel_exclusion_accounting(
        _frame(excluded=[]), tmp_path / "absent_stats.json", binding, "close_7d_earlier",
    )
    assert record["binding_active"] is False
    assert record["accounting_reconciled"] is True
    assert record["pre_label_burn_excluded_count"] is None


def test_6b_the_registry_drives_the_policy_not_an_aoi_name():
    """The real registry contract each production AOI will run under."""
    from core.experiment_context import build_experiment_context

    for aoi in ACTUAL_AOIS:
        context = build_experiment_context(aoi)
        binding = wcs.prelabel_exclusion_binding(aoi, context)
        assert binding["exclude_pre_label_burns"] == bool(
            context.get(wcs.PRELABEL_EXCLUSION_POLICY_FIELD, False)
        )
        # Whatever the policy says, it must be bindable right now.
        wcs.assert_prelabel_exclusion_binding(binding, "registry contract")


def test_6d_the_contract_is_satisfied_by_the_real_canonical_dataset():
    """The enforced contract must be satisfiable, not merely strict.

    Read-only: it opens the frozen canonical Step8A dataset of every
    policy-enabled AOI and reconciles it against that AOI's bound gate
    manifest, which is exactly what each shifted variant will have to satisfy.
    """
    from core.experiment_context import build_experiment_context

    checked = 0
    for aoi in ACTUAL_AOIS:
        context = build_experiment_context(aoi)
        binding = wcs.prelabel_exclusion_binding(aoi, context)
        if not binding["exclude_pre_label_burns"]:
            continue
        dataset = wcs.canonical_step8a_path(aoi)
        if not dataset.is_file() or not binding["binding_ready"]:
            continue
        record = wcs.assert_prelabel_exclusion_accounting(
            pd.read_parquet(dataset), wcs.canonical_step8a_stats_path(aoi),
            binding, f"{aoi}/canonical",
        )
        assert record["accounting_reconciled"] is True
        assert record["manifest_cells_in_variant"] == record["pre_label_burn_excluded_count"]
        assert (
            record["pre_label_burn_excluded_count"] + record["analysis_eligible_count"]
            == record["variant_row_count"]
        )
        checked += 1
    if not checked:
        pytest.skip("no policy-enabled AOI has its frozen canonical dataset locally")


def test_6c_the_step8a_input_role_declaration_follows_the_binding():
    without = wcs.production_stage_input_roles([2019, 2020])
    with_censor = wcs.production_stage_input_roles(
        [2019, 2020], wcs.PRELABEL_EXCLUSION_REQUIRED_ROLES,
    )
    assert wcs.PRELABEL_EXCLUSION_ROLE_MANIFEST not in without["step8a"]
    assert wcs.PRELABEL_EXCLUSION_ROLE_MANIFEST in with_censor["step8a"]
    assert without["step5"] == with_censor["step5"]


# =============================================================================
# 7-11. Cohort removal accounting
# =============================================================================
def test_7_real_non_zero_removal_counts_are_written_unchanged():
    accounting = derive_cohort_accounting(
        _cohort_metadata(), variants=VARIANTS, cohort_rows=100,
    )
    assert accounting["canonical"]["removed_prelabel_censor"] == 5
    assert accounting["canonical"]["removed_outside_primary_population"] == 10
    # Variant-specific numbers must NOT be flattened to one shared value.
    assert accounting["close_7d_earlier"]["removed_missing_required_feature_union"] == 30
    assert accounting["close_14d_earlier"]["removed_missing_required_feature_union"] == 25
    assert accounting["canonical"]["removed_missing_required_feature_union"] == 20


def test_8_a_placeholder_zero_cannot_be_published_when_the_source_is_non_zero():
    """The published row must equal the source, so a zeroed row cannot reconcile."""
    metadata = _cohort_metadata()
    zeroed = {
        **metadata,
        "removed_prelabel_censor": {variant: 0 for variant in VARIANTS},
    }
    with pytest.raises(MultiRegionWindowClosureError, match="COHORT_ACCOUNTING_MISMATCH"):
        derive_cohort_accounting(zeroed, variants=VARIANTS, cohort_rows=100)


def test_9_removal_arithmetic_reconciles_to_the_final_cohort():
    accounting = derive_cohort_accounting(
        _cohort_metadata(), variants=VARIANTS, cohort_rows=100,
    )
    for variant, row in accounting.items():
        removed = sum(
            row[field] for field in (
                "removed_not_valid_for_modeling", "removed_outside_primary_population",
                "removed_prelabel_censor", "removed_missing_required_feature_union",
                "removed_variant_only_keys", "removed_label_mismatch",
                "removed_static_invariance_failure",
            )
        )
        assert row["initial_rows"] - removed == row["final_common_cohort_rows"] == 100, variant


def test_10_a_reconciliation_mismatch_fails():
    metadata = _cohort_metadata()
    metadata["initial_rows_by_variant"]["close_7d_earlier"] += 1
    with pytest.raises(MultiRegionWindowClosureError, match="COHORT_ACCOUNTING_MISMATCH"):
        derive_cohort_accounting(metadata, variants=VARIANTS, cohort_rows=100)


def test_10b_the_persisted_cohort_must_match_the_declared_final_rows():
    with pytest.raises(MultiRegionWindowClosureError, match="COHORT_ACCOUNTING_MISMATCH"):
        derive_cohort_accounting(_cohort_metadata(), variants=VARIANTS, cohort_rows=99)


@pytest.mark.parametrize("mutation", [
    {"final_common_cohort_rows": None},
    {"initial_rows_by_variant": None},
    {"removed_prelabel_censor": {"canonical": 5}},
])
def test_11_a_missing_accounting_source_fails(mutation):
    metadata = {**_cohort_metadata(), **mutation}
    with pytest.raises(
        MultiRegionWindowClosureError, match="COHORT_ACCOUNTING_SOURCE_MISSING",
    ):
        derive_cohort_accounting(metadata, variants=VARIANTS, cohort_rows=100)


@pytest.mark.parametrize("value", [-1, 2.5, True, "7"])
def test_11b_a_non_integer_or_negative_count_fails(value):
    metadata = _cohort_metadata()
    metadata["removed_prelabel_censor"]["canonical"] = value
    with pytest.raises(MultiRegionWindowClosureError, match="COHORT_ACCOUNTING_INVALID"):
        derive_cohort_accounting(metadata, variants=VARIANTS, cohort_rows=100)


# =============================================================================
# 12-16. MODIS clipping propagation
# =============================================================================
def test_12_non_zero_clipping_propagates_from_export_provenance(tmp_path):
    _seed_export_provenance(tmp_path, "an_aoi", {"close_7d_earlier": 7, "close_14d_earlier": 14})
    resolved = resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)
    assert resolved["close_7d_earlier"]["clipped_day_count"] == 7
    assert resolved["close_14d_earlier"]["clipped_day_count"] == 14
    assert modis_clipping_summary_fields(resolved) == {
        "modis_clipped_days_7d": 7, "modis_clipped_days_14d": 14,
    }


def test_13_the_mugla_window_yields_exactly_seven_and_fourteen():
    """Derived from the registry window, not asserted from a literal table."""
    from core.experiment_context import build_experiment_context

    context = build_experiment_context("mugla_2021")
    variants = wcs.build_window_variants(context, (0, 7, 14))
    clipping = {
        variant["variant_id"]: wcs.modis_month_filter_transparency(
            variant["predictor_start_date"], variant["predictor_end_date"],
        )["clipped_day_count"]
        for variant in variants
    }
    assert clipping == {
        "canonical": 0, "close_7d_earlier": 7, "close_14d_earlier": 14,
    }

    # ...and that derivation is exactly what the export provenance carries, so
    # the summary fields it produces are 7 and 14, never 0.
    fields = modis_clipping_summary_fields({
        variant: {"clipped_day_count": clipping[variant]} for variant in SHIFTED_VARIANTS
    })
    assert fields == {"modis_clipped_days_7d": 7, "modis_clipped_days_14d": 14}


def test_14_zero_is_accepted_only_from_an_authoritative_source(tmp_path):
    """Bejís-shaped: a genuine zero is published as zero, with its provenance."""
    _seed_export_provenance(tmp_path, "an_aoi", {"close_7d_earlier": 0, "close_14d_earlier": 0})
    resolved = resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)
    assert modis_clipping_summary_fields(resolved) == {
        "modis_clipped_days_7d": 0, "modis_clipped_days_14d": 0,
    }
    assert resolved["close_7d_earlier"]["export_provenance"] == 0
    assert resolved["close_7d_earlier"]["source"].endswith("predictor_export_metadata.json")


def test_15_missing_clipping_information_is_never_converted_to_zero(tmp_path):
    (tmp_path / "an_aoi" / "variants" / "close_7d_earlier").mkdir(parents=True)
    (tmp_path / "an_aoi" / "variants" / "close_14d_earlier").mkdir(parents=True)
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_MISSING",
    ):
        resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)


def test_15b_an_export_record_without_a_transparency_block_fails(tmp_path):
    variant_root = tmp_path / "an_aoi" / "variants" / "close_7d_earlier"
    variant_root.mkdir(parents=True)
    (variant_root / "predictor_export_metadata.json").write_text(
        json.dumps({"artifact_inventory": [{"role": "modis_lst_mean"}]}), encoding="utf-8",
    )
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_MISSING",
    ):
        resolve_modis_clipping(tmp_path, "an_aoi", ("close_7d_earlier",))


def test_15c_an_inconsistent_numerator_denominator_fails(tmp_path):
    variant_root = tmp_path / "an_aoi" / "variants" / "close_7d_earlier"
    variant_root.mkdir(parents=True)
    metadata = _export_metadata(7, 53)
    metadata["artifact_inventory"][0]["date_semantics"]["duration_days"] = 61
    (variant_root / "predictor_export_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8",
    )
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_INCONSISTENT",
    ):
        resolve_modis_clipping(tmp_path, "an_aoi", ("close_7d_earlier",))


def test_16_a_summary_provenance_mismatch_fails(tmp_path):
    _seed_export_provenance(
        tmp_path, "an_aoi",
        {"close_7d_earlier": 7, "close_14d_earlier": 14},
        downstream={"close_7d_earlier": 0},
    )
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_MISMATCH",
    ):
        resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)


def test_16d_an_export_denominator_that_is_not_the_variant_window_fails(tmp_path):
    _seed_export_provenance(tmp_path, "an_aoi", {"close_7d_earlier": 7, "close_14d_earlier": 14})
    resolved = resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)
    windows = pd.DataFrame({
        "variant": list(SHIFTED_VARIANTS), "calendar_duration_days": [60, 60],
    })
    assert_modis_clipping_matches_windows(resolved, windows)  # denominators agree

    windows.loc[0, "calendar_duration_days"] = 57
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_MISMATCH",
    ):
        assert_modis_clipping_matches_windows(resolved, windows)


def test_16b_agreeing_provenance_layers_are_accepted(tmp_path):
    _seed_export_provenance(
        tmp_path, "an_aoi",
        {"close_7d_earlier": 7, "close_14d_earlier": 14},
        downstream={"close_7d_earlier": 7, "close_14d_earlier": 14},
    )
    resolved = resolve_modis_clipping(tmp_path, "an_aoi", SHIFTED_VARIANTS)
    assert resolved["close_7d_earlier"]["local_downstream_provenance"] == 7
    assert modis_clipping_summary_fields(resolved)["modis_clipped_days_7d"] == 7


def test_16c_two_modis_roles_that_disagree_fail(tmp_path):
    variant_root = tmp_path / "an_aoi" / "variants" / "close_7d_earlier"
    variant_root.mkdir(parents=True)
    metadata = _export_metadata(7, 53)
    metadata["artifact_inventory"][1]["date_semantics"][
        wcs.MODIS_CLIPPING_TRANSPARENCY_KEY
    ]["clipped_day_count"] = 3
    metadata["artifact_inventory"][1]["date_semantics"][
        wcs.MODIS_CLIPPING_TRANSPARENCY_KEY
    ]["effective_included_day_count"] = 57
    (variant_root / "predictor_export_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8",
    )
    with pytest.raises(
        MultiRegionWindowClosureError, match="MODIS_CLIPPING_PROVENANCE_INCONSISTENT",
    ):
        resolve_modis_clipping(tmp_path, "an_aoi", ("close_7d_earlier",))


# =============================================================================
# 17-20. Regression
# =============================================================================
def test_17_every_regional_aoi_is_still_supported():
    from src.multi_region_window_closure.contract import assert_regional_aoi

    for aoi in ("bejis_2022", "mugla_2021", "evia_2021_extended", "montiferru_2021"):
        assert assert_regional_aoi(aoi) == aoi


def test_18_no_multi_aoi_or_synthesis_dependency_is_reintroduced():
    import src.multi_region_window_closure.driver as driver
    import src.multi_region_window_closure.production as production

    for module in (driver, production):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "multi_aoi_transfer_synthesis" not in source
        assert "step9g" not in source


def test_19_the_regional_validator_schema_is_unchanged():
    from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS
    from src.multi_region_window_closure.validators import REGIONAL_CHECK_COUNTS
    from src.multi_region_window_closure.cohort import COHORT_INVENTORY_COLUMNS
    from src.multi_region_window_closure.schema import REGIONAL_SUMMARY_COLUMNS

    assert REGIONAL_CHECK_COUNTS == {"total": 32, "required": 31, "advisory": 1}
    spec = next(s for s in REGIONAL_ARTIFACT_SPECS if s.relative_path == "cohort_inventory.csv")
    assert spec.columns == COHORT_INVENTORY_COLUMNS
    assert "modis_clipped_days_7d" in REGIONAL_SUMMARY_COLUMNS
    assert "modis_clipped_days_14d" in REGIONAL_SUMMARY_COLUMNS


def test_20_no_network_gee_export_model_or_bootstrap_is_invoked(monkeypatch):
    """The new code paths must stay pure; any production side effect fails."""
    import src.multi_region_window_closure.production as production

    def _blocked(*_args, **_kwargs):
        raise AssertionError("a production side effect was invoked")

    monkeypatch.setattr(production, "derive_fit_accounting", _blocked)
    binding = wcs.prelabel_exclusion_binding("mugla_2021", {})
    assert binding["exclude_pre_label_burns"] is False  # empty context, no policy
    derive_cohort_accounting(_cohort_metadata(), variants=VARIANTS, cohort_rows=100)
