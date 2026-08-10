"""Regression tests for the generic multi-experiment burned-area
spatial-structure and descriptive comparison audit
(src/burned_pattern_audit.py, scripts/run_burned_pattern_audit.py,
core.pipeline_orchestrator.run_burned_pattern_audit_stage,
scripts/main.py `burned-pattern-audit`).

Uses entirely synthetic/placeholder experiment IDs and hand-built parquet
fixtures wherever possible (matching this repo's existing convention, see
tests/test_multi_aoi_transfer_synthesis.py), so nothing here depends on any
specific real AOI name."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.burned_pattern_audit as bpa
from scripts import run_burned_pattern_audit as direct_runner
from scripts.main import build_parser, cmd_burned_pattern_audit

# Placeholder experiment IDs for pure-logic / synthetic-fixture tests --
# these are NOT real registry entries (see item 2's test below, which
# confirms no real AOI name is hard-coded in the implementation).
FAKE_EXP_ALPHA = "aoi_alpha_2099"
FAKE_EXP_BETA = "aoi_beta_2099"
FAKE_EXP_FUTURE = "aoi_future_2099"


def make_frame(
    rows: list[int],
    cols: list[int],
    burned: list[int],
    elevation: list[float] | None = None,
    landcover: list[int] | None = None,
    burnable: list[bool] | None = None,
    burn_date: list[float] | None = None,
    valid_for_modeling: list[bool] | None = None,
) -> pd.DataFrame:
    n = len(rows)
    frame = pd.DataFrame({
        "row_500m": rows,
        "col_500m": cols,
        "burned": burned,
        "elevation_mean": elevation if elevation is not None else [float(i) for i in range(n)],
        "landcover_dominant": landcover if landcover is not None else [10] * n,
        "burnable_tree_shrub_grass": burnable if burnable is not None else [True] * n,
        # Step8B final-model row eligibility, defaulted to True so every
        # pre-existing fixture keeps its previous primary/sensitivity
        # semantics; only tests that care set it explicitly.
        "valid_for_modeling": valid_for_modeling if valid_for_modeling is not None else [True] * n,
    })
    if burn_date is not None:
        frame["burn_date"] = burn_date
    return frame


def write_fixture(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    root = tmp_path / name / "step8a"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "step8a_500m_modeling_dataset.parquet"
    frame.to_parquet(path, index=False)
    return path


def write_gate(tmp_path: Path, name: str, gate: dict) -> Path:
    root = tmp_path / name / "validation" / "labels"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "burned_landcover_gate.json"
    path.write_text(json.dumps(gate))
    return path


@pytest.fixture(autouse=True)
def _no_gate_by_default(monkeypatch, tmp_path):
    """Every synthetic-fixture test in this file defaults to "no canonical
    gate available" (gate_provenance/validate_against_gate degrade
    gracefully) unless a test explicitly patches canonical_gate_path again
    inside its own `with` block."""
    monkeypatch.setattr(bpa, "canonical_gate_path", lambda eid: tmp_path / "unused_gate" / "burned_landcover_gate.json")


# ---------------------------------------------------------------------------
# 8-9. Connected-component construction (pure logic, no I/O)
# ---------------------------------------------------------------------------
def test_diagonally_adjacent_cells_form_one_component():
    components, membership = bpa.build_components([(0, 0), (1, 1)])
    assert len(components) == 1
    assert membership[0] == membership[1]


def test_orthogonally_separated_patches_form_distinct_components():
    components, membership = bpa.build_components([(0, 0), (0, 5)])
    assert len(components) == 2
    assert membership[0] != membership[1]


# ---------------------------------------------------------------------------
# 10. Singleton components counted correctly
# ---------------------------------------------------------------------------
def test_singleton_components_counted_correctly():
    # Two isolated singletons plus one 3-cell component.
    coords = [(0, 0), (10, 10), (20, 20), (20, 21), (20, 22)]
    components, _ = bpa.build_components(coords)
    metrics = bpa.component_population_metrics(components, burned_cell_count=len(coords))
    assert metrics["component_count"] == 3
    assert metrics["singleton_component_count"] == 2
    assert metrics["singleton_component_fraction"] == pytest.approx(2 / 3)
    assert metrics["largest_component_cells"] == 3


# ---------------------------------------------------------------------------
# 11. Component sizes sum to the selected burned-cell count
# ---------------------------------------------------------------------------
def test_component_sizes_sum_to_burned_cell_count():
    coords = [(0, 0), (0, 1), (5, 5), (9, 9), (9, 10), (9, 11), (9, 12)]
    components, membership = bpa.build_components(coords)
    assert sum(c["size"] for c in components) == len(coords)
    assert len(membership) == len(coords)


# ---------------------------------------------------------------------------
# 12. Component IDs are deterministic (independent of input order)
# ---------------------------------------------------------------------------
def test_component_ids_are_deterministic_regardless_of_input_order():
    coords_a = [(0, 0), (0, 1), (5, 5), (9, 9), (9, 10)]
    coords_b = list(reversed(coords_a))
    components_a, membership_a = bpa.build_components(coords_a)
    components_b, membership_b = bpa.build_components(coords_b)

    coord_to_id_a = {coord: cid for coord, cid in zip(coords_a, membership_a)}
    coord_to_id_b = {coord: cid for coord, cid in zip(coords_b, membership_b)}
    assert coord_to_id_a == coord_to_id_b

    # Re-running with the same input is exactly reproducible.
    components_again, membership_again = bpa.build_components(coords_a)
    assert membership_again == membership_a
    assert [c["component_id"] for c in components_again] == [c["component_id"] for c in components_a]


# ---------------------------------------------------------------------------
# 13. Input experiment order does not change the analysis ID
# ---------------------------------------------------------------------------
def test_analysis_id_is_order_invariant():
    hashes_order_1 = {"b_exp": "hash_b", "a_exp": "hash_a"}
    hashes_order_2 = {"a_exp": "hash_a", "b_exp": "hash_b"}
    id_1 = bpa.build_analysis_id(("b_exp", "a_exp"), hashes_order_1)
    id_2 = bpa.build_analysis_id(("a_exp", "b_exp"), hashes_order_2)
    assert id_1 == id_2


# ---------------------------------------------------------------------------
# 14. Land-cover fractions sum to one
# ---------------------------------------------------------------------------
def test_landcover_fractions_sum_to_one():
    series = pd.Series([10, 10, 20, 30, 30, 30])
    rows, summary = bpa.landcover_mix(series)
    assert sum(r["fraction"] for r in rows) == pytest.approx(1.0)
    assert summary["dominant_landcover_code"] == 30
    assert summary["observed_landcover_class_count"] == 3


# ---------------------------------------------------------------------------
# 15. Unknown land-cover codes are retained (not dropped)
# ---------------------------------------------------------------------------
def test_unknown_landcover_codes_are_retained():
    assert bpa.landcover_label(999) == "unknown_class_999"
    series = pd.Series([10, 999, 999])
    rows, summary = bpa.landcover_mix(series)
    labels = {r["label"] for r in rows}
    assert "unknown_class_999" in labels
    assert summary["observed_landcover_class_count"] == 2


# ---------------------------------------------------------------------------
# 16. Elevation missing-value counts are preserved
# ---------------------------------------------------------------------------
def test_elevation_missing_value_counts_preserved():
    series = pd.Series([100.0, np.nan, 200.0, np.nan, np.nan])
    summary = bpa.elevation_summary(series)
    assert summary["valid_count"] == 2
    assert summary["missing_count"] == 3
    assert summary["valid_count"] + summary["missing_count"] == len(series)
    assert summary["min"] == 100.0
    assert summary["max"] == 200.0


# ---------------------------------------------------------------------------
# 6-7. Missing required column / duplicate grid coordinates fail clearly
# ---------------------------------------------------------------------------
def test_missing_required_column_fails_clearly():
    frame = make_frame([0, 1], [0, 1], [1, 0]).drop(columns=["elevation_mean"])
    with pytest.raises(bpa.BurnedPatternAuditError, match="elevation_mean"):
        bpa.validate_required_columns(list(frame.columns), FAKE_EXP_ALPHA)


def test_duplicate_grid_coordinates_fail_clearly():
    frame = make_frame([0, 0], [0, 0], [1, 1])
    with pytest.raises(bpa.BurnedPatternAuditError):
        bpa.validate_grid_uniqueness(frame, FAKE_EXP_ALPHA)


def test_invalid_burned_values_fail_clearly():
    frame = make_frame([0, 1], [0, 1], [1, 2])
    with pytest.raises(bpa.BurnedPatternAuditError):
        bpa.validate_burned_values(frame, FAKE_EXP_ALPHA)


# ---------------------------------------------------------------------------
# 5. Missing Step8A input fails clearly
# ---------------------------------------------------------------------------
def test_missing_step8a_input_fails_clearly(tmp_path):
    missing_path = tmp_path / "does_not_exist" / "step8a_500m_modeling_dataset.parquet"
    with patch.object(bpa, "canonical_step8a_path", return_value=missing_path):
        with pytest.raises(bpa.BurnedPatternAuditError, match="Missing canonical Step8A"):
            bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)


def test_resolve_experiments_explicit_missing_step8a_fails_clearly(tmp_path):
    missing_path = tmp_path / "nope" / "step8a_500m_modeling_dataset.parquet"
    with patch.object(bpa, "canonical_step8a_path", return_value=missing_path), \
         patch.object(bpa, "get_experiment", return_value={"experiment_id": FAKE_EXP_ALPHA, "enabled": True, "variant_status": "canonical"}):
        with pytest.raises(bpa.BurnedPatternAuditError, match="Missing canonical Step8A"):
            bpa.resolve_experiments(experiments=[FAKE_EXP_ALPHA])


# ---------------------------------------------------------------------------
# 3. --experiments / --all-enabled are mutually exclusive
# ---------------------------------------------------------------------------
def test_resolve_experiments_rejects_both_selectors():
    with pytest.raises(bpa.BurnedPatternAuditError):
        bpa.resolve_experiments(experiments=[FAKE_EXP_ALPHA], all_enabled=True)


def test_resolve_experiments_rejects_neither_selector():
    with pytest.raises(bpa.BurnedPatternAuditError):
        bpa.resolve_experiments(experiments=None, all_enabled=False)


def test_cli_rejects_both_selectors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["burned-pattern-audit", "--experiments", FAKE_EXP_ALPHA, "--all-enabled", "--dry-run"])


def test_cli_requires_one_selector():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["burned-pattern-audit", "--dry-run"])


# ---------------------------------------------------------------------------
# 1. CLI parses an arbitrary future experiment ID (no fixed choices list)
# ---------------------------------------------------------------------------
def test_cli_parses_arbitrary_future_experiment_id():
    parser = build_parser()
    args = parser.parse_args([
        "burned-pattern-audit", "--experiments", "totally_new_future_experiment_id_2099", "--dry-run",
    ])
    assert args.experiments == ["totally_new_future_experiment_id_2099"]
    assert args.all_enabled is False
    assert args.dry_run is True
    assert args.func is cmd_burned_pattern_audit


def test_cli_dispatches_through_orchestrator():
    parser = build_parser()
    args = parser.parse_args([
        "burned-pattern-audit", "--experiments", FAKE_EXP_ALPHA, FAKE_EXP_BETA, "--dry-run",
    ])
    with patch.object(sys.modules["scripts.main"].orch, "run_burned_pattern_audit_stage", return_value={"ran": False}) as mocked:
        assert cmd_burned_pattern_audit(args) == 0
    mocked.assert_called_once_with(
        experiments=[FAKE_EXP_ALPHA, FAKE_EXP_BETA], all_enabled=False, dry_run=True, force=False,
        allow_superseded_sensitivity=False,
    )


# ---------------------------------------------------------------------------
# 2. No source implementation contains a fixed allowed-AOI list
# ---------------------------------------------------------------------------
def test_no_hardcoded_real_experiment_ids_in_implementation():
    from core.regions import EXPERIMENTS as REAL_REGISTRY

    real_ids = list(REAL_REGISTRY.keys())
    for source_path in (
        _PROJECT_ROOT / "src" / "burned_pattern_audit.py",
        _PROJECT_ROOT / "scripts" / "run_burned_pattern_audit.py",
    ):
        text = source_path.read_text()
        for experiment_id in real_ids:
            assert re.search(rf"\b{re.escape(experiment_id)}\b", text) is None, (
                f"{source_path} appears to hard-code real experiment_id '{experiment_id}'."
            )


# ---------------------------------------------------------------------------
# 4, 18. Dry-run writes no files; Step8A hashes unchanged after execution
# ---------------------------------------------------------------------------
def test_dry_run_writes_no_files_and_computes_no_components(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir), \
         patch.object(bpa, "get_experiment", return_value={"experiment_id": FAKE_EXP_ALPHA, "enabled": True, "variant_status": "canonical"}):
        result = bpa.run_analysis(experiments=[FAKE_EXP_ALPHA], dry_run=True)

    assert result["ran"] is False
    assert not output_root.exists()
    assert not comparison_dir.exists()


def test_step8a_hash_unchanged_after_real_run(tmp_path):
    frame = make_frame(
        rows=[0, 0, 1, 5],
        cols=[0, 1, 1, 5],
        burned=[1, 1, 1, 0],
        elevation=[100.0, 110.0, 120.0, 130.0],
        landcover=[10, 20, 10, 40],
        burnable=[True, True, True, False],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    before_bytes = path.read_bytes()
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    assert result["ran"] is True
    assert path.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# 17. Primary all-valid and natural-vegetation sensitivity populations
#     remain distinct
# ---------------------------------------------------------------------------
def test_primary_and_sensitivity_populations_remain_distinct(tmp_path):
    frame = make_frame(
        rows=[0, 0, 1, 2],
        cols=[0, 1, 1, 2],
        burned=[1, 1, 1, 1],
        landcover=[10, 40, 10, 40],
        burnable=[True, False, True, False],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    all_valid = result["populations"][bpa.POPULATION_ALL_VALID_BURNED]
    sensitivity = result["populations"][bpa.POPULATION_BURNABLE_TSG_BURNED]
    assert all_valid["metrics"]["total_burned_cells"] == 4
    assert sensitivity["metrics"]["total_burned_cells"] == 2
    assert all_valid["metrics"]["total_burned_cells"] != sensitivity["metrics"]["total_burned_cells"]


# ---------------------------------------------------------------------------
# The sensitivity-population description must state the real frozen Step8A
# rule -- a COMBINED tree+shrub+grass fraction threshold -- and must not
# claim a dominant-land-cover-based selection.
# ---------------------------------------------------------------------------
def test_sensitivity_population_definition_states_combined_fraction_rule():
    definition = bpa.POPULATION_DEFINITIONS[bpa.POPULATION_BURNABLE_TSG_BURNED]
    normalized = " ".join(definition.split())

    # The combined-fraction rule, with all three summed fractions named.
    for fragment in (
        "landcover_tree_cover_fraction",
        "landcover_shrubland_fraction",
        "landcover_grassland_fraction",
    ):
        assert fragment in normalized, f"Definition must name '{fragment}': {normalized!r}"
    assert (
        "(landcover_tree_cover_fraction + landcover_shrubland_fraction + "
        "landcover_grassland_fraction) >= burnable_threshold"
    ) in normalized, f"Definition must state the summed-fraction threshold rule: {normalized!r}"

    # No dominant-class-based selection claim of any form.
    lowered = normalized.lower()
    for banned in (
        "tree/shrub/grass dominant land cover",
        "dominant land cover, per-cell fraction threshold",
    ):
        assert banned not in lowered, f"Stale dominant-class claim survives: {banned!r}"
    assert "landcover_dominant" in normalized, (
        "Definition must say explicitly that landcover_dominant is not the rule."
    )
    assert "no part of the selection rule" in lowered, (
        f"Definition must disclaim landcover_dominant as a selection rule: {normalized!r}"
    )

    # Still reused verbatim from Step8A, never recomputed here.
    assert bpa.BURNABLE_MASK_COLUMN in normalized
    assert "reused verbatim" in lowered and "never recomputed" in lowered


def test_sensitivity_population_selection_ignores_landcover_dominant(tmp_path):
    """Behavioural counterpart: membership follows the frozen Step8A boolean
    column verbatim, so a cropland-dominant cell whose combined fractions
    cleared the Step8A threshold is IN, and a tree-dominant cell whose
    combined fractions did not is OUT."""
    frame = make_frame(
        rows=[0, 5, 10],
        cols=[0, 5, 10],
        burned=[1, 1, 1],
        # 40 = cropland-dominant, 10 = tree-cover-dominant.
        landcover=[40, 10, 10],
        burnable=[True, False, True],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", tmp_path / "diagnostics" / "experiments"), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", tmp_path / "diagnostics" / "comparison"):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    sensitivity = result["populations"][bpa.POPULATION_BURNABLE_TSG_BURNED]
    # 2 of 3 burned cells: the cropland-dominant True row is kept, the
    # tree-dominant False row is dropped.
    assert sensitivity["metrics"]["total_burned_cells"] == 2
    assert result["populations"][bpa.POPULATION_ALL_VALID_BURNED]["metrics"]["total_burned_cells"] == 3


# ---------------------------------------------------------------------------
# Component membership covers every selected burned cell exactly once, and
# component-summary/elevation/landcover files are internally consistent.
# ---------------------------------------------------------------------------
def test_full_analysis_writes_consistent_outputs(tmp_path):
    frame = make_frame(
        rows=[0, 0, 1, 9, 20],
        cols=[0, 1, 1, 9, 20],
        burned=[1, 1, 1, 1, 0],
        elevation=[100.0, 110.0, np.nan, 130.0, 999.0],
        landcover=[10, 20, 10, 999, 40],
        burnable=[True, True, True, False, False],
        burn_date=[200.0, 201.0, 202.0, 210.0, np.nan],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    out_dir = Path(result["output_dir"])
    for name in (
        "burned_pattern_summary.json", "component_summary.csv", "component_membership.parquet",
        "elevation_summary.csv", "landcover_mix.csv", "burned_pattern_summary.md", "manifest.json",
    ):
        assert (out_dir / name).is_file()

    membership = pd.read_parquet(out_dir / "component_membership.parquet")
    all_valid_membership = membership[membership["population"] == bpa.POPULATION_ALL_VALID_BURNED]
    assert len(all_valid_membership) == 4  # 4 burned cells in all_valid_burned
    assert not all_valid_membership[["row_500m", "col_500m"]].duplicated().any()

    component_summary = pd.read_csv(out_dir / "component_summary.csv")
    all_valid_components = component_summary[component_summary["population"] == bpa.POPULATION_ALL_VALID_BURNED]
    assert all_valid_components["size_cells"].sum() == 4

    landcover_mix_df = pd.read_csv(out_dir / "landcover_mix.csv")
    all_valid_lc = landcover_mix_df[landcover_mix_df["population"] == bpa.POPULATION_ALL_VALID_BURNED]
    assert abs(all_valid_lc["fraction"].sum() - 1.0) < 1e-9
    assert "unknown_class_999" in set(all_valid_lc["label"])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["resolved_experiment_ids"] == [FAKE_EXP_ALPHA]

    summary_json = json.loads((out_dir / "burned_pattern_summary.json").read_text())
    assert any(
        "Connected components are a spatial fragmentation proxy" in line
        for line in summary_json["limitations"]
    )


# ---------------------------------------------------------------------------
# 9. Regression: a pre-label-excluded row with burned == 1 must not enter
#    components, elevation, or landcover summaries.
# ---------------------------------------------------------------------------
def test_pre_label_excluded_burned_row_is_excluded_from_all_summaries(tmp_path):
    frame = make_frame(
        rows=[0, 0, 50],
        cols=[0, 1, 50],
        burned=[1, 1, 1],
        elevation=[100.0, 110.0, 9999.0],
        landcover=[10, 10, 40],
        burnable=[True, True, False],
    )
    # Row 50/50 burned == 1 but was pre-label-excluded -- must never enter
    # the analysis universe, regardless of its raw burned label.
    frame["analysis_eligible"] = [True, True, False]
    frame["pre_label_burn_excluded"] = [False, False, True]
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    all_valid = result["populations"][bpa.POPULATION_ALL_VALID_BURNED]
    assert all_valid["metrics"]["total_burned_cells"] == 2  # excludes the pre-label-excluded row
    assert all_valid["analysis_eligible_rows"] == 2

    membership_coords = set(
        zip(all_valid["membership_df"]["row_500m"], all_valid["membership_df"]["col_500m"])
    )
    assert (50, 50) not in membership_coords

    assert all_valid["elevation"]["valid_count"] == 2
    assert all_valid["elevation"]["max"] == 110.0  # 9999.0 (pre-label-excluded row) never surfaces

    landcover_codes = {row["class_code"] for row in all_valid["landcover"]["classes"]}
    assert 40 not in landcover_codes  # the pre-label-excluded row's landcover never surfaces

    gate_info = result["manifest"]["pre_label_gate"]
    assert gate_info["pre_label_burn_excluded_count"] == 1


def test_resolve_analysis_eligible_mask_defaults_to_all_true_when_column_absent():
    frame = make_frame([0, 1], [0, 1], [1, 1])
    mask = bpa.resolve_analysis_eligible_mask(frame)
    assert bool(mask.all())
    assert len(mask) == len(frame)


def test_resolve_analysis_eligible_mask_respects_column_when_present():
    frame = make_frame([0, 1, 2], [0, 1, 2], [1, 1, 1])
    frame["analysis_eligible"] = [True, False, True]
    mask = bpa.resolve_analysis_eligible_mask(frame)
    assert mask.tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# 5-7. Canonical burned-landcover gate cross-validation and manifest
#      provenance (path, SHA-256, exclusion rule, excluded count).
# ---------------------------------------------------------------------------
def test_gate_unavailable_is_recorded_and_does_not_block(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    gate_info = result["manifest"]["pre_label_gate"]
    assert gate_info["gate_available"] is False
    assert gate_info["pre_label_burn_excluded_count"] == 0


def test_gate_burned_count_mismatch_fails_before_writing_outputs(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    gate_path = write_gate(tmp_path, FAKE_EXP_ALPHA, {
        "analysis_universe_cells_after_exclusions": 2,
        "burned_count": 999,  # deliberately wrong
        "pre_label_burn_excluded_count": 0,
        "pre_label_burn_exclusion_rule": "valid nonzero BurnDate calendar date < label_start",
    })
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "canonical_gate_path", return_value=gate_path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        with pytest.raises(bpa.BurnedPatternAuditError, match="burned-cell count"):
            bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    assert not output_root.exists()


def test_gate_analysis_universe_mismatch_fails_before_writing_outputs(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    gate_path = write_gate(tmp_path, FAKE_EXP_ALPHA, {
        "analysis_universe_cells_after_exclusions": 999,  # deliberately wrong
        "burned_count": 2,
        "pre_label_burn_excluded_count": 0,
    })
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "canonical_gate_path", return_value=gate_path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        with pytest.raises(bpa.BurnedPatternAuditError, match="analysis universe"):
            bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    assert not output_root.exists()


def test_gate_pre_label_excluded_count_mismatch_fails_before_writing_outputs(tmp_path):
    frame = make_frame([0, 1, 2], [0, 1, 2], [1, 1, 1])
    frame["analysis_eligible"] = [True, True, False]
    frame["pre_label_burn_excluded"] = [False, False, True]
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    gate_path = write_gate(tmp_path, FAKE_EXP_ALPHA, {
        "analysis_universe_cells_after_exclusions": 2,
        "burned_count": 2,
        "pre_label_burn_excluded_count": 999,  # deliberately wrong
    })
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "canonical_gate_path", return_value=gate_path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        with pytest.raises(bpa.BurnedPatternAuditError, match="pre-label-excluded count"):
            bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    assert not output_root.exists()


def test_gate_matching_values_pass_and_are_recorded_in_manifest(tmp_path):
    frame = make_frame([0, 1, 2], [0, 1, 2], [1, 1, 1])
    frame["analysis_eligible"] = [True, True, False]
    frame["pre_label_burn_excluded"] = [False, False, True]
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    gate_path = write_gate(tmp_path, FAKE_EXP_ALPHA, {
        "analysis_universe_cells_after_exclusions": 2,
        "burned_count": 2,
        "pre_label_burn_excluded_count": 1,
        "pre_label_burn_exclusion_rule": "valid nonzero BurnDate calendar date < label_start",
    })
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "canonical_gate_path", return_value=gate_path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    gate_info = result["manifest"]["pre_label_gate"]
    assert gate_info["gate_available"] is True
    assert gate_info["validated"] is True
    assert gate_info["gate_path"] == str(gate_path)
    assert gate_info["gate_sha256"] == bpa.sha256_file(gate_path)
    assert gate_info["pre_label_burn_exclusion_rule"] == "valid nonzero BurnDate calendar date < label_start"
    assert gate_info["pre_label_burn_excluded_count"] == 1


# ---------------------------------------------------------------------------
# Force / manifest guard behavior
# ---------------------------------------------------------------------------
def test_rerun_without_force_but_matching_analysis_id_is_idempotent(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        first = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)
        second = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    assert first["analysis_id"] == second["analysis_id"]


def test_rerun_with_changed_input_requires_force(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

        # Change the underlying fixture (different burned pattern -> different hash).
        frame2 = make_frame([0, 1, 2], [0, 1, 2], [1, 1, 1])
        frame2.to_parquet(path, index=False)

        with pytest.raises(bpa.BurnedPatternAuditError, match="different analysis_id"):
            bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

        # --force overwrites cleanly.
        forced = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False, force=True)
        assert forced["ran"] is True


# ---------------------------------------------------------------------------
# 20. Multi-experiment comparison automatically includes a mocked future
#     experiment resolved through the experiment registry (--all-enabled).
# ---------------------------------------------------------------------------
def test_all_enabled_comparison_includes_mocked_future_experiment(tmp_path):
    # Synthetic registry records must declare variant_status: the resolver
    # validates it fail-closed and never assumes "canonical" for a record
    # that omits it.
    fake_registry = {
        FAKE_EXP_ALPHA: {"experiment_id": FAKE_EXP_ALPHA, "enabled": True,
                         "variant_status": "canonical"},
        FAKE_EXP_FUTURE: {"experiment_id": FAKE_EXP_FUTURE, "enabled": True,
                          "variant_status": "canonical"},
    }
    fixture_paths = {
        FAKE_EXP_ALPHA: write_fixture(
            tmp_path, FAKE_EXP_ALPHA,
            make_frame([0, 1], [0, 1], [1, 1]),
        ),
        FAKE_EXP_FUTURE: write_fixture(
            tmp_path, FAKE_EXP_FUTURE,
            make_frame([0, 5], [0, 5], [1, 0]),
        ),
    }
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"

    with patch.object(bpa, "list_experiments", return_value=fake_registry), \
         patch.object(bpa, "LEGACY_EXPERIMENT_ID", "no_such_legacy_id"), \
         patch.object(bpa, "canonical_step8a_path", side_effect=lambda eid: fixture_paths[eid]), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.run_analysis(all_enabled=True, dry_run=False)

    assert result["ran"] is True
    assert FAKE_EXP_FUTURE in result["resolved_experiment_ids"]
    assert FAKE_EXP_ALPHA in result["resolved_experiment_ids"]

    # The run namespaces itself by scope under the canonical root, so the
    # comparison directory is read from the result rather than assumed.
    resolved_comparison = Path(result["comparison"]["output_dir"])
    assert resolved_comparison == comparison_dir.parent / "all_enabled" / "comparison"
    assert str(resolved_comparison).startswith(str(tmp_path))  # never the real root
    comparison_csv = pd.read_csv(resolved_comparison / "multi_aoi_burned_pattern_comparison.csv")
    assert FAKE_EXP_FUTURE in set(comparison_csv["experiment_id"])
    for column in (
        "experiment_id", "population", "burned_cell_count", "component_count",
        "singleton_component_count", "largest_component_cells", "largest_component_fraction",
        "top3_component_fraction", "component_size_median", "component_size_q90",
        "component_size_q95", "component_size_max", "elevation_min", "elevation_q05",
        "elevation_median", "elevation_q95", "elevation_max", "dominant_landcover_code",
        "dominant_landcover_label", "dominant_landcover_fraction", "observed_landcover_class_count",
        "edge_touching_component_count", "largest_component_touches_edge",
    ):
        assert column in comparison_csv.columns


def test_all_enabled_excludes_experiment_without_step8a(tmp_path):
    fake_registry = {
        FAKE_EXP_ALPHA: {"experiment_id": FAKE_EXP_ALPHA, "enabled": True,
                         "variant_status": "canonical"},
        FAKE_EXP_BETA: {"experiment_id": FAKE_EXP_BETA, "enabled": True,
                        "variant_status": "canonical"},
    }
    alpha_path = write_fixture(tmp_path, FAKE_EXP_ALPHA, make_frame([0], [0], [1]))
    beta_missing_path = tmp_path / FAKE_EXP_BETA / "step8a" / "step8a_500m_modeling_dataset.parquet"

    def fake_path(eid):
        return alpha_path if eid == FAKE_EXP_ALPHA else beta_missing_path

    with patch.object(bpa, "list_experiments", return_value=fake_registry), \
         patch.object(bpa, "LEGACY_EXPERIMENT_ID", "no_such_legacy_id"), \
         patch.object(bpa, "canonical_step8a_path", side_effect=fake_path):
        resolution = bpa.resolve_experiments(all_enabled=True)

    assert resolution.resolved_ids == (FAKE_EXP_ALPHA,)
    assert FAKE_EXP_BETA in resolution.excluded


# ---------------------------------------------------------------------------
# 19. Existing Step9/Step9G/Step10 artifacts are not modified -- output
#     namespace is fixed and unrelated to those paths, and a full run only
#     touches its own designated output tree.
# ---------------------------------------------------------------------------
def test_output_namespace_is_isolated_from_step9_step10():
    for path in (bpa.OUTPUT_ROOT, bpa.EXPERIMENTS_OUTPUT_ROOT, bpa.COMPARISON_OUTPUT_DIR):
        text = str(path).lower()
        assert "step9" not in text
        assert "step10" not in text
    assert bpa.OUTPUT_ROOT == _PROJECT_ROOT / "outputs" / "diagnostics" / "burned_pattern_audit"


def test_full_run_touches_only_its_own_output_tree(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)
    output_root = tmp_path / "diagnostics" / "experiments"
    comparison_dir = tmp_path / "diagnostics" / "comparison"
    sentinel_dir = tmp_path / "diagnostics"

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "get_experiment", return_value={"experiment_id": FAKE_EXP_ALPHA, "enabled": True, "variant_status": "canonical"}), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", output_root), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.run_analysis(experiments=[FAKE_EXP_ALPHA], dry_run=False)

    # Writes are namespaced by scope under the sandbox parent; nothing escapes it.
    scope_dir = sentinel_dir / FAKE_EXP_ALPHA
    written = {p for p in sentinel_dir.rglob("*") if p.is_file()}
    assert written
    assert all(str(p).startswith(str(scope_dir)) for p in written)
    assert Path(result["comparison"]["output_dir"]) == scope_dir / "comparison"


# ---------------------------------------------------------------------------
# One true integration test against real registry data (read-only, no
# writes) -- confirms the exact corrected mugla_2021 numbers from the
# pre-label-exclusion bug report. Matches this repo's existing convention
# of a single real-ID integration test alongside synthetic-fixture unit
# tests (see tests/test_multi_aoi_transfer_synthesis.py).
# ---------------------------------------------------------------------------
def test_real_mugla_2021_corrected_primary_population_matches_expected_numbers():
    path = bpa.canonical_step8a_path("mugla_2021")
    if not path.is_file():
        pytest.skip("real mugla_2021 Step8A dataset not present in this environment")

    df = pd.read_parquet(path)
    eligible_mask = bpa.resolve_analysis_eligible_mask(df)
    valid_extent = set(zip(df["row_500m"].astype(int), df["col_500m"].astype(int)))
    result = bpa._analyze_population(df, bpa.POPULATION_ALL_VALID_BURNED, valid_extent, eligible_mask)

    assert result["analysis_eligible_rows"] == 73049
    assert result["metrics"]["total_burned_cells"] == 3026

    counts = {row["class_code"]: row["count"] for row in result["landcover"]["classes"]}
    assert counts.get(10) == 2692  # tree_cover
    assert counts.get(20) == 73    # shrubland
    assert counts.get(30) == 135   # grassland
    assert counts.get(40) == 39    # cropland
    assert counts.get(80) == 75    # permanent_water
    assert counts.get(50) == 10    # built_up
    assert counts.get(60) == 2     # bare_sparse_vegetation


# ===========================================================================
# Superseded legacy-variant AOI-definition sensitivity permission
# (--allow-superseded-sensitivity). Synthetic registry records only.
# ===========================================================================
FAKE_EXP_LEGACY = "aoi_legacy_2099"
FAKE_EXP_SUCCESSOR = "aoi_successor_2099"


def _synthetic_registry(**overrides) -> dict:
    """Legacy variant + its canonical successor, as registry-shaped records."""
    registry = {
        FAKE_EXP_LEGACY: {
            "experiment_id": FAKE_EXP_LEGACY, "enabled": True,
            "variant_status": "legacy_superseded", "superseded_by": FAKE_EXP_SUCCESSOR,
        },
        FAKE_EXP_SUCCESSOR: {
            "experiment_id": FAKE_EXP_SUCCESSOR, "enabled": True,
            "variant_status": "canonical",
        },
        FAKE_EXP_ALPHA: {
            "experiment_id": FAKE_EXP_ALPHA, "enabled": True,
            "variant_status": "canonical",
        },
    }
    registry.update(overrides)
    return registry


def _patch_registry(registry: dict, paths: dict[str, Path]):
    """Patch the module's registry lookup and Step8A path resolution.
    `validate_variant_record` is deliberately NOT patched: the real
    fail-closed variant validation runs against the synthetic records."""
    def _get_experiment(experiment_id: str) -> dict:
        if experiment_id not in registry:
            raise ValueError(f"unknown experiment_id: {experiment_id}")
        return registry[experiment_id]

    return (
        patch.object(bpa, "get_experiment", side_effect=_get_experiment),
        patch.object(bpa, "canonical_step8a_path", side_effect=lambda eid: paths[eid]),
    )


def _legacy_pair_fixtures(tmp_path: Path) -> dict[str, Path]:
    return {
        FAKE_EXP_LEGACY: write_fixture(
            tmp_path, FAKE_EXP_LEGACY, make_frame([0, 0, 5], [0, 1, 5], [1, 1, 1]),
        ),
        FAKE_EXP_SUCCESSOR: write_fixture(
            tmp_path, FAKE_EXP_SUCCESSOR, make_frame([0, 0, 1, 9], [0, 1, 1, 9], [1, 1, 1, 1]),
        ),
        FAKE_EXP_ALPHA: write_fixture(
            tmp_path, FAKE_EXP_ALPHA, make_frame([0, 1], [0, 1], [1, 1]),
        ),
    }


def test_default_explicit_superseded_selection_still_fails(tmp_path):
    """Unchanged default behaviour: no flag -> superseded ID is refused with
    the existing message, and it is never silently dropped."""
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        with pytest.raises(bpa.BurnedPatternAuditError, match="refusing superseded experiment"):
            bpa.resolve_experiments(experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR])


def test_force_does_not_bypass_the_superseded_resolver_policy(tmp_path):
    """--force governs overwriting outputs only; it can never stand in for
    the sensitivity permission."""
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        with pytest.raises(bpa.BurnedPatternAuditError, match="refusing superseded experiment"):
            bpa.run_analysis(
                experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR], dry_run=True, force=True,
            )


def test_flag_with_legacy_and_successor_resolves_both(tmp_path):
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        resolution = bpa.resolve_experiments(
            experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR],
            allow_superseded_sensitivity=True,
        )

    assert set(resolution.resolved_ids) == {FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR}
    assert resolution.selection_mode == bpa.SELECTION_MODE_EXPLICIT_SUPERSEDED_SENSITIVITY
    assert resolution.selection_mode == "explicit_superseded_sensitivity"
    assert resolution.allow_superseded_sensitivity is True
    assert resolution.superseded_pairs == ((FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR),)
    assert resolution.excluded == {}


def test_flag_with_legacy_but_no_successor_fails(tmp_path):
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        with pytest.raises(bpa.BurnedPatternAuditError, match="is not in the requested selection"):
            bpa.resolve_experiments(
                experiments=[FAKE_EXP_LEGACY], allow_superseded_sensitivity=True,
            )


def test_flag_with_all_enabled_fails(tmp_path):
    with pytest.raises(bpa.BurnedPatternAuditError, match="cannot be combined with --all-enabled"):
        bpa.resolve_experiments(all_enabled=True, allow_superseded_sensitivity=True)


def test_flag_without_any_superseded_experiment_fails(tmp_path):
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        with pytest.raises(bpa.BurnedPatternAuditError, match="contains no superseded experiment"):
            bpa.resolve_experiments(
                experiments=[FAKE_EXP_ALPHA, FAKE_EXP_SUCCESSOR],
                allow_superseded_sensitivity=True,
            )


def test_flag_fails_when_successor_is_not_canonical(tmp_path):
    """The successor must itself validate as a canonical variant -- chaining
    a legacy variant onto another legacy variant is refused."""
    registry = _synthetic_registry()
    registry[FAKE_EXP_SUCCESSOR] = {
        "experiment_id": FAKE_EXP_SUCCESSOR, "enabled": True,
        "variant_status": "legacy_superseded", "superseded_by": FAKE_EXP_ALPHA,
    }
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(registry, paths)
    with get_exp, get_path:
        with pytest.raises(bpa.BurnedPatternAuditError, match="variant_status"):
            bpa.resolve_experiments(
                experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR],
                allow_superseded_sensitivity=True,
            )


def test_all_enabled_discovery_is_unchanged_and_still_excludes_legacy(tmp_path):
    """(D) The permission cannot reach discovery, and discovery keeps
    excluding the legacy variant with its superseded_by reason."""
    registry = _synthetic_registry()
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(registry, paths)
    with get_exp, get_path, patch.object(bpa, "list_experiments", return_value=registry):
        resolution = bpa.resolve_experiments(all_enabled=True)

    assert FAKE_EXP_LEGACY not in resolution.resolved_ids
    assert resolution.excluded[FAKE_EXP_LEGACY] == f"superseded_by_{FAKE_EXP_SUCCESSOR}"
    assert resolution.selection_mode == "all_enabled"
    assert resolution.allow_superseded_sensitivity is False
    assert resolution.superseded_pairs == ()


def test_explicit_canonical_selection_keeps_default_selection_mode(tmp_path):
    """A plain canonical selection is untouched by the new field defaults --
    this is the exact call shape src/domain_classifier_audit.py makes."""
    paths = _legacy_pair_fixtures(tmp_path)
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)
    with get_exp, get_path:
        resolution = bpa.resolve_experiments(experiments=[FAKE_EXP_ALPHA, FAKE_EXP_SUCCESSOR])

    assert resolution.selection_mode == "explicit"
    assert resolution.allow_superseded_sensitivity is False
    assert resolution.superseded_pairs == ()


# ---------------------------------------------------------------------------
# CLI plumbing for the flag
# ---------------------------------------------------------------------------
def test_main_parser_parses_allow_superseded_sensitivity():
    parser = build_parser()
    args = parser.parse_args([
        "burned-pattern-audit", "--experiments", FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR,
        "--allow-superseded-sensitivity", "--dry-run",
    ])
    assert args.allow_superseded_sensitivity is True

    default_args = parser.parse_args([
        "burned-pattern-audit", "--experiments", FAKE_EXP_ALPHA, "--dry-run",
    ])
    assert default_args.allow_superseded_sensitivity is False


def test_orchestrator_dispatch_forwards_allow_superseded_sensitivity():
    parser = build_parser()
    args = parser.parse_args([
        "burned-pattern-audit", "--experiments", FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR,
        "--allow-superseded-sensitivity", "--dry-run",
    ])
    with patch.object(sys.modules["scripts.main"].orch, "run_burned_pattern_audit_stage", return_value={"ran": False}) as mocked:
        assert cmd_burned_pattern_audit(args) == 0
    mocked.assert_called_once_with(
        experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR], all_enabled=False,
        dry_run=True, force=False, allow_superseded_sensitivity=True,
    )


def test_orchestrator_stage_forwards_flag_to_runner():
    with patch("scripts.run_burned_pattern_audit.main", return_value={"ran": False}) as mocked:
        from core import pipeline_orchestrator as orch_module
        orch_module.run_burned_pattern_audit_stage(
            experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR], all_enabled=False,
            dry_run=True, force=False, allow_superseded_sensitivity=True,
        )
    mocked.assert_called_once_with(
        experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR], all_enabled=False,
        dry_run=True, force=False, allow_superseded_sensitivity=True,
    )


def test_direct_runner_parser_parses_flag_and_defaults_false():
    parser = direct_runner.build_parser()
    args = parser.parse_args([
        "--experiments", FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR, "--allow-superseded-sensitivity",
    ])
    assert args.allow_superseded_sensitivity is True
    assert parser.parse_args(["--experiments", FAKE_EXP_ALPHA]).allow_superseded_sensitivity is False


def test_direct_runner_main_defaults_to_no_permission():
    with patch("scripts.run_burned_pattern_audit.run_analysis", return_value={"ran": False}) as mocked:
        direct_runner.main(experiments=[FAKE_EXP_ALPHA], dry_run=True)
    mocked.assert_called_once_with(
        experiments=[FAKE_EXP_ALPHA], all_enabled=False, dry_run=True, force=False,
        allow_superseded_sensitivity=False,
        # Output-path arguments default to "canonical root, scope derived from
        # the selection"; the runner spells out no path of its own.
        output_root=None, scope=None,
    )


# ---------------------------------------------------------------------------
# Third population: final-model-population morphology comparison
# ---------------------------------------------------------------------------
def test_schema_version_is_v2_and_populations_are_ordered():
    assert bpa.ANALYSIS_SCHEMA_VERSION == "burned_pattern_audit.v2"
    assert bpa.VALID_FOR_MODELING_COLUMN == "valid_for_modeling"
    assert bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED == (
        "burnable_tree_shrub_grass_valid_for_modeling_burned"
    )
    assert bpa.POPULATIONS == (
        bpa.POPULATION_ALL_VALID_BURNED,
        bpa.POPULATION_BURNABLE_TSG_BURNED,
        bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED,
    )


def test_new_population_definition_states_the_full_contract():
    definition = " ".join(
        bpa.POPULATION_DEFINITIONS[bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED].split()
    )
    for clause in (
        "analysis_eligible == True",
        "valid_for_modeling == True",
        "burnable_tree_shrub_grass == True",
        "burned == 1",
        "non-null row_500m/col_500m",
    ):
        assert clause in definition, f"missing contract clause {clause!r}: {definition!r}"
    assert "does NOT replace" in definition


def test_new_population_excludes_invalid_for_modeling_burned_tsg_cell(tmp_path):
    """One burned, TSG-True cell has valid_for_modeling=False: it stays in the
    primary and TSG populations and is dropped only by the new population."""
    frame = make_frame(
        rows=[0, 0, 5],
        cols=[0, 1, 5],
        burned=[1, 1, 1],
        burnable=[True, True, True],
        valid_for_modeling=[True, True, False],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", tmp_path / "d" / "experiments"), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", tmp_path / "d" / "comparison"):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    populations = result["populations"]
    assert populations[bpa.POPULATION_ALL_VALID_BURNED]["metrics"]["total_burned_cells"] == 3
    assert populations[bpa.POPULATION_BURNABLE_TSG_BURNED]["metrics"]["total_burned_cells"] == 3
    assert populations[bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED]["metrics"]["total_burned_cells"] == 2


def test_new_population_also_requires_the_burnable_mask(tmp_path):
    """valid_for_modeling alone is not enough: the TSG mask still applies, and
    the primary population keeps every burned cell."""
    frame = make_frame(
        rows=[0, 5, 10],
        cols=[0, 5, 10],
        burned=[1, 1, 1],
        burnable=[True, False, True],
        valid_for_modeling=[True, True, False],
    )
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", tmp_path / "d" / "experiments"), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", tmp_path / "d" / "comparison"):
        result = bpa.analyze_experiment(FAKE_EXP_ALPHA, dry_run=False)

    populations = result["populations"]
    assert populations[bpa.POPULATION_ALL_VALID_BURNED]["metrics"]["total_burned_cells"] == 3
    assert populations[bpa.POPULATION_BURNABLE_TSG_BURNED]["metrics"]["total_burned_cells"] == 2
    assert populations[bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED]["metrics"]["total_burned_cells"] == 1


def test_missing_valid_for_modeling_column_fails_clearly():
    columns = [
        "burned", "row_500m", "col_500m", "elevation_mean", "landcover_dominant",
        "burnable_tree_shrub_grass",
    ]
    with pytest.raises(bpa.BurnedPatternAuditError, match="valid_for_modeling"):
        bpa.validate_required_columns(columns, FAKE_EXP_ALPHA)


# ---------------------------------------------------------------------------
# Component concentration metrics
# ---------------------------------------------------------------------------
def test_component_concentration_metrics_for_sizes_3_1_1():
    components = [{"size": 3}, {"size": 1}, {"size": 1}]
    metrics = bpa.component_population_metrics(components, burned_cell_count=5)

    assert metrics["second_largest_component_cells"] == 1
    assert metrics["second_largest_component_fraction"] == pytest.approx(1 / 5)
    assert metrics["component_share_hhi"] == pytest.approx((3 / 5) ** 2 + (1 / 5) ** 2 + (1 / 5) ** 2)
    assert metrics["component_share_hhi"] == pytest.approx(0.44)
    assert metrics["effective_component_count"] == pytest.approx(1 / 0.44)
    # Size-sum invariant is untouched by the new fields.
    assert sum(c["size"] for c in components) == metrics["total_burned_cells"]


def test_component_concentration_metrics_single_component():
    metrics = bpa.component_population_metrics([{"size": 7}], burned_cell_count=7)
    assert metrics["second_largest_component_cells"] == 0
    assert metrics["second_largest_component_fraction"] == 0.0
    assert metrics["component_share_hhi"] == pytest.approx(1.0)
    assert metrics["effective_component_count"] == pytest.approx(1.0)


def test_component_concentration_metrics_zero_components():
    metrics = bpa.component_population_metrics([], burned_cell_count=0)
    assert metrics["second_largest_component_fraction"] is None
    assert metrics["component_share_hhi"] is None
    assert metrics["effective_component_count"] is None
    assert metrics["second_largest_component_cells"] == 0
    assert metrics["component_count"] == 0


# ---------------------------------------------------------------------------
# Comparison outputs carry the new population, the new metrics and the
# sensitivity-permission provenance.
# ---------------------------------------------------------------------------
def test_comparison_outputs_record_new_population_metrics_and_permission(tmp_path):
    paths = _legacy_pair_fixtures(tmp_path)
    comparison_dir = tmp_path / "d" / "comparison"
    get_exp, get_path = _patch_registry(_synthetic_registry(), paths)

    with get_exp, get_path, \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", tmp_path / "d" / "experiments"), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", comparison_dir):
        result = bpa.run_analysis(
            experiments=[FAKE_EXP_LEGACY, FAKE_EXP_SUCCESSOR],
            dry_run=False,
            allow_superseded_sensitivity=True,
        )

    assert result["selection_mode"] == "explicit_superseded_sensitivity"
    assert result["allow_superseded_sensitivity"] is True
    assert result["superseded_pairs"] == [{
        "superseded_experiment_id": FAKE_EXP_LEGACY,
        "canonical_successor_experiment_id": FAKE_EXP_SUCCESSOR,
    }]
    assert "NOT restored to the canonical cohort" in result["superseded_sensitivity_note"]

    resolved_comparison = Path(result["comparison"]["output_dir"])
    expected_scope = "__".join(sorted(result["resolved_experiment_ids"]))
    assert resolved_comparison == comparison_dir.parent / expected_scope / "comparison"
    assert str(resolved_comparison).startswith(str(tmp_path))  # never the real root
    comparison_csv = pd.read_csv(resolved_comparison / "multi_aoi_burned_pattern_comparison.csv")
    assert set(comparison_csv["population"]) == set(bpa.POPULATIONS)
    for column in (
        "second_largest_component_cells", "second_largest_component_fraction",
        "component_share_hhi", "effective_component_count",
    ):
        assert column in comparison_csv.columns

    manifest = json.loads((resolved_comparison / "manifest.json").read_text())
    assert manifest["selection_mode"] == "explicit_superseded_sensitivity"
    assert manifest["allow_superseded_sensitivity"] is True
    assert manifest["superseded_pairs"] == [{
        "superseded_experiment_id": FAKE_EXP_LEGACY,
        "canonical_successor_experiment_id": FAKE_EXP_SUCCESSOR,
    }]
    assert "NOT restored to the canonical cohort" in manifest["superseded_sensitivity_note"]

    comparison_md = (resolved_comparison / "multi_aoi_burned_pattern_comparison.md").read_text()
    assert "allow_superseded_sensitivity: True" in comparison_md
    assert "AOI-definition sensitivity comparator" in comparison_md
    assert bpa.POPULATION_BURNABLE_TSG_VALID_MODEL_BURNED in comparison_md
    assert "Component-share HHI" in comparison_md
    assert "Effective component count" in comparison_md


def test_default_run_records_no_permission_and_no_pairs(tmp_path):
    frame = make_frame([0, 1], [0, 1], [1, 1])
    path = write_fixture(tmp_path, FAKE_EXP_ALPHA, frame)

    with patch.object(bpa, "canonical_step8a_path", return_value=path), \
         patch.object(bpa, "get_experiment", return_value={"experiment_id": FAKE_EXP_ALPHA, "enabled": True, "variant_status": "canonical"}), \
         patch.object(bpa, "EXPERIMENTS_OUTPUT_ROOT", tmp_path / "d" / "experiments"), \
         patch.object(bpa, "COMPARISON_OUTPUT_DIR", tmp_path / "d" / "comparison"):
        result = bpa.run_analysis(experiments=[FAKE_EXP_ALPHA], dry_run=True)

    assert result["allow_superseded_sensitivity"] is False
    assert result["superseded_pairs"] == []
    assert result["superseded_sensitivity_note"] is None
    assert result["selection_mode"] == "explicit"
