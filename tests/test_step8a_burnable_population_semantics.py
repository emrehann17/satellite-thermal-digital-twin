"""Step8A burnable-count population semantics (report-layer only).

`burnable_tree_shrub_grass_count` was always accumulated over ALL grid rows,
including `valid_for_modeling == False`, while the neighbouring
`burnable_diagnostics_population` field said `"valid_for_modeling == True"`.
That field actually describes the landcover diagnostics block, so the two read
as if the counts were validity-filtered. The fix emits both populations under
unambiguous names and states which one downstream consumes.

These tests assert the report layer only. No label, predictor, population or
model may change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.paths import PROJECT_ROOT
import src.step8a_prepare_500m_modeling_dataset as prep

AOIS = ["manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021_extended"]
POP = "burnable_tree_shrub_grass"

CANONICAL = {
    "manavgat_2021": (20511, 784),
    "bejis_2022": (15190, 1100),
    "mugla_2021": (41730, 2911),
    "evia_2021_extended": (9298, 2664),
}


def _stats(aoi: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / "outputs" / "experiments" / aoi / "step8a" / "step8a_dataset_stats.json").read_text()
    )


def _frame(aoi: str) -> pd.DataFrame:
    return pd.read_parquet(
        PROJECT_ROOT / "outputs" / "experiments" / aoi / "step8a" / "step8a_500m_modeling_dataset.parquet"
    )


@pytest.mark.parametrize("aoi", AOIS)
def test_canonical_population_is_valid_filtered(aoi):
    """The canonical downstream population is mask AND valid_for_modeling."""
    df = _frame(aoi)
    mask = df[POP].astype(bool) & df["valid_for_modeling"].astype(bool)
    expected_total, expected_burned = CANONICAL[aoi]
    assert int(mask.sum()) == expected_total
    assert int(df.loc[mask, "burned"].sum()) == expected_burned


@pytest.mark.parametrize("aoi", AOIS)
def test_all_rows_count_is_at_least_valid_filtered_count(aoi):
    df = _frame(aoi)
    all_rows = int(df[POP].astype(bool).sum())
    valid_only = int((df[POP].astype(bool) & df["valid_for_modeling"].astype(bool)).sum())
    assert all_rows >= valid_only
    assert all_rows - valid_only == int(
        (df[POP].astype(bool) & ~df["valid_for_modeling"].astype(bool)).sum()
    )


@pytest.mark.parametrize("aoi", AOIS)
def test_legacy_field_still_counts_all_rows(aoi):
    """The legacy field must keep its historical meaning, unchanged."""
    stats = _stats(aoi)
    df = _frame(aoi)
    assert stats["burnable_tree_shrub_grass_count"] == int(df[POP].astype(bool).sum())


def _synthetic_frame() -> pd.DataFrame:
    """Both burnable masks, with a deliberate valid/invalid split per mask."""
    return pd.DataFrame({
        POP:                   [True, True,  True,  True,  False, False],
        "burnable_tree_shrub": [True, True,  False, False, False, True],
        "valid_for_modeling":  [True, False, True,  False, True,  True],
    })


def test_new_explicit_fields_separate_the_two_populations():
    """The emitted counts must distinguish all-rows from valid-filtered.

    Verified on a synthetic frame rather than on whichever AOI happens to
    have been regenerated: the semantics are a property of the report
    writer, and re-running Step8A over the real AOIs purely to satisfy a
    unit test would rewrite frozen modeling datasets.
    """
    fields = prep.burnable_counts_by_population_fields(_synthetic_frame())
    # burnable_tree_shrub_grass: 4 rows masked, 2 of them valid_for_modeling
    assert fields["burnable_tree_shrub_grass_count_all_rows"] == 4
    assert fields["burnable_tree_shrub_grass_count_valid_for_modeling"] == 2
    # burnable_tree_shrub: 3 rows masked, 2 of them valid_for_modeling
    assert fields["burnable_tree_shrub_count_all_rows"] == 3
    assert fields["burnable_tree_shrub_count_valid_for_modeling"] == 2
    # The valid-filtered population can never exceed the all-rows one.
    for column in prep.BURNABLE_MASK_COLUMNS:
        assert fields[f"{column}_count_valid_for_modeling"] <= fields[f"{column}_count_all_rows"]


def test_missing_frame_or_column_yields_none_not_zero():
    """None means 'not computable'; 0 would read as a genuinely empty mask."""
    assert prep.burnable_counts_by_population_fields(None) == {
        "burnable_tree_shrub_grass_count_all_rows": None,
        "burnable_tree_shrub_grass_count_valid_for_modeling": None,
        "burnable_tree_shrub_count_all_rows": None,
        "burnable_tree_shrub_count_valid_for_modeling": None,
    }
    partial = pd.DataFrame({POP: [True], "valid_for_modeling": [True]})
    fields = prep.burnable_counts_by_population_fields(partial)
    assert fields["burnable_tree_shrub_grass_count_all_rows"] == 1
    assert fields["burnable_tree_shrub_count_all_rows"] is None


def test_semantics_block_names_the_canonical_population():
    semantics = prep.BURNABLE_COUNT_POPULATION_SEMANTICS
    assert "valid_for_modeling == True" in semantics["canonical_downstream_population"]
    assert "LEGACY" in semantics["burnable_tree_shrub_grass_count"]
    # The legacy field must never be advertised as the modeling population.
    assert "Do NOT report it as the modeling population" in (
        semantics["burnable_tree_shrub_grass_count"]
    )


def test_new_explicit_fields_agree_with_any_regenerated_aoi():
    """Opportunistic: an AOI regenerated after the fix must be self-consistent.

    Skips rather than fails when no AOI carries the new fields yet -- their
    absence means Step8A has not been re-run since the report-layer change,
    which is a state of the outputs, not a defect in the code under test.
    """
    checked = 0
    for aoi in AOIS:
        stats = _stats(aoi)
        if "burnable_tree_shrub_grass_count_valid_for_modeling" not in stats:
            continue
        df = _frame(aoi)
        assert stats["burnable_tree_shrub_grass_count_all_rows"] == int(df[POP].astype(bool).sum())
        assert stats["burnable_tree_shrub_grass_count_valid_for_modeling"] == int(
            (df[POP].astype(bool) & df["valid_for_modeling"].astype(bool)).sum()
        )
        assert stats["burnable_tree_shrub_grass_count_valid_for_modeling"] == CANONICAL[aoi][0]
        semantics = stats["burnable_count_population_semantics"]
        assert "valid_for_modeling == True" in semantics["canonical_downstream_population"]
        assert "LEGACY" in semantics["burnable_tree_shrub_grass_count"]
        checked += 1
    if checked == 0:
        pytest.skip("no AOI has been regenerated since the report-layer fix")


def test_report_fix_does_not_change_models():
    """Step8B metrics must not reference the diagnostic count fields at all."""
    for aoi in AOIS:
        metrics = json.loads(
            (PROJECT_ROOT / "outputs" / "experiments" / aoi / "step8b"
             / "step8b_model_comparison_metrics.json").read_text()
        )
        blob = json.dumps(metrics)
        for field in (
            "burnable_tree_shrub_grass_count_all_rows",
            "burnable_count_population_semantics",
            "burnable_diagnostics_population",
        ):
            assert field not in blob, f"{aoi}: Step8B leaked diagnostic field {field}"


def test_step9_and_step10_consume_the_valid_filtered_population():
    """Frozen Step9B counts must equal the valid-filtered population."""
    root = PROJECT_ROOT / "outputs" / "cross_region"
    seen = {}
    for path in sorted(root.glob("*/step9b/cross_region_transfer_metrics.json")):
        payload = json.loads(path.read_text())
        for result in payload.get("results", []):
            if result.get("population") != POP:
                continue
            for role in ("source", "target"):
                eid = result.get(f"{role}_experiment_id")
                if eid in CANONICAL:
                    seen.setdefault(eid, set()).add(
                        (result[f"{role}_cell_count"], result[f"{role}_positive_count"])
                    )
    assert seen, "no frozen Step9B population counts found"
    for eid, values in seen.items():
        assert values == {CANONICAL[eid]}, f"{eid}: Step9B counts drifted -> {values}"
