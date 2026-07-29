"""Tests for the generic, directed, label-blind marginal Area-of-Applicability
analysis (src/marginal_area_of_applicability.py).

Everything runs against a fully synthetic Step8A tree under tmp_path, injected
through the module's public `experiments_root` / `output_root` parameters --
never by monkeypatching another module's PROJECT_ROOT. Experiment IDs are
taken dynamically from the registry so no AOI name is hard-coded here either.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.marginal_area_of_applicability as aoa
from core.pipeline_orchestrator import LEGACY_EXPERIMENT_ID
from core.regions import get_experiment, list_experiments


# =============================================================================
# Registry-driven, non-hard-coded experiment IDs
# =============================================================================
REGISTRY_IDS = tuple(
    sorted(e for e in list_experiments(include_disabled=False) if e != LEGACY_EXPERIMENT_ID)
)


def experiment_ids(n: int) -> list[str]:
    if len(REGISTRY_IDS) < n:
        pytest.skip(f"registry has fewer than {n} enabled experiments")
    return list(REGISTRY_IDS[:n])


def step8a_dir(experiments_root: Path, experiment_id: str) -> Path:
    namespace = get_experiment(experiment_id)["output_namespace"]
    return Path(experiments_root) / namespace / "step8a"


# =============================================================================
# Synthetic Step8A fixture
# =============================================================================
def synthetic_frame(
    n: int = 12, *, seed: int = 0, offset: float = 0.0, scale: float = 1.0,
    landcover: list | None = None, burned: list | None = None,
    eligible: list | None = None, burnable: list | None = None,
    overrides: dict[str, list] | None = None, duplicate_grid: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = list(range(n))
    cols = [0] * n
    if duplicate_grid:
        rows[-1] = rows[0]
    frame = pd.DataFrame({"row_500m": rows, "col_500m": cols})
    for i, feature in enumerate(aoa.numeric_features()):
        frame[feature] = offset + scale * rng.normal(loc=i, scale=1.0, size=n)
    frame[aoa.categorical_features()[0]] = (
        landcover if landcover is not None else [10, 20] * (n // 2) + [10] * (n % 2)
    )
    # `burned` is deliberately present in the fixture: the analysis must never
    # read it, and the tests below prove that.
    frame["burned"] = burned if burned is not None else [0, 1] * (n // 2) + [0] * (n % 2)
    frame["analysis_eligible"] = eligible if eligible is not None else [True] * n
    frame[aoa.BURNABLE_MASK_COLUMN] = burnable if burnable is not None else [True] * n
    frame["pre_label_burn_excluded"] = [not e for e in frame["analysis_eligible"]]
    for column, values in (overrides or {}).items():
        frame[column] = values
    return frame


def write_experiment(experiments_root: Path, experiment_id: str, frame: pd.DataFrame) -> Path:
    directory = step8a_dir(experiments_root, experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "step8a_dataset_stats.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "step": "step8a_prepare_500m_modeling_dataset",
    }))
    path = directory / "step8a_500m_modeling_dataset.parquet"
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def two_experiments(tmp_path):
    root = tmp_path / "experiments"
    ids = experiment_ids(2)
    write_experiment(root, ids[0], synthetic_frame(seed=1, offset=0.0))
    write_experiment(root, ids[1], synthetic_frame(seed=2, offset=5.0))
    return root, ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =============================================================================
# 1-4. Directed pair generation and determinism
# =============================================================================
def test_three_experiments_yield_exactly_six_directed_pairs():
    pairs = aoa.ordered_pairs(experiment_ids(3))
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    assert all(s != t for s, t in pairs)


def test_four_experiments_yield_exactly_twelve_directed_pairs():
    pairs = aoa.ordered_pairs(experiment_ids(4))
    assert len(pairs) == 12
    assert len(set(pairs)) == 12


def test_selection_order_changes_neither_pairs_nor_analysis_id(tmp_path):
    ids = experiment_ids(3)
    forward = aoa.ordered_pairs(ids)
    reversed_order = aoa.ordered_pairs(list(reversed(ids)))
    shuffled = aoa.ordered_pairs([ids[1], ids[2], ids[0]])
    assert forward == reversed_order == shuffled

    config_a = aoa.scientific_configuration(ids[0], ids[1], "a" * 64, "b" * 64)
    config_b = aoa.scientific_configuration(ids[0], ids[1], "a" * 64, "b" * 64)
    assert aoa.compute_analysis_id(config_a) == aoa.compute_analysis_id(config_b)


def test_selection_order_changes_no_analysis_id_end_to_end(tmp_path):
    """A full run must be identical whichever order the caller lists the
    experiments in -- pair identities and the comparison identity included."""
    ids = experiment_ids(3)
    root = tmp_path / "experiments"
    for index, experiment_id in enumerate(ids):
        write_experiment(root, experiment_id, synthetic_frame(seed=index + 1, offset=index * 3.0))

    forward = aoa.run_analysis(
        experiments=list(ids), output_root=tmp_path / "out_a", experiments_root=root,
    )
    reverse = aoa.run_analysis(
        experiments=list(reversed(ids)), output_root=tmp_path / "out_b", experiments_root=root,
    )
    assert forward["pair_analysis_ids"] == reverse["pair_analysis_ids"]
    assert forward["comparison_analysis_id"] == reverse["comparison_analysis_id"]
    assert forward["directed_pair_count"] == reverse["directed_pair_count"] == 6

    for name in ("multi_aoi_marginal_aoa_comparison.csv", "multi_aoi_marginal_aoa_comparison.md"):
        assert _sha256(tmp_path / "out_a" / "comparison" / name) == \
               _sha256(tmp_path / "out_b" / "comparison" / name)


def test_forward_and_reverse_are_distinct_analyses(two_experiments, tmp_path):
    root, ids = two_experiments
    source, target = ids
    result = aoa.run_analysis(
        experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
    )
    forward = aoa.pair_token(source, target)
    reverse = aoa.pair_token(target, source)
    assert forward != reverse
    assert result["pair_analysis_ids"][forward] != result["pair_analysis_ids"][reverse]
    # Both directions exist on disk, each in its own namespace.
    assert (tmp_path / "out" / "pairs" / forward / "marginal_aoa_summary.json").is_file()
    assert (tmp_path / "out" / "pairs" / reverse / "marginal_aoa_summary.json").is_file()

    forward_summary = json.loads(
        (tmp_path / "out" / "pairs" / forward / "marginal_aoa_summary.json").read_text()
    )
    assert forward_summary["source_experiment_id"] == source
    assert forward_summary["target_experiment_id"] == target


def test_pair_token_is_never_sorted():
    later, earlier = "zzz_experiment", "aaa_experiment"
    assert aoa.pair_token(later, earlier) == f"{later}__{earlier}"


# =============================================================================
# 5-10. Numeric marginal support
# =============================================================================
def test_source_bounds_are_inclusive():
    source = pd.Series([1.0, 5.0])
    target = pd.Series([1.0, 5.0, 3.0])  # both endpoints and an interior value
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["source_min"] == 1.0 and row["source_max"] == 5.0
    assert row["target_n_in_source_range"] == 3
    assert row["fraction_outside_source_range"] == 0.0


def test_below_and_above_counted_separately():
    source = pd.Series([0.0, 10.0])
    target = pd.Series([-5.0, -1.0, 5.0, 11.0])
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["target_n_below_source_min"] == 2
    assert row["target_n_above_source_max"] == 1
    assert row["target_n_in_source_range"] == 1
    assert row["fraction_below_source_min"] == 0.5
    assert row["fraction_above_source_max"] == 0.25
    assert row["mean_absolute_exceedance"] == pytest.approx((5.0 + 1.0 + 0.0 + 1.0) / 4)
    assert row["max_absolute_exceedance"] == 5.0


def test_fraction_denominator_is_finite_target_count():
    source = pd.Series([0.0, 1.0])
    target = pd.Series([2.0, np.nan, np.nan])  # 1 finite, 2 missing
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["target_n_total"] == 3
    assert row["target_n_finite"] == 1
    # Denominator is the FINITE count, not the total.
    assert row["fraction_outside_source_range"] == 1.0
    # ...while the missing fraction uses the total.
    assert row["fraction_target_missing"] == pytest.approx(2 / 3)


def test_target_missing_is_not_outside_and_is_reported():
    source = pd.Series([0.0, 1.0])
    target = pd.Series([0.5, np.nan])
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["target_n_missing"] == 1
    assert row["target_n_below_source_min"] == 0
    assert row["target_n_above_source_max"] == 0
    assert row["fraction_outside_source_range"] == 0.0


def test_all_missing_source_is_support_unavailable():
    source = pd.Series([np.nan, np.nan])
    target = pd.Series([0.0, 100.0])
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["support_status"] == aoa.SUPPORT_STATUS_UNAVAILABLE
    assert row["source_min"] is None and row["source_max"] is None
    # Nothing may be declared outside a support that does not exist.
    assert row["target_n_below_source_min"] == 0
    assert row["target_n_above_source_max"] == 0
    assert row["fraction_outside_source_range"] is None
    assert row["target_n_not_assessable"] == 2


def test_constant_source_range_behaves_correctly():
    source = pd.Series([3.0, 3.0, 3.0])
    target = pd.Series([3.0, 3.0, 2.999, 3.001])
    row = aoa.numeric_feature_support(source, target, "f")
    assert row["source_range_width"] == 0.0
    assert row["target_n_in_source_range"] == 2
    assert row["target_n_below_source_min"] == 1
    assert row["target_n_above_source_max"] == 1
    assert row["fraction_outside_source_range"] == 0.5


def test_scientific_regression_numeric_support():
    """target [0, 1, 2] against source support [0, 1]."""
    row = aoa.numeric_feature_support(pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0, 2.0]), "f")
    assert row["target_n_in_source_range"] == 2
    assert row["target_n_above_source_max"] == 1
    assert row["target_n_below_source_min"] == 0
    assert row["fraction_outside_source_range"] == pytest.approx(1 / 3)


# =============================================================================
# 11-12. Categorical marginal support
# =============================================================================
def test_categorical_unseen_levels_detected():
    row = aoa.categorical_feature_support(
        pd.Series([10, 20, 10]), pd.Series([20, 30, 40]), "landcover",
    )
    assert row["source_observed_levels"] == ["10", "20"]
    assert row["target_unseen_levels"] == ["30", "40"]
    assert row["target_n_unseen_level"] == 2
    assert row["fraction_target_unseen_level"] == pytest.approx(2 / 3)


def test_categorical_missing_is_not_unseen():
    row = aoa.categorical_feature_support(
        pd.Series([10, 20]), pd.Series([20, None, np.nan]), "landcover",
    )
    assert row["target_n_missing"] == 2
    assert row["target_unseen_levels"] == []
    assert row["target_n_unseen_level"] == 0
    assert row["fraction_target_unseen_level"] == 0.0


def test_categorical_regression_unseen_and_missing():
    """source levels {10, 20}; target {20, 30, missing}."""
    row = aoa.categorical_feature_support(
        pd.Series([10, 20]), pd.Series([20, 30, None]), "landcover",
    )
    assert row["target_unseen_levels"] == ["30"]
    assert row["target_n_unseen_level"] == 1
    assert row["fraction_target_unseen_level"] == pytest.approx(1 / 2)
    assert row["target_n_missing"] == 1


def test_categorical_support_unavailable_when_source_all_missing():
    row = aoa.categorical_feature_support(
        pd.Series([None, np.nan]), pd.Series([10, 20]), "landcover",
    )
    assert row["support_status"] == aoa.SUPPORT_STATUS_UNAVAILABLE
    assert row["target_unseen_levels"] == []
    assert row["target_n_unseen_level"] == 0
    assert row["fraction_target_unseen_level"] is None


def test_numeric_looking_levels_have_one_canonical_form():
    assert aoa.canonical_level(10) == aoa.canonical_level(10.0) == aoa.canonical_level("10") == "10"
    row = aoa.categorical_feature_support(
        pd.Series([10, 10.0, "10"]), pd.Series([10.0]), "landcover",
    )
    assert row["source_observed_levels"] == ["10"]
    assert row["target_n_unseen_level"] == 0


# =============================================================================
# 13. Cell-status precedence
# =============================================================================
def _cell_status(target_frame: pd.DataFrame, source_frame: pd.DataFrame) -> list[str]:
    numeric_rows = [
        aoa.numeric_feature_support(source_frame[f], target_frame[f], f)
        for f in aoa.numeric_features()
    ]
    categorical_rows = [
        aoa.categorical_feature_support(source_frame[f], target_frame[f], f)
        for f in aoa.categorical_features()
    ]
    table = aoa.build_target_cell_table("s", "t", target_frame, numeric_rows, categorical_rows)
    return table["cell_support_status"].tolist()


def test_cell_status_precedence_inside_outside_not_assessable():
    source = synthetic_frame(n=4, seed=3)
    target = synthetic_frame(n=4, seed=3)  # identical -> everything inside

    first_numeric = aoa.numeric_features()[0]
    # Row 1: pushed far outside the source range.
    target.loc[1, first_numeric] = source[first_numeric].max() + 1000.0
    # Row 2: missing only -> not assessable.
    target.loc[2, first_numeric] = np.nan
    # Row 3: outside AND missing -> outside wins.
    target.loc[3, first_numeric] = source[first_numeric].max() + 1000.0
    target.loc[3, aoa.numeric_features()[1]] = np.nan

    statuses = _cell_status(target, source)
    assert statuses[0] == aoa.CELL_STATUS_INSIDE
    assert statuses[1] == aoa.CELL_STATUS_OUTSIDE
    assert statuses[2] == aoa.CELL_STATUS_NOT_ASSESSABLE
    assert statuses[3] == aoa.CELL_STATUS_OUTSIDE


def test_outside_plus_missing_keeps_the_missing_count():
    source = synthetic_frame(n=2, seed=4)
    target = synthetic_frame(n=2, seed=4)
    first, second = aoa.numeric_features()[0], aoa.numeric_features()[1]
    target.loc[1, first] = source[first].max() + 500.0
    target.loc[1, second] = np.nan

    numeric_rows = [
        aoa.numeric_feature_support(source[f], target[f], f) for f in aoa.numeric_features()
    ]
    categorical_rows = [
        aoa.categorical_feature_support(source[f], target[f], f)
        for f in aoa.categorical_features()
    ]
    table = aoa.build_target_cell_table("s", "t", target, numeric_rows, categorical_rows)
    row = table.iloc[1]
    assert row["cell_support_status"] == aoa.CELL_STATUS_OUTSIDE
    assert row["features_missing_count"] == 1
    assert row["total_features_outside_count"] == 1


def test_source_support_unavailable_makes_cells_not_assessable():
    source = synthetic_frame(n=3, seed=5)
    target = synthetic_frame(n=3, seed=5)
    source[aoa.numeric_features()[0]] = np.nan  # no source support for this feature
    statuses = _cell_status(target, source)
    assert set(statuses) == {aoa.CELL_STATUS_NOT_ASSESSABLE}


# =============================================================================
# 14-16. Label firewall
# =============================================================================
def test_burned_never_enters_the_parquet_column_list(two_experiments):
    root, ids = two_experiments
    captured: list[list[str]] = []

    def spy(path, columns=None):
        captured.append(list(columns))
        return pd.read_parquet(path, columns=columns)

    path = aoa.resolve_dataset_path(ids[0], root)
    aoa.load_population(path, ids[0], read_parquet=spy)

    assert captured, "read_parquet was not called"
    for columns in captured:
        assert "burned" not in columns
        assert "burn_date" not in columns
        assert "burn_month" not in columns
        assert "cell_id" not in columns
        assert set(aoa.all_features()).issubset(columns)


def test_full_run_never_requests_a_label_column(two_experiments, tmp_path):
    root, ids = two_experiments
    real_read_parquet = pd.read_parquet
    captured: list[list[str]] = []

    def spy(path, *args, **kwargs):
        columns = kwargs.get("columns")
        if columns is not None:
            captured.append(list(columns))
        return real_read_parquet(path, *args, **kwargs)

    with patch.object(aoa.pd, "read_parquet", side_effect=spy):
        aoa.run_analysis(
            experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
        )

    assert captured
    forbidden_labels = {"burned", "burn_date", "burn_month", "burn_day_of_year", "label_source"}
    for columns in captured:
        assert not (forbidden_labels & set(columns))


def test_changing_burned_values_cannot_change_the_output(tmp_path):
    ids = experiment_ids(2)

    def build(root: Path, burned_flip: bool) -> None:
        for index, experiment_id in enumerate(ids):
            frame = synthetic_frame(seed=index + 1, offset=index * 5.0)
            if burned_flip:
                frame["burned"] = 1 - frame["burned"]
            write_experiment(root, experiment_id, frame)

    root_a, root_b = tmp_path / "exp_a", tmp_path / "exp_b"
    out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
    build(root_a, burned_flip=False)
    build(root_b, burned_flip=True)

    result_a = aoa.run_analysis(experiments=list(ids), output_root=out_a, experiments_root=root_a)
    result_b = aoa.run_analysis(experiments=list(ids), output_root=out_b, experiments_root=root_b)

    assert result_a["pair_analysis_ids"].keys() == result_b["pair_analysis_ids"].keys()

    # Every SCIENTIFIC output is byte-identical.
    #
    # `analysis_id` is deliberately excluded: by contract it hashes the source
    # and target Step8A SHA-256, so rewriting the label column necessarily
    # moves it. That is provenance tracking the input FILE, not the analysis
    # reading the label -- which is exactly the distinction under test, and
    # why the scientific payload below must not move at all.
    scientific_files = [
        name for name in aoa.PAIR_OUTPUT_FILENAMES
        if name not in ("manifest.json", "marginal_aoa_summary.json", "marginal_aoa_report.md")
    ]
    for token in result_a["output_paths"]:
        for name in scientific_files:
            assert _sha256(out_a / "pairs" / token / name) == \
                   _sha256(out_b / "pairs" / token / name), f"{token}/{name} differs"

        summary_a = json.loads((out_a / "pairs" / token / "marginal_aoa_summary.json").read_text())
        summary_b = json.loads((out_b / "pairs" / token / "marginal_aoa_summary.json").read_text())
        del summary_a["analysis_id"], summary_b["analysis_id"]
        assert summary_a == summary_b, f"{token} summary differs beyond its analysis_id"

        report_a = (out_a / "pairs" / token / "marginal_aoa_report.md").read_text()
        report_b = (out_b / "pairs" / token / "marginal_aoa_report.md").read_text()
        strip = lambda text: [ln for ln in text.splitlines() if "analysis_id" not in ln]
        assert strip(report_a) == strip(report_b), f"{token} report differs beyond its analysis_id"


def test_forbidden_column_in_the_feature_contract_fails_fast(monkeypatch):
    monkeypatch.setattr(
        aoa, "SHARED_THERMAL_MODEL_FEATURES", list(aoa.all_features()) + ["burned"],
    )
    with pytest.raises(aoa.MarginalAoAError, match="Forbidden column"):
        aoa.validate_feature_contract()


def test_no_output_contains_a_label_column(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    result = aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    forbidden = ("burned", "burn_date", "burn_month", "label_source")

    for token in result["output_paths"]:
        cells = pd.read_parquet(out / "pairs" / token / "marginal_aoa_target_cells.parquet")
        assert not (set(forbidden) & set(cells.columns))
        for name in ("marginal_aoa_numeric_features.csv", "marginal_aoa_categorical_features.csv"):
            frame = pd.read_csv(out / "pairs" / token / name)
            assert not (set(forbidden) & set(frame.columns))
        summary = json.loads((out / "pairs" / token / "marginal_aoa_summary.json").read_text())
        assert summary["label_firewall"]["label_columns_loaded"] == []
        assert summary["label_firewall"]["burned_column_read"] is False

    comparison = pd.read_csv(out / "comparison" / "multi_aoi_marginal_aoa_comparison.csv")
    assert not (set(forbidden) & set(comparison.columns))


# =============================================================================
# 17-19. Population contract
# =============================================================================
def test_analysis_eligible_false_rows_are_excluded():
    frame = synthetic_frame(n=6, seed=6, eligible=[True, True, False, True, False, True])
    population = aoa.resolve_population(frame, "x")
    assert len(population) == 4


def test_non_primary_population_rows_are_excluded():
    frame = synthetic_frame(n=6, seed=7, burnable=[True, False, True, False, True, True])
    population = aoa.resolve_population(frame, "x")
    assert len(population) == 4


def test_duplicate_grid_cell_fails_fast():
    frame = synthetic_frame(n=6, seed=8, duplicate_grid=True)
    with pytest.raises(aoa.MarginalAoAError, match="duplicate"):
        aoa.resolve_population(frame, "x")


def test_empty_population_fails_fast():
    frame = synthetic_frame(n=4, seed=9, burnable=[False] * 4)
    with pytest.raises(aoa.MarginalAoAError, match="empty"):
        aoa.resolve_population(frame, "x")


def test_missing_analysis_eligible_column_means_every_row_eligible():
    frame = synthetic_frame(n=4, seed=10).drop(columns=["analysis_eligible"])
    assert len(aoa.resolve_population(frame, "x")) == 4


# =============================================================================
# 20-21. Dry run and missing input
# =============================================================================
def test_dry_run_writes_nothing(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    result = aoa.run_analysis(
        experiments=list(ids), dry_run=True, output_root=out, experiments_root=root,
    )
    assert result["ran"] is False
    assert result["dry_run"] is True
    assert result["files_written"] is False
    assert result["model_fit"] is False
    assert result["prediction_run"] is False
    assert result["bootstrap_run"] is False
    assert result["label_firewall"]["label_columns_loaded"] == []
    assert result["directed_pair_count"] == 2
    assert result["planned_output_paths"]
    assert not out.exists()


def test_missing_step8a_raises_clearly(tmp_path):
    ids = experiment_ids(2)
    root = tmp_path / "experiments"
    write_experiment(root, ids[0], synthetic_frame(seed=1))
    # ids[1] deliberately absent
    with pytest.raises(aoa.MarginalAoAError, match="Missing canonical Step8A dataset"):
        aoa.run_analysis(
            experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
        )


# =============================================================================
# 22-24. Provenance, force guard, determinism
# =============================================================================
def test_input_hashes_are_identical_before_and_after(two_experiments, tmp_path):
    root, ids = two_experiments
    before = {i: _sha256(aoa.resolve_dataset_path(i, root)) for i in ids}
    result = aoa.run_analysis(
        experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
    )
    after = {i: _sha256(aoa.resolve_dataset_path(i, root)) for i in ids}
    assert before == after
    assert result["input_sha256_before"] == result["input_sha256_after"] == before


def test_different_analysis_id_cannot_overwrite_without_force(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)

    # A changed source dataset changes the analysis identity.
    write_experiment(root, ids[0], synthetic_frame(seed=99, offset=1.0))
    with pytest.raises(aoa.MarginalAoAError, match="DIFFERENT analysis_id"):
        aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)

    result = aoa.run_analysis(
        experiments=list(ids), output_root=out, experiments_root=root, force=True,
    )
    assert result["ran"] is True


def test_identical_rerun_is_deterministic(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    first = aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    token = next(iter(first["output_paths"]))
    hashes = {
        name: _sha256(out / "pairs" / token / name)
        for name in aoa.PAIR_OUTPUT_FILENAMES if name != "manifest.json"
    }
    second = aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    assert second["pair_analysis_ids"] == first["pair_analysis_ids"]
    for name, digest in hashes.items():
        assert _sha256(out / "pairs" / token / name) == digest


# =============================================================================
# 25-26, 33. Outputs
# =============================================================================
def test_every_expected_pair_output_file_is_written(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    result = aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    assert set(result["output_paths"]) == {
        aoa.pair_token(ids[0], ids[1]), aoa.pair_token(ids[1], ids[0]),
    }
    for token in result["output_paths"]:
        for name in aoa.PAIR_OUTPUT_FILENAMES:
            assert (out / "pairs" / token / name).is_file(), f"{token}/{name} missing"
    for name in aoa.COMPARISON_OUTPUT_FILENAMES:
        assert (out / "comparison" / name).is_file(), f"comparison/{name} missing"


def test_comparison_row_count_equals_directed_pair_count(tmp_path):
    ids = experiment_ids(3)
    root = tmp_path / "experiments"
    for index, experiment_id in enumerate(ids):
        write_experiment(root, experiment_id, synthetic_frame(seed=index + 1, offset=index * 3.0))
    out = tmp_path / "out"
    result = aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)

    assert result["directed_pair_count"] == 6
    comparison = pd.read_csv(out / "comparison" / "multi_aoi_marginal_aoa_comparison.csv")
    assert len(comparison) == 6
    payload = json.loads((out / "comparison" / "multi_aoi_marginal_aoa_comparison.json").read_text())
    assert payload["directed_pair_count"] == 6
    assert len(payload["rows"]) == 6


def test_output_ordering_is_deterministic(tmp_path):
    ids = experiment_ids(3)
    root = tmp_path / "experiments"
    for index, experiment_id in enumerate(ids):
        write_experiment(root, experiment_id, synthetic_frame(seed=index + 1, offset=index * 3.0))
    out = tmp_path / "out"
    aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)

    comparison = pd.read_csv(out / "comparison" / "multi_aoi_marginal_aoa_comparison.csv")
    keys = list(zip(comparison["source_experiment_id"], comparison["target_experiment_id"]))
    assert keys == sorted(keys)

    token = aoa.pair_token(ids[0], ids[1])
    numeric = pd.read_csv(out / "pairs" / token / "marginal_aoa_numeric_features.csv")
    assert numeric["feature"].tolist() == list(aoa.numeric_features())
    cells = pd.read_parquet(out / "pairs" / token / "marginal_aoa_target_cells.parquet")
    grid = list(zip(cells["row_500m"], cells["col_500m"]))
    assert grid == sorted(grid)


def test_top_outside_features_sorted_by_fraction_then_name(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    token = aoa.pair_token(ids[0], ids[1])
    summary = json.loads((out / "pairs" / token / "marginal_aoa_summary.json").read_text())
    entries = summary["top_outside_support_features"]
    assert entries == sorted(entries, key=lambda e: (-e["fraction_outside"], e["feature"]))


# =============================================================================
# 31-32, 34-35. Isolation guards
# =============================================================================
def test_no_real_experiment_id_is_hard_coded_in_the_implementation():
    for module_path in (
        _PROJECT_ROOT / "src" / "marginal_area_of_applicability.py",
        _PROJECT_ROOT / "scripts" / "run_marginal_area_of_applicability.py",
    ):
        source = module_path.read_text(encoding="utf-8")
        for experiment_id in REGISTRY_IDS:
            assert experiment_id not in source, f"{experiment_id} hard-coded in {module_path.name}"


def test_run_touches_no_other_namespace(two_experiments, tmp_path):
    root, ids = two_experiments
    sentinels = {}
    for relative in (
        "experiments/sentinel_step8a.parquet",
        "cross_region/sentinel_step9.json",
        "cross_region/sentinel_step10.json",
        "diagnostics/other_namespace/sentinel.json",
    ):
        path = tmp_path / "frozen" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"frozen": True}))
        sentinels[path] = _sha256(path)

    step8a_before = {i: _sha256(aoa.resolve_dataset_path(i, root)) for i in ids}
    aoa.run_analysis(experiments=list(ids), output_root=tmp_path / "out", experiments_root=root)

    for path, digest in sentinels.items():
        assert _sha256(path) == digest, f"sentinel modified: {path}"
    assert {i: _sha256(aoa.resolve_dataset_path(i, root)) for i in ids} == step8a_before


def test_analysis_never_fits_a_model(two_experiments, tmp_path):
    root, ids = two_experiments

    def _boom(*_args, **_kwargs):
        raise AssertionError("marginal AoA must never fit a model")

    from sklearn.base import BaseEstimator

    with patch.object(BaseEstimator, "fit", _boom, create=True):
        result = aoa.run_analysis(
            experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
        )
    assert result["model_fit"] is False
    assert result["prediction_run"] is False


def test_analysis_never_samples_a_bootstrap(two_experiments, tmp_path):
    root, ids = two_experiments

    def _boom(*_args, **_kwargs):
        raise AssertionError("marginal AoA must never draw random samples")

    with patch.object(np.random, "default_rng", _boom):
        result = aoa.run_analysis(
            experiments=list(ids), output_root=tmp_path / "out", experiments_root=root,
        )
    assert result["bootstrap_run"] is False


def test_implementation_imports_no_model_or_bootstrap_machinery():
    source = (_PROJECT_ROOT / "src" / "marginal_area_of_applicability.py").read_text(encoding="utf-8")
    for banned in ("import sklearn", "from sklearn", "RandomForest", "roc_auc_score"):
        assert banned not in source, f"{banned!r} must not appear in the implementation"


# =============================================================================
# Markdown wording
# =============================================================================
def test_report_states_the_interpretation_boundary(two_experiments, tmp_path):
    root, ids = two_experiments
    out = tmp_path / "out"
    aoa.run_analysis(experiments=list(ids), output_root=out, experiments_root=root)
    report = (out / "pairs" / aoa.pair_token(*ids) / "marginal_aoa_report.md").read_text()
    lowered = report.lower()
    assert "each predictor separately" in lowered
    assert "multivariate joint support" in lowered
    assert "correlation structure" in lowered
    assert "does not guarantee transfer success" in lowered
    assert "no target label" in lowered or "uses no target label" in lowered
    assert "no statistical significance" in lowered
    for banned in ("statistically significant", "proves concept shift", "explains transfer failure"):
        assert banned not in lowered


def test_no_aoi_name_appears_in_the_limitations_text():
    joined = " ".join(aoa.LIMITATIONS)
    for experiment_id in REGISTRY_IDS:
        assert experiment_id not in joined
        display = get_experiment(experiment_id)["display_name"].split()[0]
        assert display not in joined
