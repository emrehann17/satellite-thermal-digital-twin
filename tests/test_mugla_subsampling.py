"""tests/test_mugla_subsampling.py

Focused tests for `src/mugla_subsampling.py` (`mugla_subsampling.v1`).

Every test uses SMALL SYNTHETIC fixtures and a `tmp_path`-injected
`output_root` / `experiments_root`. No test reads a canonical production
artifact, no test writes to a canonical path, no test fits a full-scale model
and no test contacts Earth Engine.

Run:
    PYTHONPATH="$PWD" python -m pytest -q tests/test_mugla_subsampling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.mugla_subsampling as mss
import scripts.validate_mugla_subsampling as vms
from src.mugla_subsampling import MuglaSubsamplingError

# The frozen analysis is bound to these three AOI ids; the fixtures below carry
# synthetic DATA under those names inside a tmp_path, never production data.
MUGLA = "mugla_2021"
MANAVGAT = "manavgat_2021"
BEJIS = "bejis_2022"
EXPERIMENTS = [MANAVGAT, BEJIS, MUGLA]

# Small enough to fit fast; large enough that 5 folds over 10-cell blocks are
# well posed and every stratum keeps both classes.
MUGLA_BLOCKS_PER_SIDE = 10      # 100 blocks -> 900 cells, 200 strata
COHORT_BLOCKS_PER_SIDE = 6      # 36 blocks -> 324 cells
CELLS_PER_BLOCK_SIDE = 3

TEST_SAMPLE_SIZE = 450          # exactly half of the synthetic Mugla population
TEST_REPEATS = 2


# =============================================================================
# Synthetic fixtures
# =============================================================================
def _make_frame(blocks_per_side: int, seed: int) -> pd.DataFrame:
    """A synthetic Step8A-shaped modeling dataset on a 10-cell block grid.

    Every block carries 3 positive and 6 negative cells, so every block yields
    both a label-0 and a label-1 stratum.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for block_row in range(blocks_per_side):
        for block_col in range(blocks_per_side):
            for r in range(CELLS_PER_BLOCK_SIDE):
                for c in range(CELLS_PER_BLOCK_SIDE):
                    burned = int((r + c) % 3 == 0)
                    signal = 1.4 if burned else 0.0
                    row_500m = block_row * mss.BLOCK_SIZE_CELLS + r
                    col_500m = block_col * mss.BLOCK_SIZE_CELLS + c
                    rows.append({
                        # Canonical Step8A identity: cell_id is r{row}_c{col}.
                        "cell_id": f"r{row_500m}_c{col_500m}",
                        "row_500m": row_500m,
                        "col_500m": col_500m,
                        "burned": burned,
                        "valid_for_modeling": True,
                        mss.POPULATION: True,
                        "ndvi_mean": float(rng.normal(0.4 - 0.1 * signal, 0.05)),
                        "elevation_mean": float(rng.normal(500 + 40 * signal, 25)),
                        "slope_mean": float(rng.normal(10 + signal, 2)),
                        "landcover_dominant": int(rng.choice([10, 20, 30])),
                        "lst_anomaly_mean": float(rng.normal(signal, 0.4)),
                        "current_lst_mean": float(rng.normal(305 + 2 * signal, 1.5)),
                        "current_tvdi_mean": float(rng.normal(0.5 + 0.1 * signal, 0.05)),
                        "tvdi_difference_mean": float(rng.normal(0.05 * signal, 0.03)),
                        "downscaled_lst_mean": float(rng.normal(304 + 2 * signal, 1.5)),
                        "fused_lst_mean": float(rng.normal(304.5 + 2 * signal, 1.5)),
                    })
    return pd.DataFrame(rows)


def _population(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = mss.assign_large_blocks(frame, mss.BLOCK_SIZE_CELLS)
    population = assigned[assigned["valid_for_modeling"] & assigned[mss.POPULATION]].copy()
    population["label"] = population["burned"].astype(int)
    return population.reset_index(drop=True)


def _pseudo_probabilities(population: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Deterministic, label-correlated probabilities standing in for a fit."""
    rng = np.random.default_rng(seed)
    labels = population["label"].to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}
    for offset, family in enumerate(mss.MODEL_FAMILIES):
        noise = rng.normal(0.0, 0.18 + 0.05 * offset, size=len(labels))
        out[family] = np.clip(0.25 + 0.4 * labels + noise, 0.001, 0.999)
    return out


def _write_fold_artifact(experiments_root: Path, mugla_population: pd.DataFrame) -> None:
    """The frozen full-Mugla 10-cell condition this analysis inherits folds from."""
    directory = (experiments_root / MUGLA / "robustness" / "step8_big_blocks"
                 / "block_10_cells")
    directory.mkdir(parents=True, exist_ok=True)

    blocks = sorted(mugla_population[mss.BLOCK_COLUMN].unique())
    block_fold = {block: index % mss.FOLD_COUNT for index, block in enumerate(blocks)}
    probabilities = _pseudo_probabilities(mugla_population, seed=7)

    frozen = pd.DataFrame({
        "experiment_id": MUGLA,
        "block_size_cells": mss.BLOCK_SIZE_CELLS,
        "population": mss.POPULATION,
        "cell_id": mugla_population["cell_id"].to_numpy(),
        "row_500m": mugla_population["row_500m"].to_numpy(),
        "col_500m": mugla_population["col_500m"].to_numpy(),
        # The frozen artifact spells the same partition differently on purpose.
        "spatial_block_id": [
            "block10_" + block.removeprefix("b10_r").replace("_c", "_")
            for block in mugla_population[mss.BLOCK_COLUMN]
        ],
        "fold_id": [block_fold[block] for block in mugla_population[mss.BLOCK_COLUMN]],
        "burned": mugla_population["label"].to_numpy(),
        "valid_for_evaluation": True,
    })
    for family in mss.MODEL_FAMILIES:
        frozen[f"{family}_probability"] = probabilities[family]
    frozen.to_parquet(directory / mss.WITHIN_REFERENCE_OOF_NAME, index=False)

    document = {
        "analysis_id": "synthetic",
        "experiment_id": MUGLA,
        "block_size_cells": mss.BLOCK_SIZE_CELLS,
        "nominal_scale": mss.BLOCK_NOMINAL_SCALE,
        "primary_population": mss.POPULATION,
        "n_splits_used": mss.FOLD_COUNT,
        "cv": {"class": "StratifiedGroupKFold", "n_splits": mss.FOLD_COUNT,
               "random_state": mss.FOLD_RANDOM_STATE},
    }
    y_true = frozen["burned"].to_numpy()
    for family in mss.MODEL_FAMILIES:
        metrics = mss._metric_from_predictions(
            y_true, frozen[f"{family}_probability"].to_numpy())
        document[f"{family}_roc_auc"] = metrics["roc_auc"]
        document[f"{family}_pr_auc"] = metrics["pr_auc"]
        document[f"{family}_brier"] = metrics["brier_score"]
    (directory / mss.WITHIN_REFERENCE_METRICS_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8")


def _write_transfer_artifact(output_root: Path, source_id: str, target_id: str,
                             target_population: pd.DataFrame, digests: dict[str, str],
                             seed: int) -> None:
    directory = output_root / "cross_region" / f"{source_id}__{target_id}" / "step9b"
    directory.mkdir(parents=True, exist_ok=True)
    direction = mss.direction_token(source_id, target_id)
    probabilities = _pseudo_probabilities(target_population, seed)

    predictions = pd.DataFrame({
        "transfer_direction": direction,
        "population": mss.POPULATION,
        "target_experiment_id": target_id,
        "target_cell_id": target_population["cell_id"].to_numpy(),
        "target_spatial_block_id": target_population[mss.BLOCK_COLUMN].to_numpy(),
        "burned": target_population["label"].to_numpy(),
    })
    for family in mss.MODEL_FAMILIES:
        predictions[f"{family}_probability"] = probabilities[family]
    predictions.to_parquet(directory / mss.TRANSFER_PREDICTIONS_NAME, index=False)

    y_true = predictions["burned"].to_numpy()
    result: dict = {
        "transfer_direction": direction,
        "population": mss.POPULATION,
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "skipped": False,
        "target_cell_count": int(len(predictions)),
        "target_positive_count": int(y_true.sum()),
    }
    for family in mss.MODEL_FAMILIES:
        result[f"{family}_metrics"] = mss._metric_from_predictions(
            y_true, predictions[f"{family}_probability"].to_numpy())
    document = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "source_only": True,
        "spatial_block_size_cells": mss.CANONICAL_SMALL_BLOCK_SIZE_CELLS,
        "resolved_inputs": {
            experiment_id: {"experiment_id": experiment_id, "dataset_sha256": digest}
            for experiment_id, digest in digests.items()
        },
        "results": [result],
    }
    (directory / mss.TRANSFER_METRICS_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {
        MUGLA: _make_frame(MUGLA_BLOCKS_PER_SIDE, seed=101),
        MANAVGAT: _make_frame(COHORT_BLOCKS_PER_SIDE, seed=202),
        BEJIS: _make_frame(COHORT_BLOCKS_PER_SIDE, seed=303),
    }


@pytest.fixture(scope="module")
def populations(frames) -> dict[str, pd.DataFrame]:
    return {key: _population(frame) for key, frame in frames.items()}


@pytest.fixture(scope="module")
def experiments_root(tmp_path_factory, frames, populations) -> Path:
    root = tmp_path_factory.mktemp("experiments")
    for experiment_id, frame in frames.items():
        directory = root / experiment_id / "step8a"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / "step8a_500m_modeling_dataset.parquet", index=False)
    _write_fold_artifact(root, populations[MUGLA])
    return root


@pytest.fixture(scope="module")
def digests(experiments_root) -> dict[str, str]:
    return {
        experiment_id: mss.sha256_file(
            experiments_root / experiment_id / "step8a"
            / "step8a_500m_modeling_dataset.parquet")
        for experiment_id in EXPERIMENTS
    }


@pytest.fixture(scope="module")
def output_root(tmp_path_factory, populations, digests) -> Path:
    root = tmp_path_factory.mktemp("outputs")
    for index, (source_id, target_id) in enumerate(mss.SOURCE_PAIRS + mss.TARGET_PAIRS):
        _write_transfer_artifact(root, source_id, target_id, populations[target_id],
                                 digests, seed=400 + index)
    return root


@pytest.fixture()
def frozen_contract(monkeypatch, digests):
    """Point the hash gate at the synthetic fixtures and shrink the grid."""
    monkeypatch.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
    monkeypatch.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
    monkeypatch.setattr(mss, "N_REPEATS", TEST_REPEATS)
    return digests


@pytest.fixture(scope="module")
def artifact(tmp_path_factory, experiments_root, output_root, digests):
    """One complete synthetic end-to-end run, shared read-only across tests."""
    run_root = tmp_path_factory.mktemp("run_outputs")
    # Reuse the module-scoped cross_region references inside the run's root.
    import shutil
    shutil.copytree(output_root / "cross_region", run_root / "cross_region")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
        patcher.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
        patcher.setattr(mss, "N_REPEATS", TEST_REPEATS)
        result = mss.run_analysis(
            experiments=EXPERIMENTS, output_root=run_root,
            experiments_root=experiments_root)
    return {"result": result, "root": Path(result["output_namespace"]),
            "output_root": run_root}


@pytest.fixture(scope="module")
def context(experiments_root, output_root, digests):
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
        patcher.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
        patcher.setattr(mss, "N_REPEATS", TEST_REPEATS)
        inventory = mss.build_frozen_input_inventory(EXPERIMENTS, experiments_root)
        return mss.build_context(EXPERIMENTS, inventory, experiments_root, output_root)


# =============================================================================
# Contract and identity
# =============================================================================
class TestContract:
    def test_schema_and_namespace(self):
        assert mss.SCHEMA_VERSION == "mugla_subsampling.v1"
        assert mss.DIAGNOSTIC_NAMESPACE == "mugla_subsampling"
        assert mss.DIAGNOSTIC_CLASS == "population_size_matched_subsampling_sensitivity"
        assert mss.STAGES == ("plan", "fit", "summarize")

    def test_frozen_scalars(self):
        assert mss.TARGET_SAMPLE_SIZE == 20511
        assert mss.N_REPEATS == 20
        assert mss.BLOCK_SIZE_CELLS == 10
        assert mss.FOLD_COUNT == 5
        assert mss.ESTIMATOR_SEED == 42

    def test_deterministic_analysis_id(self):
        config = mss.build_scientific_config(EXPERIMENTS)
        assert mss.compute_analysis_id(config) == mss.compute_analysis_id(config)
        assert mss.compute_analysis_id(config) == mss.compute_analysis_id(
            mss.build_scientific_config(EXPERIMENTS))

    def test_analysis_id_changes_with_the_contract(self, monkeypatch):
        before = mss.compute_analysis_id(mss.build_scientific_config(EXPERIMENTS))
        monkeypatch.setattr(mss, "N_REPEATS", 19)
        after = mss.compute_analysis_id(mss.build_scientific_config(EXPERIMENTS))
        assert before != after

    def test_stage_order_is_enforced(self):
        assert mss.validate_stage_range("plan", "fit") == ["plan", "fit"]
        with pytest.raises(MuglaSubsamplingError):
            mss.validate_stage_range("summarize", "plan")
        with pytest.raises(MuglaSubsamplingError):
            mss.validate_stage_range("plan", "nope")

    def test_no_evia_and_no_kozan(self):
        for token in ("evia_2021", "evia_2021_extended", "kozan_2023", "EVIA_test"):
            with pytest.raises(MuglaSubsamplingError):
                mss.assert_not_excluded(token)

    def test_resolve_experiments_rejects_excluded_and_duplicates(self):
        assert mss.resolve_experiments(None) == list(mss.PRIMARY_EXPERIMENTS)
        with pytest.raises(MuglaSubsamplingError):
            mss.resolve_experiments([MANAVGAT, BEJIS, "evia_2021"])
        with pytest.raises(MuglaSubsamplingError):
            mss.resolve_experiments([MUGLA, MUGLA, MANAVGAT])

    def test_module_contacts_no_earth_engine(self):
        source = Path(mss.__file__).read_text(encoding="utf-8")
        for token in ("import ee", "ee.Initialize", "gee_utils", "earthengine"):
            assert token not in source

    def test_expected_unique_fit_count(self):
        expected = mss.expected_unique_fit_count()
        assert expected == {"within_fits": 200, "source_fits": 40, "target_fits": 0,
                            "unique_fits": 240, "reuse_events": 40}


class TestHashGate:
    def test_matching_digests_pass(self, experiments_root, frozen_contract):
        inventory = mss.build_frozen_input_inventory(EXPERIMENTS, experiments_root)
        assert mss.assert_canonical_step8a_hashes(inventory)["all_match"]

    def test_mismatched_digest_fails_closed(self, monkeypatch, experiments_root, digests):
        broken = dict(digests)
        broken[MUGLA] = "0" * 64
        monkeypatch.setattr(mss, "CANONICAL_STEP8A_SHA256", broken)
        inventory = mss.build_frozen_input_inventory(EXPERIMENTS, experiments_root)
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_canonical_step8a_hashes(inventory, strict=True)

    def test_unregistered_experiment_fails_closed(self, monkeypatch, experiments_root,
                                                  digests):
        partial = {key: value for key, value in digests.items() if key != BEJIS}
        monkeypatch.setattr(mss, "CANONICAL_STEP8A_SHA256", partial)
        inventory = mss.build_frozen_input_inventory(EXPERIMENTS, experiments_root)
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_canonical_step8a_hashes(inventory, strict=True)

    def test_synthetic_frame_is_not_treated_as_production(self, experiments_root,
                                                          frozen_contract):
        inventory = mss.build_frozen_input_inventory(EXPERIMENTS, experiments_root)
        assert mss.is_production_mugla_frame(inventory) is False


# =============================================================================
# Allocation
# =============================================================================
class TestHamiltonAllocation:
    def test_floor_and_remainder_are_integer_exact(self):
        table = pd.DataFrame({
            "stratum_id": ["a|L0", "b|L0", "c|L1"],
            mss.BLOCK_COLUMN: ["a", "b", "c"],
            "label": [0, 0, 1],
            "capacity": [10, 7, 3],
        })
        allocated = mss.hamilton_allocation(table, 10)
        total = 20
        assert list(allocated["quota_numerator"]) == [100, 70, 30]
        assert list(allocated["floor_allocation"]) == [100 // total, 70 // total,
                                                       30 // total]
        assert list(allocated["remainder_numerator"]) == [100 % total, 70 % total,
                                                          30 % total]
        assert int(allocated["allocation_count"].sum()) == 10

    def test_tie_break_is_stratum_id_ascending(self):
        # Four identical capacities => a four-way remainder tie for two units.
        table = pd.DataFrame({
            "stratum_id": ["d|L0", "a|L0", "c|L0", "b|L0"],
            mss.BLOCK_COLUMN: ["d", "a", "c", "b"],
            "label": [0, 0, 0, 0],
            "capacity": [3, 3, 3, 3],
        })
        allocated = mss.hamilton_allocation(table, 6).set_index("stratum_id")
        assert int(allocated.attrs["strata_tied_at_cut"]) == 4
        assert int(allocated.attrs["tie_units_awarded"]) == 2
        winners = sorted(allocated.index[allocated["received_remainder_unit"]])
        assert winners == ["a|L0", "b|L0"]

    def test_tie_break_is_independent_of_input_order(self):
        table = pd.DataFrame({
            "stratum_id": ["a|L0", "b|L0", "c|L0", "d|L0"],
            mss.BLOCK_COLUMN: ["a", "b", "c", "d"],
            "label": [0, 0, 0, 0],
            "capacity": [3, 3, 3, 3],
        })
        forward = mss.hamilton_allocation(table, 6).set_index("stratum_id")
        backward = mss.hamilton_allocation(
            table.iloc[::-1].reset_index(drop=True), 6).set_index("stratum_id")
        assert forward["allocation_count"].to_dict() == backward["allocation_count"].to_dict()

    def test_capacity_is_never_exceeded(self, context):
        allocation = context["allocation_table"]
        assert int((allocation["allocation_count"] > allocation["capacity"]).sum()) == 0
        assert int(allocation["capacity_headroom"].min()) >= 0

    def test_no_stratum_is_dropped(self, context):
        assert int(context["allocation_table"]["allocation_count"].min()) >= 1

    def test_allocation_sums_to_the_target(self, context):
        assert int(context["allocation_table"]["allocation_count"].sum()) == TEST_SAMPLE_SIZE

    def test_over_target_request_fails_closed(self, context):
        with pytest.raises(MuglaSubsamplingError):
            mss.hamilton_allocation(context["capacity_table"], 10 ** 9)

    def test_assert_allocation_valid_rejects_overdraw(self, context):
        broken = context["allocation_table"].copy()
        broken.loc[broken.index[0], "allocation_count"] = int(
            broken.loc[broken.index[0], "capacity"]) + 1
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_allocation_valid(broken, TEST_SAMPLE_SIZE)

    def test_assert_allocation_valid_rejects_dropped_stratum(self, context):
        broken = context["allocation_table"].copy()
        broken.loc[broken.index[0], "allocation_count"] = 0
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_allocation_valid(broken, int(broken["allocation_count"].sum()))

    def test_prevalence_bound_is_respected(self, context):
        prevalence = context["prevalence"]
        assert prevalence["prevalence_within_bound"] is True
        assert prevalence["prevalence_absolute_drift"] <= prevalence["prevalence_drift_bound"]

    def test_synthetic_tie_break_is_exercised(self, context):
        # The synthetic grid is built so the largest-remainder cut is a real tie.
        attrs = context["allocation_table"].attrs
        assert int(attrs["strata_tied_at_cut"]) > 0
        assert 0 < int(attrs["tie_units_awarded"]) < int(attrs["strata_tied_at_cut"])


# =============================================================================
# Selection
# =============================================================================
class TestSelection:
    def test_exact_sample_size_every_repeat(self, context):
        sizes = context["selected_cells"].groupby("repeat_id").size()
        assert set(int(value) for value in sizes) == {TEST_SAMPLE_SIZE}

    def test_no_replacement(self, context):
        selected = context["selected_cells"]
        assert int(selected.duplicated(["repeat_id", "cell_id"]).sum()) == 0

    def test_exact_class_counts_every_repeat(self, context):
        selected = context["selected_cells"]
        positives = selected.groupby("repeat_id")["label"].sum()
        sizes = selected.groupby("repeat_id").size()
        expected_positives = context["prevalence"]["sampled_positives"]
        assert set(int(value) for value in positives) == {expected_positives}
        assert set(int(value) for value in (sizes - positives)) == {
            TEST_SAMPLE_SIZE - expected_positives}

    def test_all_blocks_retained(self, context):
        blocks = int(context["mugla"][mss.BLOCK_COLUMN].nunique())
        per_repeat = context["selected_cells"].groupby("repeat_id")["large_block_id"].nunique()
        assert set(int(value) for value in per_repeat) == {blocks}

    def test_selection_is_a_canonical_subset(self, context):
        canonical = set(context["mugla"]["cell_id"].astype(str))
        selected = set(context["selected_cells"]["cell_id"].astype(str))
        assert selected <= canonical
        assert selected != canonical

    def test_repeat_seeds_differ_deterministically(self, context):
        seeds = context["selected_cells"].groupby("repeat_id")["repeat_seed"].first()
        assert len(set(seeds)) == TEST_REPEATS
        assert int(seeds.loc[0]) == mss.repeat_seed(0)

    def test_repeats_select_different_cells(self, context):
        selected = context["selected_cells"]
        hashes = {
            int(repeat_id): mss.sample_hash(group["cell_id"].tolist())
            for repeat_id, group in selected.groupby("repeat_id")
        }
        assert len(set(hashes.values())) == TEST_REPEATS

    def test_row_order_invariance(self, context, frozen_contract):
        population = context["mugla"].copy()
        population["stratum_id"] = [
            mss.stratum_id_of(block, label)
            for block, label in zip(population[mss.BLOCK_COLUMN], population["label"])
        ]
        shuffled = population.sample(frac=1.0, random_state=4242).reset_index(drop=True)
        for repeat_id in range(TEST_REPEATS):
            straight = mss.select_repeat(population, context["allocation_table"], repeat_id)
            permuted = mss.select_repeat(shuffled, context["allocation_table"], repeat_id)
            assert (set(straight["cell_id"].astype(str))
                    == set(permuted["cell_id"].astype(str)))

    def test_stratum_iteration_order_invariance(self, context, frozen_contract):
        population = context["mugla"].copy()
        population["stratum_id"] = [
            mss.stratum_id_of(block, label)
            for block, label in zip(population[mss.BLOCK_COLUMN], population["label"])
        ]
        permuted_strata = context["allocation_table"].sample(frac=1.0, random_state=77)
        for repeat_id in range(TEST_REPEATS):
            straight = mss.select_repeat(population, context["allocation_table"], repeat_id)
            reordered = mss.select_repeat(population, permuted_strata, repeat_id)
            assert (set(straight["cell_id"].astype(str))
                    == set(reordered["cell_id"].astype(str)))

    def test_selection_is_reproducible_from_the_seed_alone(self, context, frozen_contract):
        population = context["mugla"].copy()
        population["stratum_id"] = [
            mss.stratum_id_of(block, label)
            for block, label in zip(population[mss.BLOCK_COLUMN], population["label"])
        ]
        again = mss.select_repeat(population, context["allocation_table"], 0)
        emitted = context["selected_cells"]
        assert (set(again["cell_id"].astype(str))
                == set(emitted.loc[emitted["repeat_id"] == 0, "cell_id"].astype(str)))

    def test_per_stratum_counts_equal_the_allocation(self, context):
        allocation = dict(zip(context["allocation_table"]["stratum_id"],
                              context["allocation_table"]["allocation_count"]))
        counts = context["selected_cells"].groupby(["repeat_id", "stratum_id"]).size()
        for (_repeat_id, stratum_id), count in counts.items():
            assert int(count) == int(allocation[stratum_id])

    def test_selected_cells_carry_the_required_provenance(self, context):
        for column in ("repeat_id", "cell_id", "large_block_id", "label", "fold_id",
                       "sampling_seed", "stratum_id", "allocation_count",
                       "mugla_step8a_sha256"):
            assert column in context["selected_cells"].columns

    def test_sampling_seed_is_derived_from_repeat_and_stratum(self, context):
        row = context["selected_cells"].iloc[0]
        assert int(row["sampling_seed"]) == mss.stratum_seed(
            int(row["repeat_id"]), str(row["stratum_id"]))


# =============================================================================
# Fold inheritance
# =============================================================================
class TestFolds:
    def test_fold_mapping_is_inherited_from_the_artifact(self, context):
        mapping = context["fold_mapping"]
        assert set(mapping["fold_source"].unique()) == {"persisted_artifact"}
        assert context["fold_provenance"]["reoptimised_per_repeat"] is False
        assert len(mapping) == len(context["mugla"])

    def test_block_never_crosses_a_fold(self, context):
        spans = context["fold_mapping"].groupby(mss.BLOCK_COLUMN)["fold_id"].nunique()
        assert int(spans.max()) == 1

    def test_every_repeat_inherits_the_same_mapping(self, context):
        lookup = dict(zip(context["fold_mapping"]["cell_id"].astype(str),
                          context["fold_mapping"]["fold_id"].astype(int)))
        selected = context["selected_cells"]
        mismatches = sum(
            1 for cell_id, fold_id in zip(selected["cell_id"].astype(str),
                                          selected["fold_id"].astype(int))
            if lookup[cell_id] != fold_id
        )
        assert mismatches == 0

    def test_per_fold_composition_is_identical_across_repeats(self, context):
        summary = context["fold_summary"]
        assert summary["identical_across_repeats"] is True
        assert len(summary["per_fold_rows"]) == mss.FOLD_COUNT
        assert sum(summary["per_fold_rows"]) == TEST_SAMPLE_SIZE
        assert sum(summary["per_fold_positives"]) == context["prevalence"]["sampled_positives"]

    def test_both_classes_on_both_sides_of_every_fold(self, context):
        composition = mss.fold_composition(context["selected_cells"])
        for _, row in composition.iterrows():
            assert int(row["positives"]) >= 1
            assert int(row["rows"]) - int(row["positives"]) >= 1

    def test_missing_fold_artifact_fails_closed(self, tmp_path, context):
        with pytest.raises(MuglaSubsamplingError):
            mss.load_frozen_fold_mapping(context["mugla"], tmp_path)

    def test_block_spanning_two_folds_fails_closed(self, context):
        broken = context["fold_mapping"].copy()
        first_block = broken[mss.BLOCK_COLUMN].iloc[0]
        block_rows = broken.index[broken[mss.BLOCK_COLUMN] == first_block]
        broken.loc[block_rows[0], "fold_id"] = (
            int(broken.loc[block_rows[0], "fold_id"]) + 1) % mss.FOLD_COUNT
        broken.loc[block_rows[1], "fold_id"] = (
            int(broken.loc[block_rows[1], "fold_id"]) + 2) % mss.FOLD_COUNT
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_fold_contract(broken)

    def test_incomplete_fold_coverage_fails_closed(self, context, experiments_root):
        truncated = context["mugla"].iloc[:-5]
        with pytest.raises(MuglaSubsamplingError):
            mss.load_frozen_fold_mapping(truncated, experiments_root)

    def test_oof_coverage_assertion(self):
        mss.assert_full_oof_coverage(np.ones(5, dtype=int), "ctx")
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_full_oof_coverage(np.array([1, 0, 1]), "ctx")
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_full_oof_coverage(np.array([1, 2, 1]), "ctx")


# =============================================================================
# Fit registry and arms
# =============================================================================
class TestFitRegistry:
    def test_duplicate_identity_is_never_refitted(self):
        registry = mss.FitRegistry()
        calls = {"n": 0}

        def _compute():
            calls["n"] += 1
            return np.array([0.5])

        identity = mss.source_fit_identity(0, "thermal")
        registry.get_or_fit(identity, mss.ARM_SOURCE, _compute)
        registry.get_or_fit(identity, mss.ARM_SOURCE, _compute)
        assert calls["n"] == 1
        assert registry.fit_count == 1
        assert registry.reuse_count == 1

    def test_source_identity_excludes_the_target(self):
        assert mss.source_fit_identity(3, "thermal") == "mugla_as_source|3|thermal"
        assert mss.within_fit_identity(3, 2, "baseline") == "within_mugla|3|2|baseline"

    def test_accounting_separates_the_arms(self):
        registry = mss.FitRegistry()
        registry.get_or_fit(mss.within_fit_identity(0, 0, "baseline"),
                            mss.ARM_WITHIN, lambda: np.array([0.1]))
        registry.get_or_fit(mss.source_fit_identity(0, "baseline"),
                            mss.ARM_SOURCE, lambda: np.array([0.2]))
        registry.get_or_fit(mss.source_fit_identity(0, "baseline"),
                            mss.ARM_SOURCE, lambda: np.array([0.2]))
        accounting = registry.accounting()
        assert accounting["within_fits"] == 1
        assert accounting["source_fits"] == 2 - 1
        assert accounting["target_fits"] == 0
        assert accounting["reuse_events"] == 1

    def test_released_identity_cannot_be_silently_refitted(self):
        registry = mss.FitRegistry()
        identity = mss.source_fit_identity(0, "thermal")
        registry.get_or_fit(identity, mss.ARM_SOURCE, lambda: np.array([0.3]))
        registry.release(f"{mss.ARM_SOURCE}|0|")
        with pytest.raises(MuglaSubsamplingError):
            registry.get_or_fit(identity, mss.ARM_SOURCE, lambda: np.array([0.3]))


class TestArms:
    def test_target_arm_performs_no_fit(self, monkeypatch, context):
        def _forbidden(*args, **kwargs):
            raise AssertionError("the target arm must not construct a model")

        monkeypatch.setattr(mss, "build_pipeline", _forbidden)
        selected = context["selected_cells"]
        repeat_cells = selected[selected["repeat_id"] == 0]
        repeat_frame = (context["mugla"].set_index("cell_id", drop=False)
                        .loc[repeat_cells["cell_id"].to_numpy()].copy())
        repeat_frame["fold_id"] = repeat_cells["fold_id"].to_numpy()
        predictions = mss.run_target_arm(repeat_frame.reset_index(drop=True), 0,
                                         context["references"])
        assert len(predictions) == TEST_SAMPLE_SIZE * len(mss.TARGET_PAIRS)
        assert bool(predictions["reused_from_artifact"].all())

    def test_target_arm_is_an_exact_subset_of_the_frozen_predictions(self, context):
        selected = context["selected_cells"]
        repeat_cells = selected[selected["repeat_id"] == 0]
        repeat_frame = (context["mugla"].set_index("cell_id", drop=False)
                        .loc[repeat_cells["cell_id"].to_numpy()].copy())
        repeat_frame["fold_id"] = repeat_cells["fold_id"].to_numpy()
        predictions = mss.run_target_arm(repeat_frame.reset_index(drop=True), 0,
                                         context["references"])
        for direction, entry in context["references"]["target"].items():
            frozen = entry["predictions"].set_index("target_cell_id")
            emitted = predictions[predictions["direction"] == direction]
            assert set(emitted["target_cell_id"]) == set(
                repeat_cells["cell_id"].astype(str))
            for family in mss.MODEL_FAMILIES:
                expected = frozen.loc[emitted["target_cell_id"].to_numpy(),
                                      f"{family}_probability"].to_numpy()
                assert np.array_equal(emitted[f"{family}_probability"].to_numpy(),
                                      expected)

    def test_source_fit_is_reused_for_both_targets(self, context):
        registry = mss.FitRegistry()
        selected = context["selected_cells"]
        repeat_cells = selected[selected["repeat_id"] == 0]
        repeat_frame = (context["mugla"].set_index("cell_id", drop=False)
                        .loc[repeat_cells["cell_id"].to_numpy()].copy().reset_index(drop=True))
        targets = {target_id: context["populations"][target_id]
                   for _, target_id in mss.SOURCE_PAIRS}
        predictions = mss.run_source_arm(repeat_frame, 0, targets, registry)
        accounting = registry.accounting()
        assert accounting["source_fits"] == len(mss.MODEL_FAMILIES)
        assert accounting["reuse_events"] == len(mss.MODEL_FAMILIES)
        assert accounting["source_reuse_per_fit"] == 2.0
        for _, target_id in mss.SOURCE_PAIRS:
            direction = mss.direction_token(MUGLA, target_id)
            assert (len(predictions[predictions["direction"] == direction])
                    == len(context["populations"][target_id]))

    def test_source_arm_target_cohorts_stay_full_and_unmodified(self, context):
        registry = mss.FitRegistry()
        selected = context["selected_cells"]
        repeat_cells = selected[selected["repeat_id"] == 0]
        repeat_frame = (context["mugla"].set_index("cell_id", drop=False)
                        .loc[repeat_cells["cell_id"].to_numpy()].copy().reset_index(drop=True))
        targets = {target_id: context["populations"][target_id]
                   for _, target_id in mss.SOURCE_PAIRS}
        predictions = mss.run_source_arm(repeat_frame, 0, targets, registry)
        for _, target_id in mss.SOURCE_PAIRS:
            direction = mss.direction_token(MUGLA, target_id)
            emitted = predictions[predictions["direction"] == direction]
            canonical = context["populations"][target_id]
            assert set(emitted["target_cell_id"]) == set(canonical["cell_id"])
            assert int(emitted[mss.TARGET_COLUMN].sum()) == int(canonical["label"].sum())

    def test_within_arm_gives_complete_oof_coverage(self, context):
        registry = mss.FitRegistry()
        selected = context["selected_cells"]
        repeat_cells = selected[selected["repeat_id"] == 0]
        repeat_frame = (context["mugla"].set_index("cell_id", drop=False)
                        .loc[repeat_cells["cell_id"].to_numpy()].copy())
        repeat_frame["fold_id"] = repeat_cells["fold_id"].to_numpy()
        repeat_frame["repeat_id"] = 0
        predictions = mss.run_within_arm(repeat_frame.reset_index(drop=True), 0, registry)
        assert len(predictions) == TEST_SAMPLE_SIZE
        assert int(predictions["cell_id"].duplicated().sum()) == 0
        for family in mss.MODEL_FAMILIES:
            assert not predictions[f"{family}_probability"].isna().any()
        assert registry.accounting()["within_fits"] == mss.FOLD_COUNT * len(
            mss.MODEL_FAMILIES)


# =============================================================================
# Metric arithmetic
# =============================================================================
class TestMetricArithmetic:
    def test_roc_and_pr_delta_arithmetic(self):
        for metric in ("roc_auc", "pr_auc"):
            assert mss.natural_delta(0.80, 0.75) == pytest.approx(0.05)
            assert mss.oriented_delta(metric, 0.80, 0.75) == pytest.approx(0.05)
            assert mss.oriented_delta(metric, 0.70, 0.75) == pytest.approx(-0.05)
            assert mss.metric_orientation(metric) == mss.ORIENTATION_HIGHER

    def test_brier_natural_and_oriented_delta(self):
        # A LOWER Brier is better, so a lower subsample value is a POSITIVE
        # oriented delta while the natural delta stays negative.
        assert mss.natural_delta(0.09, 0.11) == pytest.approx(-0.02)
        assert mss.oriented_delta("brier_score", 0.09, 0.11) == pytest.approx(0.02)
        assert mss.oriented_delta("brier_score", 0.13, 0.11) == pytest.approx(-0.02)
        assert mss.metric_orientation("brier_score") == mss.ORIENTATION_NEGATED

    def test_oriented_delta_positive_always_means_better(self):
        assert mss.oriented_delta("roc_auc", 0.9, 0.8) > 0
        assert mss.oriented_delta("pr_auc", 0.9, 0.8) > 0
        assert mss.oriented_delta("brier_score", 0.05, 0.10) > 0

    def test_none_propagates(self):
        assert mss.natural_delta(None, 0.5) is None
        assert mss.oriented_delta("roc_auc", 0.5, None) is None

    def test_interval_summary(self):
        values = [0.10, 0.20, 0.30, 0.40, 0.50]
        summary = mss.subsampling_interval(values)
        assert summary["median"] == pytest.approx(0.30)
        assert summary["minimum"] == pytest.approx(0.10)
        assert summary["maximum"] == pytest.approx(0.50)
        assert summary["n_repeats_observed"] == 5
        assert summary["interval_lower"] == pytest.approx(
            float(np.percentile(values, 2.5, method="linear")))
        assert summary["interval_upper"] == pytest.approx(
            float(np.percentile(values, 97.5, method="linear")))

    def test_interval_handles_no_observations(self):
        summary = mss.subsampling_interval([None, float("nan")])
        assert summary["n_repeats_observed"] == 0
        assert summary["median"] is None

    def test_reference_position_token_for_higher_is_better(self):
        assert mss.reference_position("roc_auc", 0.50, 0.60, 0.80) == mss.POSITION_BELOW
        assert mss.reference_position("roc_auc", 0.70, 0.60, 0.80) == mss.POSITION_INSIDE
        assert mss.reference_position("roc_auc", 0.90, 0.60, 0.80) == mss.POSITION_ABOVE

    def test_reference_position_token_for_brier_uses_the_oriented_scale(self):
        # Brier 0.02 is BETTER than the [0.10, 0.20] subsample range, so on the
        # oriented scale the reference sits ABOVE it.
        assert mss.reference_position("brier_score", 0.02, 0.10, 0.20) == mss.POSITION_ABOVE
        assert mss.reference_position("brier_score", 0.15, 0.10, 0.20) == mss.POSITION_INSIDE
        assert mss.reference_position("brier_score", 0.40, 0.10, 0.20) == mss.POSITION_BELOW

    def test_interpretation_sentences_are_the_two_permitted_ones(self):
        assert mss.interpretation_sentence(mss.POSITION_INSIDE) == mss.SENTENCE_INSIDE
        assert mss.interpretation_sentence(mss.POSITION_ABOVE) == mss.SENTENCE_OUTSIDE
        assert mss.interpretation_sentence(mss.POSITION_BELOW) == mss.SENTENCE_OUTSIDE
        with pytest.raises(MuglaSubsamplingError):
            mss.interpretation_sentence("maybe")


class TestReferences:
    def test_all_references_recompute_from_their_prediction_vectors(self, context):
        rows = context["reference_rows"]
        assert len(rows) == (len(mss.all_direction_rows()) * len(mss.MODEL_FAMILIES)
                             * len(mss.METRICS))
        assert all(row["recomputation_matches"] for row in rows)

    def test_reference_arms_and_directions_are_complete(self, context):
        pairs = {(row["arm"], row["direction"]) for row in context["reference_rows"]}
        assert pairs == set(mss.all_direction_rows())

    def test_direction_resolution_uses_the_source_first_pair_directory(self, context):
        for direction, entry in context["references"]["target"].items():
            expected = f"{entry['source_experiment_id']}__{entry['target_experiment_id']}"
            assert Path(entry["metrics_path"]).parent.parent.name == expected

    def test_unreproducible_reference_fails_closed(self, context):
        broken = {
            "within": dict(context["references"]["within"]),
            "source": dict(context["references"]["source"]),
            "target": dict(context["references"]["target"]),
        }
        broken["within"] = dict(broken["within"])
        broken["within"]["values"] = {
            family: {metric: 0.123456 for metric in mss.METRICS}
            for family in mss.MODEL_FAMILIES
        }
        with pytest.raises(MuglaSubsamplingError):
            mss.build_reference_metrics(broken)


# =============================================================================
# End-to-end artifact
# =============================================================================
class TestEndToEnd:
    def test_all_three_stages_ran(self, artifact):
        assert artifact["result"]["stages_executed"] == ["plan", "fit", "summarize"]
        assert artifact["result"]["earth_engine_used"] is False

    def test_every_required_output_exists(self, artifact):
        root = artifact["root"]
        for relative in ("config.json", "input_hashes.json", "sampling_inventory.csv",
                         "stratum_allocation.csv", "selected_cells.parquet",
                         "fold_mapping.parquet", "reference_metrics.csv",
                         "repeat_metrics.csv", "subsampling_summary.csv",
                         "summary.json", "report.md", "manifest.json",
                         "stages/plan.json", "stages/fit.json", "stages/summarize.json"):
            assert (root / relative).exists(), relative
        predictions = root / mss.OOF_PREDICTIONS_DIRNAME
        assert predictions.is_dir()
        assert {path.name for path in predictions.glob("*.parquet")} == {
            f"part-{arm}.parquet" for arm in mss.ARMS}

    def test_manifest_exposes_one_logical_dataset(self, artifact):
        manifest = json.loads((artifact["root"] / "manifest.json").read_text())
        logical = manifest["logical_datasets"][mss.OOF_PREDICTIONS_DIRNAME]
        assert logical["kind"] == "partitioned_parquet_dataset"
        assert set(logical["parts"]) == {f"part-{arm}.parquet" for arm in mss.ARMS}
        loose = [entry["path"] for entry in manifest["files"]
                 if entry["path"].startswith(mss.OOF_PREDICTIONS_DIRNAME)]
        assert loose == []

    def test_fit_accounting_matches_the_contract(self, artifact):
        marker = json.loads((artifact["root"] / "stages" / "fit.json").read_text())
        accounting = marker["fit_accounting"]
        assert accounting["within_fits"] == TEST_REPEATS * len(mss.MODEL_FAMILIES) * mss.FOLD_COUNT
        assert accounting["source_fits"] == TEST_REPEATS * len(mss.MODEL_FAMILIES)
        assert accounting["target_fits"] == 0
        assert accounting["reuse_events"] == accounting["source_fits"]
        assert accounting["unique_fits"] == (accounting["within_fits"]
                                            + accounting["source_fits"])

    def test_repeat_metrics_are_complete(self, artifact):
        repeat_metrics = pd.read_csv(artifact["root"] / "repeat_metrics.csv")
        expected = (len(mss.all_direction_rows()) * len(mss.MODEL_FAMILIES)
                    * len(mss.METRICS) * TEST_REPEATS)
        assert len(repeat_metrics) == expected
        assert not repeat_metrics["subsample_value"].isna().any()
        assert set(repeat_metrics["arm"].unique()) == set(mss.ARMS)

    def test_summary_rows_are_complete(self, artifact):
        summary = pd.read_csv(artifact["root"] / "subsampling_summary.csv")
        assert len(summary) == (len(mss.all_direction_rows()) * len(mss.MODEL_FAMILIES)
                                * len(mss.METRICS))
        assert set(summary["n_repeats_observed"]) == {TEST_REPEATS}
        assert set(summary["reference_position"]) <= set(mss.POSITION_TOKENS)

    def test_same_repeat_sample_is_shared_across_all_three_arms(self, artifact):
        root = artifact["root"]
        selected = pd.read_parquet(root / "selected_cells.parquet")
        within = pd.read_parquet(root / mss.OOF_PREDICTIONS_DIRNAME
                                 / f"part-{mss.ARM_WITHIN}.parquet")
        target = pd.read_parquet(root / mss.OOF_PREDICTIONS_DIRNAME
                                 / f"part-{mss.ARM_TARGET}.parquet")
        for repeat_id in range(TEST_REPEATS):
            expected = set(selected.loc[selected["repeat_id"] == repeat_id,
                                        "cell_id"].astype(str))
            assert set(within.loc[within["repeat_id"] == repeat_id,
                                  "cell_id"].astype(str)) == expected
            for direction in mss.target_directions():
                rows = target[(target["repeat_id"] == repeat_id)
                              & (target["direction"] == direction)]
                assert set(rows["target_cell_id"].astype(str)) == expected

    def test_brier_orientation_in_the_emitted_metrics(self, artifact):
        repeat_metrics = pd.read_csv(artifact["root"] / "repeat_metrics.csv")
        brier = repeat_metrics[repeat_metrics["metric"] == "brier_score"]
        assert np.allclose(brier["oriented_delta"], -brier["natural_delta"], atol=1e-12)
        assert (brier["metric_orientation"] == mss.ORIENTATION_NEGATED).all()
        aucs = repeat_metrics[repeat_metrics["metric"] != "brier_score"]
        assert np.allclose(aucs["oriented_delta"], aucs["natural_delta"], atol=1e-12)

    def test_natural_delta_arithmetic_in_the_emitted_metrics(self, artifact):
        repeat_metrics = pd.read_csv(artifact["root"] / "repeat_metrics.csv")
        assert np.allclose(
            repeat_metrics["natural_delta"],
            repeat_metrics["subsample_value"] - repeat_metrics["full_reference_value"],
            atol=1e-12)

    def test_no_forbidden_language(self, artifact):
        assert mss.scan_forbidden_tokens(artifact["root"]) == []

    def test_no_p_values_or_confidence_columns(self, artifact):
        for path in artifact["root"].rglob("*.csv"):
            columns = list(pd.read_csv(path, nrows=0).columns)
            assert not any(column.startswith("ci_") for column in columns), path
            assert not any("p_value" in column for column in columns), path
        summary = pd.read_csv(artifact["root"] / "subsampling_summary.csv")
        assert "subsampling_interval_lower" in summary.columns
        assert "subsampling_interval_upper" in summary.columns

    def test_only_permitted_interpretation_sentences(self, artifact):
        summary = pd.read_csv(artifact["root"] / "subsampling_summary.csv")
        assert set(summary["interpretation_sentence"].dropna()) <= {
            mss.SENTENCE_INSIDE, mss.SENTENCE_OUTSIDE}

    def test_no_evia_participates_anywhere(self, artifact):
        """Evia may appear ONLY as a key of the exclusion declaration.

        Naming it there is the provenance record of the exclusion; naming it
        anywhere else would mean it participated.
        """
        root = artifact["root"]
        config = json.loads((root / "config.json").read_text())
        scientific = config["scientific_config"]
        assert not any("evia" in experiment_id.lower()
                       for experiment_id in scientific["experiments"])
        assert all("evia" in key.lower() or "kozan" in key.lower()
                   for key in scientific["excluded_experiments"]
                   if "evia" in key.lower())
        for directions in scientific["directions"].values():
            assert not any("evia" in direction.lower() for direction in directions)

        # Every other emitted artifact must not mention it at all.
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in (".csv", ".md"):
                assert "evia" not in path.read_text(encoding="utf-8").lower(), path
            elif path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                for column in frame.columns:
                    if frame[column].dtype == object or str(frame[column].dtype) == "str":
                        assert not frame[column].astype(str).str.lower() \
                            .str.contains("evia").any(), f"{path}:{column}"

    def test_summary_and_report_never_mention_an_excluded_aoi(self, artifact):
        for relative in ("summary.json", "report.md"):
            text = (artifact["root"] / relative).read_text(encoding="utf-8").lower()
            assert "evia" not in text
            assert "kozan" not in text

    def test_summary_carries_the_limitations_and_joint_reading(self, artifact):
        summary = json.loads((artifact["root"] / "summary.json").read_text())
        assert len(summary["limitations"]) == len(mss.LIMITATIONS)
        assert summary["three_arm_reading"]["joint_reading_required"] is True
        assert summary["earth_engine_used"] is False
        assert summary["bootstrap_performed"] is False

    def test_output_is_confined_to_the_namespace(self, artifact):
        root = artifact["root"]
        assert root.parent.name == mss.DIAGNOSTIC_NAMESPACE
        assert root.parent.parent.name == "diagnostics"
        manifest = json.loads((root / "manifest.json").read_text())
        for entry in manifest["files"]:
            mss.assert_inside_namespace(root / entry["path"], root)

    def test_canonical_inputs_are_unchanged(self, artifact, digests, experiments_root):
        for experiment_id, digest in digests.items():
            path = (experiments_root / experiment_id / "step8a"
                    / "step8a_500m_modeling_dataset.parquet")
            assert mss.sha256_file(path) == digest


# =============================================================================
# Run safety: dry-run, resume, force
# =============================================================================
class TestRunSafety:
    def test_dry_run_writes_nothing_and_fits_nothing(self, monkeypatch, tmp_path,
                                                     experiments_root, output_root,
                                                     frozen_contract):
        run_root = tmp_path / "outputs"

        def _forbidden(*args, **kwargs):
            raise AssertionError("dry-run must not construct a model")

        monkeypatch.setattr(mss, "build_pipeline", _forbidden)
        result = mss.run_analysis(experiments=EXPERIMENTS, dry_run=True,
                                  output_root=run_root,
                                  experiments_root=experiments_root)
        assert result["dry_run"] is True
        assert result["ran"] is False
        assert result["fit_performed"] is False
        assert result["files_written"] == []
        assert result["expected_fit_accounting"]["target_fits"] == 0
        assert not run_root.exists()

    def test_dry_run_reports_the_planned_layout(self, tmp_path, experiments_root,
                                                frozen_contract):
        result = mss.run_analysis(experiments=EXPERIMENTS, dry_run=True,
                                  output_root=tmp_path / "outputs",
                                  experiments_root=experiments_root)
        planned = result["planned_outputs"]
        for relative in ("config.json", "selected_cells.parquet", "manifest.json",
                         "subsampling_summary.csv"):
            assert relative in planned

    def test_existing_namespace_is_not_overwritten(self, artifact, experiments_root,
                                                   digests):
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
            patcher.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
            patcher.setattr(mss, "N_REPEATS", TEST_REPEATS)
            with pytest.raises(MuglaSubsamplingError):
                mss.run_analysis(experiments=EXPERIMENTS,
                                 output_root=artifact["output_root"],
                                 experiments_root=experiments_root)

    def test_resume_reuses_hash_bound_stages(self, artifact, experiments_root, digests):
        root = artifact["root"]
        before = {path: mss.sha256_file(path)
                  for path in sorted(root.rglob("*")) if path.is_file()}
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
            patcher.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
            patcher.setattr(mss, "N_REPEATS", TEST_REPEATS)
            result = mss.run_analysis(experiments=EXPERIMENTS, resume=True,
                                      output_root=artifact["output_root"],
                                      experiments_root=experiments_root)
        assert all(entry.get("reused") for entry in result["stage_results"])
        after = {path: mss.sha256_file(path)
                 for path in sorted(root.rglob("*")) if path.is_file()}
        assert before == after

    def test_partial_stage_is_rejected_by_resume(self, artifact):
        analysis_id = artifact["root"].name
        state = mss.verify_stage_complete(analysis_id, "fit", artifact["output_root"])
        assert state["complete"] is True

        marker_path = mss.stage_marker_path(analysis_id, "fit", artifact["output_root"])
        original = marker_path.read_text(encoding="utf-8")
        marker = json.loads(original)
        marker["files"]["repeat_metrics.csv"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        try:
            broken = mss.verify_stage_complete(analysis_id, "fit",
                                               artifact["output_root"])
            assert broken["complete"] is False
            assert "hash drift" in broken["reason"]
        finally:
            marker_path.write_text(original, encoding="utf-8")

    def test_missing_partition_is_rejected_by_resume(self, artifact):
        analysis_id = artifact["root"].name
        assert mss.verify_arm_partition(analysis_id, mss.ARM_WITHIN,
                                        artifact["output_root"]) is True
        assert mss.verify_arm_partition(analysis_id, "not_an_arm",
                                        artifact["output_root"]) is False

    def test_prerequisite_stage_is_required(self, tmp_path, experiments_root,
                                            frozen_contract):
        with pytest.raises(MuglaSubsamplingError):
            mss.run_analysis(experiments=EXPERIMENTS, from_stage="fit",
                             to_stage="fit", output_root=tmp_path / "outputs",
                             experiments_root=experiments_root)

    def test_force_quarantines_without_deleting(self, tmp_path, experiments_root,
                                                output_root, digests):
        import shutil
        run_root = tmp_path / "outputs"
        shutil.copytree(output_root / "cross_region", run_root / "cross_region")
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(mss, "CANONICAL_STEP8A_SHA256", dict(digests))
            patcher.setattr(mss, "TARGET_SAMPLE_SIZE", TEST_SAMPLE_SIZE)
            patcher.setattr(mss, "N_REPEATS", TEST_REPEATS)
            first = mss.run_analysis(experiments=EXPERIMENTS, from_stage="plan",
                                     to_stage="plan", output_root=run_root,
                                     experiments_root=experiments_root)
            root = Path(first["output_namespace"])
            marker = mss.sha256_file(root / "config.json")
            second = mss.run_analysis(experiments=EXPERIMENTS, from_stage="plan",
                                      to_stage="plan", force=True,
                                      output_root=run_root,
                                      experiments_root=experiments_root)
        quarantined = Path(second["quarantined_previous_namespace"])
        assert quarantined.is_dir()
        assert (quarantined / "config.json").is_file()
        assert mss.sha256_file(quarantined / "config.json") == marker
        assert mss.QUARANTINE_DIRNAME in quarantined.parts

    def test_namespace_containment_guard(self, tmp_path):
        root = tmp_path / "namespace"
        root.mkdir()
        mss.assert_inside_namespace(root / "inner" / "file.json", root)
        with pytest.raises(MuglaSubsamplingError):
            mss.assert_inside_namespace(tmp_path / "elsewhere.json", root)


# =============================================================================
# Validator identity contracts
# =============================================================================
def _paired_frame(n_blocks: int = 6) -> pd.DataFrame:
    """Four cells per block, labelled [1, 1, 0, 0].

    Every block therefore yields two strata of capacity exactly 2, so a
    half-size target allocates exactly one cell per stratum. Two repeats can
    then partition the population perfectly: each is a proper subset, and
    their union is the whole population.
    """
    rng = np.random.default_rng(9)
    rows: list[dict] = []
    for block in range(n_blocks):
        for offset, label in enumerate((1, 1, 0, 0)):
            row_500m = block * mss.BLOCK_SIZE_CELLS + offset
            col_500m = 0
            signal = 1.4 if label else 0.0
            rows.append({
                "cell_id": f"r{row_500m}_c{col_500m}",
                "row_500m": row_500m,
                "col_500m": col_500m,
                "burned": label,
                "valid_for_modeling": True,
                mss.POPULATION: True,
                "ndvi_mean": float(rng.normal(0.4 - 0.1 * signal, 0.05)),
                "elevation_mean": float(rng.normal(500 + 40 * signal, 25)),
                "slope_mean": float(rng.normal(10 + signal, 2)),
                "landcover_dominant": int(rng.choice([10, 20, 30])),
                "lst_anomaly_mean": float(rng.normal(signal, 0.4)),
                "current_lst_mean": float(rng.normal(305 + 2 * signal, 1.5)),
                "current_tvdi_mean": float(rng.normal(0.5 + 0.1 * signal, 0.05)),
                "tvdi_difference_mean": float(rng.normal(0.05 * signal, 0.03)),
                "downscaled_lst_mean": float(rng.normal(304 + 2 * signal, 1.5)),
                "fused_lst_mean": float(rng.normal(304.5 + 2 * signal, 1.5)),
            })
    return pd.DataFrame(rows)


def _partitioning_selection(population: pd.DataFrame,
                            allocation: pd.DataFrame) -> pd.DataFrame:
    """Two repeats that partition the population, one cell per stratum each."""
    enriched = population.copy()
    enriched["stratum_id"] = [
        mss.stratum_id_of(block, label)
        for block, label in zip(enriched[mss.BLOCK_COLUMN], enriched["label"])
    ]
    counts = dict(zip(allocation["stratum_id"], allocation["allocation_count"]))
    frames: list[pd.DataFrame] = []
    for repeat_id in (0, 1):
        picked: list[pd.DataFrame] = []
        for stratum_id, group in enriched.groupby("stratum_id", sort=True):
            ordered = group.sort_values("cell_id", kind="mergesort")
            take = int(counts[stratum_id])
            slice_ = ordered.iloc[repeat_id * take:(repeat_id + 1) * take].copy()
            slice_["sampling_seed"] = mss.stratum_seed(repeat_id, stratum_id)
            slice_["allocation_count"] = take
            picked.append(slice_)
        selection = pd.concat(picked, ignore_index=True)
        selection["repeat_id"] = repeat_id
        selection["repeat_seed"] = mss.repeat_seed(repeat_id)
        frames.append(selection)
    return pd.concat(frames, ignore_index=True)


class TestValidatorRepeatSubsetIdentity:
    """C5 is a per-repeat property, never a property of the repeats' union."""

    def test_helper_accepts_a_union_that_covers_the_population(self):
        canonical = {f"c{i}" for i in range(10)}
        selected = pd.DataFrame({
            "repeat_id": [0] * 5 + [1] * 5,
            "cell_id": [f"c{i}" for i in range(5)] + [f"c{i}" for i in range(5, 10)],
        })
        status = vms.repeat_subset_status(selected, canonical)
        assert set(selected["cell_id"]) == canonical          # union IS full
        assert all(entry["is_subset"] and entry["is_proper"]
                   for entry in status.values())

    def test_helper_rejects_a_single_repeat_that_covers_the_population(self):
        canonical = {f"c{i}" for i in range(4)}
        selected = pd.DataFrame({
            "repeat_id": [0] * 4 + [1] * 2,
            "cell_id": ["c0", "c1", "c2", "c3", "c0", "c1"],
        })
        status = vms.repeat_subset_status(selected, canonical)
        assert status[0]["is_proper"] is False
        assert status[1]["is_proper"] is True

    def test_helper_flags_a_foreign_cell(self):
        canonical = {"c0", "c1", "c2"}
        selected = pd.DataFrame({"repeat_id": [0, 0], "cell_id": ["c0", "elsewhere"]})
        status = vms.repeat_subset_status(selected, canonical)
        assert status[0]["is_subset"] is False
        assert status[0]["foreign_cells"] == 1

    def test_c5_passes_when_the_repeat_union_covers_the_whole_population(self):
        population = _population(_paired_frame())
        capacity = mss.stratum_capacity_table(population)
        target_total = len(population) // 2
        allocation = mss.hamilton_allocation(capacity, target_total)
        selected = _partitioning_selection(population, allocation)

        # Precondition: each repeat is half the population, the union is all of it.
        assert set(selected["cell_id"]) == set(population["cell_id"])
        assert set(selected.groupby("repeat_id").size()) == {target_total}

        report = vms.Report()
        vms.run_sampling_checks(report, selected, allocation, population,
                                {"n_repeats": 2, "target_sample_size": target_total,
                                 "production": False})
        c5 = next(check for check in report.checks if check["check_id"] == "C5")
        assert c5["status"] == vms.PASS, c5
        assert c5["observed"]["union_over_repeats"] == len(population)
        assert "not a failure" in (c5["note"] or "")

    def test_c5_fails_when_one_repeat_is_the_whole_population(self):
        population = _population(_paired_frame())
        capacity = mss.stratum_capacity_table(population)
        target_total = len(population) // 2
        allocation = mss.hamilton_allocation(capacity, target_total)
        selected = _partitioning_selection(population, allocation)
        # Repeat 0 now contains every canonical cell: a real violation.
        whole = selected[selected["repeat_id"] == 1].copy()
        extra = selected[selected["repeat_id"] == 0].copy()
        extra["repeat_id"] = 1
        broken = pd.concat([selected[selected["repeat_id"] == 0], whole, extra],
                           ignore_index=True)

        report = vms.Report()
        vms.run_sampling_checks(report, broken, allocation, population,
                                {"n_repeats": 2, "target_sample_size": target_total,
                                 "production": False})
        c5 = next(check for check in report.checks if check["check_id"] == "C5")
        assert c5["status"] == vms.FAIL
        assert 1 in c5["observed"]["offending_repeats"]


class TestValidatorCompositeCellIdentity:
    """H5b must count (target region, cell_id), never the bare AOI-local token."""

    def _source_partition(self, manavgat_cells: int, bejis_cells: int,
                          duplicate_in_manavgat: bool = False) -> pd.DataFrame:
        """Two target cohorts whose cell_id tokens deliberately COLLIDE."""
        manavgat = [f"r{i}_c0" for i in range(manavgat_cells)]
        # Same tokens, different region -> a bare cell_id count would collapse them.
        bejis = [f"r{i}_c0" for i in range(bejis_cells)]
        if duplicate_in_manavgat:
            manavgat[-1] = manavgat[0]
        rows = pd.DataFrame({
            "repeat_id": [0] * (len(manavgat) + len(bejis)),
            "direction": ([mss.direction_token(MUGLA, MANAVGAT)] * len(manavgat)
                          + [mss.direction_token(MUGLA, BEJIS)] * len(bejis)),
            "target_experiment_id": ([MANAVGAT] * len(manavgat)
                                     + [BEJIS] * len(bejis)),
            "target_cell_id": manavgat + bejis,
        })
        return rows

    def test_bare_cell_id_undercounts_but_composite_identity_is_correct(self):
        source = self._source_partition(20511, 15190)
        # The collision is real: the bare token set is only as large as the
        # bigger cohort, so a bare nunique would report 20,511 instead of 35,701.
        assert int(source["target_cell_id"].nunique()) == 20511
        assert vms.composite_identity_count(
            source, ("target_experiment_id", "target_cell_id")) == 35701
        assert vms.composite_identity_count(
            source, ("direction", "target_cell_id")) == 35701

    def test_direction_and_target_experiment_identities_agree(self):
        source = self._source_partition(500, 400)
        by_experiment = vms.composite_identity_count(
            source, ("target_experiment_id", "target_cell_id"))
        by_direction = vms.composite_identity_count(
            source, ("direction", "target_cell_id"))
        assert by_experiment == by_direction == 900

    def test_real_duplicate_within_one_target_still_reduces_the_count(self):
        source = self._source_partition(500, 400, duplicate_in_manavgat=True)
        # 499 distinct Manavgat cells + 400 Bejis cells: the intra-target
        # duplicate is a genuine defect and must NOT be masked.
        assert vms.composite_identity_count(
            source, ("target_experiment_id", "target_cell_id")) == 899

    def test_intra_target_duplicate_is_detected_per_repeat_and_direction(self):
        source = self._source_partition(500, 400, duplicate_in_manavgat=True)
        duplicates = {
            f"repeat {int(repeat_id)} / {direction}":
                int(len(group) - group["target_cell_id"].nunique())
            for (repeat_id, direction), group in source.groupby(
                ["repeat_id", "direction"])
            if len(group) != group["target_cell_id"].nunique()
        }
        assert duplicates == {f"repeat 0 / {mss.direction_token(MUGLA, MANAVGAT)}": 1}

    def test_missing_identity_column_fails_loudly(self):
        source = self._source_partition(3, 2).drop(columns=["target_experiment_id"])
        with pytest.raises(KeyError):
            vms.composite_identity_count(
                source, ("target_experiment_id", "target_cell_id"))


class TestValidatorExcludesItsOwnReport:
    """`--write-report` writes into the namespace; a later run must not scan it.

    The report echoes every check's expected/observed values, so it necessarily
    contains the excluded-AOI names (from B6) and the forbidden-vocabulary
    denylist (from K1). Scanning it would make the validator fail on its own
    output the second time it is run.
    """

    def _namespace(self, tmp_path: Path) -> Path:
        root = tmp_path / "ns"
        root.mkdir()
        (root / "report.md").write_text("clean run report\n", encoding="utf-8")
        (root / vms.VALIDATION_REPORT_NAME).write_text(
            json.dumps({"checks": [
                {"check_id": "B6", "observed": {"experiments": ["evia_2021",
                                                                "kozan_2023"]}},
                {"check_id": "K1", "expected": "no p_value or significance token"},
            ]}), encoding="utf-8")
        return root

    def test_token_scan_skips_the_validation_report(self, tmp_path):
        root = self._namespace(tmp_path)
        assert vms._scan_tokens(root, ("evia", "kozan")) == []

    def test_token_scan_still_sees_a_real_run_product(self, tmp_path):
        root = self._namespace(tmp_path)
        (root / "summary.json").write_text('{"aoi": "evia_2021"}', encoding="utf-8")
        hits = vms._scan_tokens(root, ("evia",))
        assert [hit["path"] for hit in hits] == ["summary.json"]

    def test_run_products_filter_drops_only_the_report(self):
        hits = [
            {"path": vms.VALIDATION_REPORT_NAME, "token": "significance"},
            {"path": "report.md", "token": "significance"},
        ]
        assert vms._run_products(Path("."), hits) == [
            {"path": "report.md", "token": "significance"}]
