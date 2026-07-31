"""tests/test_few_shot_recovery.py

Focused tests for `src/few_shot_recovery.py` (`few_shot_recovery.v1`).

Every test uses SMALL SYNTHETIC fixtures and a `tmp_path`-injected
`output_root` / `experiments_root`. No test reads a canonical production
artifact, no test writes to a canonical path, no test fits a full-scale model
and no test contacts Earth Engine.

Run:
    PYTHONPATH="$PWD" python -m pytest -q tests/test_few_shot_recovery.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.few_shot_recovery as fsr
from src.few_shot_recovery import FewShotRecoveryError


# =============================================================================
# Synthetic fixtures
# =============================================================================
SYNTHETIC_EXPERIMENTS = ("alpha_2021", "beta_2021", "gamma_2021")

# Small enough to fit fast, large enough that strict 5-fold spatial CV over
# 10-cell blocks is well posed and every budget in BUDGETS is feasible.
BLOCKS_PER_SIDE = 8          # 64 blocks of 10x10 cells -> 64 blocks
CELLS_PER_BLOCK_SIDE = 3     # 3x3 sampled cells inside each 10-cell block


def _make_experiment_frame(experiment_id: str, seed: int,
                           positive_block_fraction: float = 0.45) -> pd.DataFrame:
    """A synthetic Step8A-shaped modeling dataset on a 10-cell block grid."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_blocks = BLOCKS_PER_SIDE * BLOCKS_PER_SIDE
    burn_blocks = set(
        rng.choice(n_blocks, size=int(n_blocks * positive_block_fraction), replace=False)
        .tolist()
    )
    for block_row in range(BLOCKS_PER_SIDE):
        for block_col in range(BLOCKS_PER_SIDE):
            block_index = block_row * BLOCKS_PER_SIDE + block_col
            burns = block_index in burn_blocks
            for r in range(CELLS_PER_BLOCK_SIDE):
                for c in range(CELLS_PER_BLOCK_SIDE):
                    # Mixed blocks: only some cells inside a burning block burn.
                    burned = int(burns and ((r + c) % 3 == 0))
                    signal = 1.4 if burned else 0.0
                    row_500m = block_row * fsr.BLOCK_SIZE_CELLS + r
                    col_500m = block_col * fsr.BLOCK_SIZE_CELLS + c
                    rows.append({
                        # Canonical Step8A identity: cell_id is r{row}_c{col}.
                        "cell_id": f"r{row_500m}_c{col_500m}",
                        "row_500m": row_500m,
                        "col_500m": col_500m,
                        "burned": burned,
                        "valid_for_modeling": True,
                        fsr.POPULATION: True,
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


@pytest.fixture(scope="module")
def synthetic_frames() -> dict[str, pd.DataFrame]:
    return {
        experiment_id: _make_experiment_frame(experiment_id, seed=100 + index)
        for index, experiment_id in enumerate(SYNTHETIC_EXPERIMENTS)
    }


@pytest.fixture(scope="module")
def synthetic_experiments_root(tmp_path_factory, synthetic_frames) -> Path:
    root = tmp_path_factory.mktemp("experiments")
    for experiment_id, frame in synthetic_frames.items():
        directory = root / experiment_id / "step8a"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / "step8a_500m_modeling_dataset.parquet", index=False)
    return root


@pytest.fixture()
def unhashed_experiments(monkeypatch, synthetic_experiments_root):
    """Register the synthetic datasets' real digests so the hash gate passes."""
    digests = _synthetic_digests(synthetic_experiments_root)
    monkeypatch.setattr(fsr, "CANONICAL_STEP8A_SHA256", digests)
    return digests


@pytest.fixture(scope="module")
def small_context(synthetic_frames):
    """Read-only across tests: no test mutates the context."""
    return fsr.build_target_context(list(SYNTHETIC_EXPERIMENTS), None, synthetic_frames)


def _synthetic_digests(experiments_root: Path) -> dict[str, str]:
    return {
        experiment_id: fsr.sha256_file(
            experiments_root / experiment_id / "step8a"
            / "step8a_500m_modeling_dataset.parquet")
        for experiment_id in SYNTHETIC_EXPERIMENTS
    }


def _artifact_grid(root: Path) -> dict:
    """The budget/repeat grid an artifact declares about itself."""
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    scientific = config["scientific_configuration"]
    budgets = list(scientific["budgets"])
    return {
        "budgets": budgets,
        "nonzero_budgets": [k for k in budgets if k > 0],
        "n_repeats": int(scientific["n_repeats"]),
    }


def _tiny_tier_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"large_block_id": "b10_r0_c0", "block_row_count": 9, "block_positive_count": 3,
         "block_tier": fsr.TIER_BOTH},
        {"large_block_id": "b10_r0_c1", "block_row_count": 9, "block_positive_count": 2,
         "block_tier": fsr.TIER_BOTH},
        {"large_block_id": "b10_r1_c0", "block_row_count": 9, "block_positive_count": 9,
         "block_tier": fsr.TIER_POSITIVE},
        {"large_block_id": "b10_r1_c1", "block_row_count": 9, "block_positive_count": 0,
         "block_tier": fsr.TIER_NEGATIVE},
        {"large_block_id": "b10_r2_c0", "block_row_count": 9, "block_positive_count": 0,
         "block_tier": fsr.TIER_NEGATIVE},
    ])


# =============================================================================
# Contract: identity, stages, pairs
# =============================================================================
class TestContract:
    def test_analysis_id_is_deterministic(self):
        inventory = {e: {"sha256": f"{index:064d}"}
                     for index, e in enumerate(SYNTHETIC_EXPERIMENTS)}
        config_a = fsr.build_scientific_config(list(SYNTHETIC_EXPERIMENTS), inventory)
        config_b = fsr.build_scientific_config(list(SYNTHETIC_EXPERIMENTS), inventory)
        assert fsr.compute_analysis_id(config_a) == fsr.compute_analysis_id(config_b)
        assert len(fsr.compute_analysis_id(config_a)) == 64

    def test_analysis_id_changes_with_the_inputs(self):
        base = {e: {"sha256": f"{i:064d}"} for i, e in enumerate(SYNTHETIC_EXPERIMENTS)}
        other = dict(base)
        other[SYNTHETIC_EXPERIMENTS[0]] = {"sha256": "f" * 64}
        first = fsr.compute_analysis_id(
            fsr.build_scientific_config(list(SYNTHETIC_EXPERIMENTS), base))
        second = fsr.compute_analysis_id(
            fsr.build_scientific_config(list(SYNTHETIC_EXPERIMENTS), other))
        assert first != second

    def test_analysis_id_ignores_wall_clock_and_commit(self):
        inventory = {e: {"sha256": "a" * 64} for e in SYNTHETIC_EXPERIMENTS}
        config = fsr.build_scientific_config(list(SYNTHETIC_EXPERIMENTS), inventory)
        assert "created_at_utc" not in config
        assert "git_commit" not in config
        assert "package_versions" not in config

    def test_exactly_six_directed_pairs(self):
        pairs = fsr.directed_pairs(list(fsr.PRIMARY_EXPERIMENTS))
        assert len(pairs) == fsr.EXPECTED_DIRECTED_PAIRS == 6
        assert all(source != target for source, target in pairs)

    def test_selection_order_does_not_change_pairs(self):
        forward = set(fsr.directed_pairs(["a", "b", "c"]))
        reverse = set(fsr.directed_pairs(["c", "b", "a"]))
        assert forward == reverse

    def test_pair_token_is_never_sorted(self):
        assert fsr.direction_token("z_exp", "a_exp") == "z_exp_to_a_exp"
        assert fsr.direction_token("a_exp", "z_exp") == "a_exp_to_z_exp"

    def test_self_pair_is_rejected(self):
        with pytest.raises(FewShotRecoveryError):
            fsr.direction_token("alpha_2021", "alpha_2021")

    def test_duplicate_experiment_fails_closed(self):
        with pytest.raises(FewShotRecoveryError):
            fsr.resolve_experiments(["alpha_2021", "alpha_2021"])

    def test_stage_range_is_ordered(self):
        assert fsr.validate_stage_range("plan", "summarize") == ["plan", "fit", "summarize"]
        assert fsr.validate_stage_range("fit", "fit") == ["fit"]

    def test_reversed_stage_range_fails_closed(self):
        with pytest.raises(FewShotRecoveryError):
            fsr.validate_stage_range("summarize", "plan")

    def test_unknown_stage_fails_closed(self):
        with pytest.raises(FewShotRecoveryError):
            fsr.validate_stage_range("not-a-stage", "summarize")


class TestEviaExclusion:
    @pytest.mark.parametrize("experiment_id", ["evia_2021", "evia_2021_extended", "EVIA_2021"])
    def test_evia_is_rejected(self, experiment_id):
        with pytest.raises(FewShotRecoveryError):
            fsr.assert_not_excluded(experiment_id)

    def test_evia_is_rejected_by_resolve_experiments(self):
        with pytest.raises(FewShotRecoveryError):
            fsr.resolve_experiments(["manavgat_2021", "evia_2021_extended"])

    def test_evia_is_rejected_by_the_loader(self, synthetic_frames):
        with pytest.raises(FewShotRecoveryError):
            fsr.load_target_frame("evia_2021_extended", None,
                                  frame=synthetic_frames["alpha_2021"])

    def test_primary_experiments_contain_no_evia(self):
        assert not any("evia" in e for e in fsr.PRIMARY_EXPERIMENTS)


# =============================================================================
# Inputs and the hash gate
# =============================================================================
class TestHashGate:
    def test_matching_hashes_pass(self, synthetic_experiments_root, unhashed_experiments):
        inventory = fsr.build_frozen_input_inventory(
            list(SYNTHETIC_EXPERIMENTS), synthetic_experiments_root)
        result = fsr.assert_canonical_step8a_hashes(inventory, strict=True)
        assert result["all_match"] is True

    def test_hash_mismatch_fails_closed(self, monkeypatch, synthetic_experiments_root):
        monkeypatch.setattr(fsr, "CANONICAL_STEP8A_SHA256",
                            {e: "0" * 64 for e in SYNTHETIC_EXPERIMENTS})
        inventory = fsr.build_frozen_input_inventory(
            list(SYNTHETIC_EXPERIMENTS), synthetic_experiments_root)
        with pytest.raises(FewShotRecoveryError, match="hash mismatch"):
            fsr.assert_canonical_step8a_hashes(inventory, strict=True)

    def test_unregistered_experiment_fails_closed(self, monkeypatch,
                                                  synthetic_experiments_root):
        monkeypatch.setattr(fsr, "CANONICAL_STEP8A_SHA256", {})
        inventory = fsr.build_frozen_input_inventory(
            list(SYNTHETIC_EXPERIMENTS), synthetic_experiments_root)
        with pytest.raises(FewShotRecoveryError, match="No registered canonical hash"):
            fsr.assert_canonical_step8a_hashes(inventory, strict=True)

    def test_missing_dataset_fails_closed(self, tmp_path):
        with pytest.raises(FewShotRecoveryError, match="missing"):
            fsr.build_frozen_input_inventory(["alpha_2021"], tmp_path)

    def test_the_three_canonical_digests_are_registered(self):
        assert set(fsr.CANONICAL_STEP8A_SHA256) == set(fsr.PRIMARY_EXPERIMENTS)
        assert all(len(d) == 64 for d in fsr.CANONICAL_STEP8A_SHA256.values())


# =============================================================================
# Blocks, folds, inventory
# =============================================================================
class TestBlocksAndFolds:
    def test_blocks_use_the_canonical_ten_cell_utility(self, synthetic_frames):
        frame = fsr.load_target_frame("alpha_2021", None,
                                      frame=synthetic_frames["alpha_2021"])
        assert fsr.BLOCK_COLUMN in frame.columns
        assert frame[fsr.BLOCK_COLUMN].str.match(r"^b10_r\d+_c\d+$").all()

    def test_blocks_are_assigned_before_population_filtering(self, synthetic_frames):
        frame = synthetic_frames["alpha_2021"].copy()
        frame.loc[frame.index[:20], fsr.POPULATION] = False
        filtered = fsr.load_target_frame("alpha_2021", None, frame=frame)
        full = fsr.load_target_frame("alpha_2021", None, frame=synthetic_frames["alpha_2021"])
        shared = set(filtered["cell_id"]) & set(full["cell_id"])
        left = filtered.set_index("cell_id")[fsr.BLOCK_COLUMN]
        right = full.set_index("cell_id")[fsr.BLOCK_COLUMN]
        assert all(left[cell] == right[cell] for cell in shared)

    def test_block_inventory_counts_match_the_frame(self, small_context):
        rows = fsr.build_block_inventory("aid", small_context)
        whole = [r for r in rows if r["outer_fold"] == fsr.FOLD_SENTINEL_OOF]
        assert len(whole) == len(SYNTHETIC_EXPERIMENTS)
        for row in whole:
            frame = small_context[row["target_experiment"]]["frame"]
            assert row["population_rows"] == len(frame)
            assert row["total_blocks"] == frame[fsr.BLOCK_COLUMN].nunique()
            assert (row["blocks_both_classes"] + row["blocks_burned_only"]
                    + row["blocks_unburned_only"]) == row["total_blocks"]

    def test_folds_have_no_block_overlap(self, small_context):
        for experiment_id, entry in small_context.items():
            for fold_entry in entry["folds"]:
                assert not set(fold_entry["train_blocks"]) & set(fold_entry["eval_blocks"])

    def test_every_row_lands_in_exactly_one_evaluation_fold(self, small_context):
        for experiment_id, entry in small_context.items():
            n_rows = len(entry["frame"])
            coverage = np.zeros(n_rows, dtype=int)
            for fold_entry in entry["folds"]:
                coverage[fold_entry["test_idx"]] += 1
            assert (coverage == 1).all()

    def test_fold_count_is_never_silently_reduced(self, small_context):
        for entry in small_context.values():
            assert len(entry["folds"]) == fsr.N_OUTER_FOLDS

    def test_fold_assignment_is_target_only(self, small_context):
        """The two directions into a target share one fold assignment."""
        entry = small_context["alpha_2021"]
        again = fsr.build_outer_folds(entry["frame"])
        for original, repeated in zip(entry["folds"], again):
            assert np.array_equal(original["test_idx"], repeated[1])

    def test_too_small_a_population_fails_closed(self, synthetic_frames):
        tiny = synthetic_frames["alpha_2021"].head(20).copy()
        with pytest.raises(FewShotRecoveryError):
            fsr.load_target_frame("alpha_2021", None, frame=tiny)


# =============================================================================
# Selection
# =============================================================================
class TestSelection:
    def test_seed_is_deterministic(self):
        first = fsr.selection_seed("a", "b", 2, 7)
        second = fsr.selection_seed("a", "b", 2, 7)
        assert first == second and 0 <= first < 2 ** 32

    def test_seed_depends_on_direction_fold_and_repeat(self):
        base = fsr.selection_seed("a", "b", 0, 0)
        assert fsr.selection_seed("b", "a", 0, 0) != base
        assert fsr.selection_seed("a", "b", 1, 0) != base
        assert fsr.selection_seed("a", "b", 0, 1) != base

    def test_seed_is_independent_of_budget_and_family(self):
        """One ordering serves every budget, and both families see it."""
        import inspect
        signature = inspect.signature(fsr.selection_seed)
        assert list(signature.parameters) == [
            "source_id", "target_id", "outer_fold", "repeat_id"]

    def test_tier_ordering_is_both_then_positive_then_negative(self):
        tier_table = _tiny_tier_table()
        ordering = fsr.nested_block_ordering(tier_table, seed=11)
        lookup = tier_table.set_index("large_block_id")["block_tier"].to_dict()
        ranks = [fsr.TIER_ORDER.index(lookup[b]) for b in ordering]
        assert ranks == sorted(ranks)

    def test_ordering_covers_every_pool_block_once(self):
        tier_table = _tiny_tier_table()
        ordering = fsr.nested_block_ordering(tier_table, seed=3)
        assert sorted(ordering) == sorted(tier_table["large_block_id"])

    def test_ordering_is_invariant_to_input_row_order(self):
        tier_table = _tiny_tier_table()
        shuffled = tier_table.sample(frac=1.0, random_state=9).reset_index(drop=True)
        assert (fsr.nested_block_ordering(tier_table, seed=5)
                == fsr.nested_block_ordering(shuffled, seed=5))

    def test_every_prefix_contains_a_positive_block(self):
        tier_table = _tiny_tier_table()
        ordering = fsr.nested_block_ordering(tier_table, seed=17)
        lookup = tier_table.set_index("large_block_id")["block_positive_count"].to_dict()
        for k in range(1, len(ordering) + 1):
            assert sum(lookup[b] for b in ordering[:k]) > 0

    def test_selection_plan_is_nested(self, small_context, unhashed_experiments):
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        selected_rows, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        selected = pd.DataFrame(selected_rows, columns=fsr.SELECTED_BLOCK_COLUMNS)
        fsr.assert_nested_budgets(selected)

    def test_nesting_violation_is_detected(self):
        rows = []
        base = {column: "x" for column in fsr.PROVENANCE_COLUMNS}
        base["direction"] = "a_to_b"
        for budget, block in ((1, "b10_r0_c0"), (2, "b10_r9_c9")):
            for rank in range(budget):
                rows.append({**base, "outer_fold": 0, "repeat_id": 0,
                             "budget_blocks": budget, "selection_rank": rank,
                             "adaptation_block_id": f"{block}_{rank}"})
        with pytest.raises(FewShotRecoveryError, match="not nested"):
            fsr.assert_nested_budgets(pd.DataFrame(rows))

    def test_selection_never_touches_an_evaluation_block(self, small_context,
                                                         unhashed_experiments):
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        selected_rows, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        selected = pd.DataFrame(selected_rows, columns=fsr.SELECTED_BLOCK_COLUMNS)
        for target, entry in small_context.items():
            eval_blocks = {f["fold"]: set(f["eval_blocks"]) for f in entry["folds"]}
            subset = selected[selected["target_experiment"] == target]
            for _, row in subset.iterrows():
                assert row["adaptation_block_id"] not in eval_blocks[row["outer_fold"]]

    def test_every_selection_contains_a_burned_block(self, small_context,
                                                     unhashed_experiments):
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        selected_rows, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        selected = pd.DataFrame(selected_rows, columns=fsr.SELECTED_BLOCK_COLUMNS)
        totals = selected.groupby(
            ["direction", "outer_fold", "repeat_id", "budget_blocks"]
        )["block_positive_count"].sum()
        assert (totals > 0).all()

    def test_ten_repeats_for_every_nonzero_budget(self, small_context, unhashed_experiments):
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        selected_rows, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        selected = pd.DataFrame(selected_rows, columns=fsr.SELECTED_BLOCK_COLUMNS)
        repeats = selected.groupby(
            ["direction", "outer_fold", "budget_blocks"])["repeat_id"].nunique()
        assert set(repeats.unique()) == {fsr.N_REPEATS}
        assert 0 not in set(selected["budget_blocks"])

    def test_incomplete_budget_is_rejected_rather_than_reduced(self, small_context,
                                                               unhashed_experiments,
                                                               monkeypatch):
        """k is never silently lowered when the pool is too small."""
        monkeypatch.setattr(fsr, "BUDGETS", (0, 1, 2, 10_000))
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        with pytest.raises(FewShotRecoveryError, match="exceeds the training pool"):
            fsr.build_selection_plan("aid", list(SYNTHETIC_EXPERIMENTS), small_context,
                                     inventory)

    def test_plan_is_reproducible(self, small_context, unhashed_experiments):
        inventory = {e: {"sha256": d} for e, d in unhashed_experiments.items()}
        first, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        second, _ = fsr.build_selection_plan(
            "aid", list(SYNTHETIC_EXPERIMENTS), small_context, inventory)
        assert pd.DataFrame(first).equals(pd.DataFrame(second))


# =============================================================================
# Fit registry
# =============================================================================
class TestFitRegistry:
    def test_repeated_identity_is_never_refitted(self):
        registry = fsr.FitRegistry()
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return np.array([0.5])

        for _ in range(5):
            registry.get_or_fit("raw|a|b|thermal", fsr.CONDITION_RAW, compute)
        assert calls["n"] == 1
        assert registry.fit_count == 1
        assert registry.reuse_count == 4

    def test_distinct_identities_are_fitted_separately(self):
        registry = fsr.FitRegistry()
        registry.get_or_fit("a", fsr.CONDITION_FEWSHOT, lambda: 1)
        registry.get_or_fit("b", fsr.CONDITION_FEWSHOT, lambda: 2)
        assert registry.fit_count == 2

    def test_raw_identity_is_fold_and_repeat_independent(self):
        base = fsr.fit_identity(fsr.CONDITION_RAW, family="thermal",
                                source_id="a", target_id="b", outer_fold=0,
                                budget=0, repeat_id=0)
        other = fsr.fit_identity(fsr.CONDITION_RAW, family="thermal",
                                 source_id="a", target_id="b", outer_fold=4,
                                 budget=0, repeat_id=9)
        assert base == other

    def test_ceiling_identity_is_source_independent(self):
        first = fsr.fit_identity(fsr.CONDITION_CEILING, family="thermal",
                                 source_id="a", target_id="t", outer_fold=2)
        second = fsr.fit_identity(fsr.CONDITION_CEILING, family="thermal",
                                  source_id="zzz", target_id="t", outer_fold=2)
        assert first == second

    def test_ceiling_identity_separates_folds_and_families(self):
        base = fsr.fit_identity(fsr.CONDITION_CEILING, family="thermal",
                                target_id="t", outer_fold=2)
        assert base != fsr.fit_identity(fsr.CONDITION_CEILING, family="thermal",
                                        target_id="t", outer_fold=3)
        assert base != fsr.fit_identity(fsr.CONDITION_CEILING, family="baseline",
                                        target_id="t", outer_fold=2)

    def test_few_shot_identity_separates_every_axis(self):
        keys = {
            fsr.fit_identity(fsr.CONDITION_FEWSHOT, family=family, source_id="a",
                             target_id="b", outer_fold=fold, budget=budget,
                             repeat_id=repeat)
            for family in fsr.MODEL_FAMILIES
            for fold in range(fsr.N_OUTER_FOLDS)
            for budget in fsr.NONZERO_BUDGETS
            for repeat in range(fsr.N_REPEATS)
        }
        assert len(keys) == (len(fsr.MODEL_FAMILIES) * fsr.N_OUTER_FOLDS
                             * len(fsr.NONZERO_BUDGETS) * fsr.N_REPEATS)

    def test_expected_fit_count_is_three_thousand_six_hundred_and_forty_two(self):
        expected = fsr.expected_unique_fit_count(6, 3)
        assert expected == {
            "raw_fits": 12, "few_shot_fits": 3600, "ceiling_fits": 30,
            "unique_fits": 3642,
        }

    def test_accounting_reports_reuse_ratios(self):
        registry = fsr.FitRegistry()
        for _ in range(5):
            registry.get_or_fit("raw|a|b|thermal", fsr.CONDITION_RAW, lambda: 1)
        for _ in range(2):
            registry.get_or_fit("ceiling|t|0|thermal", fsr.CONDITION_CEILING, lambda: 1)
        accounting = registry.accounting()
        assert accounting["raw_fits"] == 1
        assert accounting["raw_reuse_per_fit"] == 5.0
        assert accounting["ceiling_reuse_per_fit"] == 2.0

    def test_release_frees_memory_without_changing_the_count(self):
        registry = fsr.FitRegistry()
        registry.get_or_fit("raw|a|b|thermal", fsr.CONDITION_RAW, lambda: np.zeros(10))
        released = registry.release("raw|a|b|")
        assert released == 1
        assert registry.fit_count == 1
        assert len(registry.identities()) == 1


# =============================================================================
# Metrics, orientation and recovery arithmetic
# =============================================================================
class TestRecoveryArithmetic:
    def test_brier_is_negated_and_others_are_not(self):
        assert fsr.oriented("brier_score", 0.25) == -0.25
        assert fsr.oriented("roc_auc", 0.8) == 0.8
        assert fsr.oriented("pr_auc", 0.3) == 0.3

    def test_orientation_label_is_correct(self):
        assert fsr.metric_orientation("brier_score") == fsr.ORIENTATION_NEGATED
        assert fsr.metric_orientation("roc_auc") == fsr.ORIENTATION_HIGHER

    def test_recovery_matches_the_closed_form(self):
        result = fsr.recovery_quantities(0.50, 0.70, 0.90)
        assert result["absolute_recovery"] == pytest.approx(0.20)
        assert result["ceiling_gap"] == pytest.approx(0.40)
        assert result["recovery_fraction"] == pytest.approx(0.5)
        assert result["recovery_fraction_status"] == fsr.STATUS_INTERPRETABLE

    def test_recovery_below_zero_is_preserved(self):
        result = fsr.recovery_quantities(0.60, 0.50, 0.90)
        assert result["recovery_fraction"] == pytest.approx(-0.3333333333333333)
        assert result["recovery_fraction"] < 0
        assert result["recovery_negative"] is True

    def test_recovery_above_one_is_preserved(self):
        result = fsr.recovery_quantities(0.50, 0.95, 0.90)
        assert result["recovery_fraction"] == pytest.approx(1.125)
        assert result["recovery_fraction"] > 1
        assert result["recovery_above_ceiling"] is True

    def test_recovery_is_not_clipped_or_absolute_valued(self):
        negative = fsr.recovery_quantities(0.60, 0.20, 0.90)["recovery_fraction"]
        assert negative == pytest.approx(-1.3333333333333333)
        assert negative != abs(negative)

    def test_denominator_near_zero_is_undefined(self):
        result = fsr.recovery_quantities(0.5, 0.7, 0.5 + 1e-9)
        assert result["recovery_fraction"] is None
        assert result["denominator_near_zero"] is True
        assert result["recovery_fraction_status"] == fsr.STATUS_DEGENERATE

    def test_exactly_at_the_threshold_stays_defined(self):
        result = fsr.recovery_quantities(0.5, 0.6, 0.5 + 1e-5)
        assert result["recovery_fraction"] is not None

    def test_ceiling_at_or_below_raw_is_flagged(self):
        result = fsr.recovery_quantities(0.80, 0.85, 0.60)
        assert result["ceiling_not_above_raw"] is True
        assert result["recovery_fraction_status"] == fsr.STATUS_CEILING_NOT_ABOVE_RAW
        assert result["recovery_fraction"] is not None  # signed value preserved

    def test_brier_orientation_flips_the_direction_of_improvement(self):
        """A LOWER Brier is better, so recovery must come out positive."""
        raw, fewshot, ceiling = 0.20, 0.15, 0.10
        result = fsr.recovery_quantities(
            fsr.oriented("brier_score", raw),
            fsr.oriented("brier_score", fewshot),
            fsr.oriented("brier_score", ceiling),
        )
        assert result["absolute_recovery"] > 0
        assert result["recovery_fraction"] == pytest.approx(0.5)

    def test_missing_values_are_undefined_not_zero(self):
        assert fsr.recovery_quantities(None, 0.5, 0.9)["recovery_fraction"] is None
        assert fsr.recovery_quantities(0.5, None, 0.9)["recovery_fraction"] is None


class TestSelectionInterval:
    def test_matches_numpy_percentile(self):
        values = [0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.15, 0.35, 0.25, 0.45]
        interval = fsr.selection_interval(values)
        assert interval["selection_interval_lower"] == pytest.approx(
            float(np.percentile(values, 2.5, method="linear")))
        assert interval["selection_interval_upper"] == pytest.approx(
            float(np.percentile(values, 97.5, method="linear")))
        assert interval["selection_median"] == pytest.approx(float(np.median(values)))

    def test_single_realisation_is_degenerate(self):
        interval = fsr.selection_interval([0.42])
        assert interval["selection_median"] == 0.42
        assert interval["selection_interval_lower"] == 0.42
        assert interval["selection_interval_upper"] == 0.42
        assert interval["n_repeats_observed"] == 1

    def test_interval_lies_inside_the_observed_range(self):
        values = [0.1, 0.9, 0.5, 0.4, 0.6, 0.2, 0.8, 0.3, 0.7, 0.45]
        interval = fsr.selection_interval(values)
        assert interval["selection_min"] <= interval["selection_interval_lower"]
        assert interval["selection_interval_upper"] <= interval["selection_max"]

    def test_none_values_are_dropped(self):
        interval = fsr.selection_interval([None, 0.5, float("nan"), 0.7])
        assert interval["n_repeats_observed"] == 2

    def test_interval_names_avoid_confidence_vocabulary(self):
        keys = set(fsr.selection_interval([0.1, 0.2]))
        assert "selection_interval_lower" in keys and "selection_interval_upper" in keys
        assert not any("ci" == key or key.startswith("ci_") for key in keys)


class TestOofCoverage:
    def test_full_coverage_passes(self):
        fsr.assert_full_oof_coverage(np.ones(10, dtype=int), "ctx")

    def test_a_gap_fails_closed(self):
        coverage = np.ones(10, dtype=int)
        coverage[3] = 0
        with pytest.raises(FewShotRecoveryError, match="coverage violated"):
            fsr.assert_full_oof_coverage(coverage, "ctx")

    def test_a_duplicate_fails_closed(self):
        coverage = np.ones(10, dtype=int)
        coverage[7] = 2
        with pytest.raises(FewShotRecoveryError, match="coverage violated"):
            fsr.assert_full_oof_coverage(coverage, "ctx")


# =============================================================================
# Model contract
# =============================================================================
class TestModelContract:
    def test_hyperparameters_are_the_canonical_ones(self):
        classifier = fsr.build_classifier(fsr.MODEL_NAME, fsr.ESTIMATOR_SEED)
        params = classifier.get_params(deep=False)
        assert params["n_estimators"] == 300
        assert params["min_samples_leaf"] == 3
        assert params["class_weight"] == "balanced"
        assert params["random_state"] == 42

    def test_config_declares_the_pre_existing_class_weighting(self):
        config = fsr.build_scientific_config(
            list(SYNTHETIC_EXPERIMENTS),
            {e: {"sha256": "a" * 64} for e in SYNTHETIC_EXPERIMENTS})
        weighting = config["model"]["pre_existing_class_weighting"]
        assert weighting["present"] is True
        assert weighting["mechanism"] == "class_weight='balanced'"
        assert config["model"]["sample_weight_argument_used"] is False
        assert config["model"]["oversampling_performed"] is False
        assert config["model"]["tuning_performed"] is False

    def test_fit_never_passes_sample_weight(self):
        source = Path(fsr.__file__).read_text(encoding="utf-8")
        assert "sample_weight=" not in source

    def test_feature_lists_are_the_canonical_constants(self):
        from src.step8b_train_baseline_vs_thermal_model import (
            BASELINE_FEATURES, THERMAL_MODEL_FEATURES)
        assert fsr.FEATURE_LISTS["baseline"] == list(BASELINE_FEATURES)
        assert fsr.FEATURE_LISTS["thermal"] == list(THERMAL_MODEL_FEATURES)

    def test_no_forbidden_feature_reaches_the_model(self):
        for features in fsr.FEATURE_LISTS.values():
            fsr.check_no_forbidden_features(features)

    def test_module_does_not_import_earth_engine(self):
        source = Path(fsr.__file__).read_text(encoding="utf-8")
        assert "import ee" not in source
        assert "gee_utils" not in source


# =============================================================================
# End-to-end on synthetic data
# =============================================================================
def _shrink(patcher, experiments_root: Path) -> None:
    """Shrink only the loop SIZES, never a contract rule.

    Budgets, repeats and forest size come down so the suite runs in seconds.
    Nesting, tiering, the firewall, the fit identities, the OOF coverage
    assertion and the recovery arithmetic are all the production code path.
    """
    patcher.setattr(fsr, "BUDGETS", (0, 1, 2))
    patcher.setattr(fsr, "NONZERO_BUDGETS", (1, 2))
    patcher.setattr(fsr, "N_REPEATS", 2)
    patcher.setattr(fsr, "CANONICAL_STEP8A_SHA256", _synthetic_digests(experiments_root))

    def small_forest(model_name, random_state):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=8, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=1)

    patcher.setattr("src.step8b_train_baseline_vs_thermal_model.build_classifier",
                    small_forest)


@pytest.fixture(scope="module")
def tiny_run(synthetic_experiments_root, tmp_path_factory):
    """One complete plan->fit->summarize run, shared by the read-only tests."""
    output_root = tmp_path_factory.mktemp("tiny_run") / "outputs"
    with pytest.MonkeyPatch.context() as patcher:
        _shrink(patcher, synthetic_experiments_root)
        result = fsr.run_analysis(
            experiments=list(SYNTHETIC_EXPERIMENTS),
            from_stage="plan", to_stage="summarize",
            output_root=output_root, experiments_root=synthetic_experiments_root,
        )
    return result, fsr.analysis_root(result["analysis_id"], output_root), output_root


@pytest.fixture()
def mutable_run(tiny_run, monkeypatch, synthetic_experiments_root, tmp_path):
    """A private COPY of the shared run, for tests that mutate the namespace.

    Keeps the shrink patches active so `run_analysis` can be re-entered against
    the same analysis_id.
    """
    result, _, output_root = tiny_run
    copied_root = tmp_path / "outputs"
    shutil.copytree(output_root, copied_root)
    _shrink(monkeypatch, synthetic_experiments_root)
    return result, fsr.analysis_root(result["analysis_id"], copied_root), copied_root


class TestEndToEnd:
    def test_every_required_output_exists(self, tiny_run):
        _, root, _ = tiny_run
        for relative in (
            "config.json", "input_hashes.json", "target_block_inventory.csv",
            "direction_budget_feasibility.csv", "selected_blocks.parquet",
            "repeat_metrics.csv", "recovery_curve.csv", "summary.json", "report.md",
            "manifest.json", "stages/plan.json", "stages/fit.json",
            "stages/summarize.json",
        ):
            assert (root / relative).exists(), relative
        assert (root / fsr.OOF_PREDICTIONS_DIRNAME).is_dir()

    def test_oof_predictions_read_as_one_logical_dataset(self, tiny_run):
        _, root, _ = tiny_run
        frame = pd.read_parquet(root / fsr.OOF_PREDICTIONS_DIRNAME)
        assert set(frame["direction"]) == {
            fsr.direction_token(s, t)
            for s, t in fsr.directed_pairs(list(SYNTHETIC_EXPERIMENTS))
        }
        assert len(frame["direction"].unique()) == 6

    def test_predictions_are_partitioned_per_direction(self, tiny_run):
        _, root, _ = tiny_run
        parts = sorted((root / fsr.OOF_PREDICTIONS_DIRNAME).glob("*.parquet"))
        assert len(parts) == 6

    def test_complete_oof_coverage_exactly_once_per_repeat(self, tiny_run):
        _, root, _ = tiny_run
        frame = pd.read_parquet(root / fsr.OOF_PREDICTIONS_DIRNAME)
        for _, group in frame.groupby(
            ["direction", "condition", "budget_blocks", "repeat_id"]
        ):
            assert not group["cell_id"].duplicated().any()
            assert not group[["baseline_probability", "thermal_probability"]].isna().any().any()

    def test_no_evaluation_block_is_ever_an_adaptation_block(self, tiny_run):
        _, root, _ = tiny_run
        selected = pd.read_parquet(root / "selected_blocks.parquet")
        oof = pd.read_parquet(root / fsr.OOF_PREDICTIONS_DIRNAME)
        raw = oof[oof["condition"] == fsr.CONDITION_RAW]
        eval_blocks = {
            (direction, int(fold)): set(group["evaluation_block_id"])
            for (direction, fold), group in raw.groupby(["direction", "outer_fold"])
        }
        for _, row in selected.iterrows():
            key = (row["direction"], int(row["outer_fold"]))
            assert row["adaptation_block_id"] not in eval_blocks[key]

    def test_fit_accounting_matches_the_expected_identities(self, tiny_run):
        _, root, _ = tiny_run
        marker = json.loads((root / "stages" / "fit.json").read_text(encoding="utf-8"))
        accounting = marker["fit_accounting"]
        # Read the grid back from the artifact: the shrink patches are no longer
        # active, and the artifact must describe its own contract.
        grid = _artifact_grid(root)
        n_directions, n_targets = 6, 3
        expected_raw = n_directions * len(fsr.MODEL_FAMILIES)
        expected_ceiling = n_targets * fsr.N_OUTER_FOLDS * len(fsr.MODEL_FAMILIES)
        expected_fewshot = (n_directions * len(fsr.MODEL_FAMILIES) * fsr.N_OUTER_FOLDS
                            * len(grid["nonzero_budgets"]) * grid["n_repeats"])
        assert accounting["raw_fits"] == expected_raw
        assert accounting["ceiling_fits"] == expected_ceiling
        assert accounting["few_shot_fits"] == expected_fewshot
        assert accounting["unique_fits"] == (
            expected_raw + expected_ceiling + expected_fewshot)

    def test_raw_fits_are_reused_across_folds(self, tiny_run):
        _, root, _ = tiny_run
        marker = json.loads((root / "stages" / "fit.json").read_text(encoding="utf-8"))
        assert marker["fit_accounting"]["raw_reuse_per_fit"] == float(fsr.N_OUTER_FOLDS)

    def test_ceiling_fits_are_reused_across_source_directions(self, tiny_run):
        _, root, _ = tiny_run
        marker = json.loads((root / "stages" / "fit.json").read_text(encoding="utf-8"))
        assert marker["fit_accounting"]["ceiling_reuse_per_fit"] == 2.0

    def test_raw_is_source_only_and_ceiling_is_target_only(self, tiny_run):
        _, root, _ = tiny_run
        metrics = pd.read_csv(root / "repeat_metrics.csv")
        fold_rows = metrics[metrics["evaluation_level"] == fsr.EVALUATION_LEVEL_FOLD]
        raw = fold_rows[fold_rows["condition"] == fsr.CONDITION_RAW]
        ceiling = fold_rows[fold_rows["condition"] == fsr.CONDITION_CEILING]
        assert (raw["n_train_target_rows"] == 0).all()
        assert (raw["budget_blocks"] == 0).all()
        assert (ceiling["n_train_source_rows"] == 0).all()

    def test_few_shot_uses_the_full_source_plus_adaptation_rows(self, tiny_run):
        _, root, _ = tiny_run
        metrics = pd.read_csv(root / "repeat_metrics.csv")
        rows = metrics[(metrics["evaluation_level"] == fsr.EVALUATION_LEVEL_FOLD)
                       & (metrics["condition"] == fsr.CONDITION_FEWSHOT)]
        assert (rows["n_train_rows"]
                == rows["n_train_source_rows"] + rows["adaptation_row_count"]).all()
        assert (rows["adaptation_positive_count"] > 0).all()

    def test_recovery_curve_has_one_row_per_direction_family_metric_budget(self, tiny_run):
        _, root, _ = tiny_run
        curve = pd.read_csv(root / "recovery_curve.csv")
        grid = _artifact_grid(root)
        expected = (6 * len(fsr.MODEL_FAMILIES) * len(fsr.METRICS)
                    * len(grid["budgets"]))
        assert len(curve) == expected

    def test_curve_recovery_fraction_reproduces_the_closed_form(self, tiny_run):
        _, root, _ = tiny_run
        curve = pd.read_csv(root / "recovery_curve.csv")
        for _, row in curve.iterrows():
            expected = fsr.recovery_quantities(
                float(row["raw_oriented"]), float(row["fewshot_oriented"]),
                float(row["ceiling_oriented"]))
            if expected["recovery_fraction"] is None:
                assert pd.isna(row["recovery_fraction"])
            else:
                assert float(row["recovery_fraction"]) == pytest.approx(
                    expected["recovery_fraction"], abs=1e-12)

    def test_brier_rows_keep_the_natural_sign_and_negate_only_the_oriented_value(
            self, tiny_run):
        _, root, _ = tiny_run
        curve = pd.read_csv(root / "recovery_curve.csv")
        brier = curve[curve["metric"] == "brier_score"]
        assert (brier["metric_orientation"] == fsr.ORIENTATION_NEGATED).all()
        assert (brier["raw_value"] >= 0).all()
        assert np.allclose(brier["raw_oriented"], -brier["raw_value"])
        others = curve[curve["metric"] != "brier_score"]
        assert np.allclose(others["raw_oriented"], others["raw_value"])

    def test_zero_budget_carries_one_realisation(self, tiny_run):
        _, root, _ = tiny_run
        curve = pd.read_csv(root / "recovery_curve.csv")
        zero = curve[curve["budget_blocks"] == 0]
        assert (zero["n_repeats"] == 1).all()
        assert np.allclose(zero["selection_interval_lower"],
                           zero["selection_interval_upper"])

    def test_nonzero_budgets_carry_every_repeat(self, tiny_run):
        _, root, _ = tiny_run
        curve = pd.read_csv(root / "recovery_curve.csv")
        grid = _artifact_grid(root)
        nonzero = curve[curve["budget_blocks"] > 0]
        assert (nonzero["n_repeats"] == grid["n_repeats"]).all()
        selected = pd.read_parquet(root / "selected_blocks.parquet")
        repeats = selected.groupby(
            ["direction", "outer_fold", "budget_blocks"])["repeat_id"].nunique()
        assert set(repeats.unique()) == {grid["n_repeats"]}

    def test_no_forbidden_uncertainty_vocabulary_in_the_outputs(self, tiny_run):
        _, root, _ = tiny_run
        allowed = ("is not a confidence interval",
                   "supports no claim about statistical support",
                   "NOT a selection interval",
                   "no p-value is reported",
                   "p_values_produced", "forbidden_terms_source")
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".json", ".csv"}:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if any(marker in line for marker in allowed):
                    continue
                lowered = line.lower()
                for term in ("confidence interval", "95% ci", "p-value", "significant"):
                    assert term not in lowered, f"{path}: {line[:120]}"

    def test_no_evia_row_anywhere(self, tiny_run):
        """Evia may be NAMED as excluded; it may never appear as data."""
        _, root, _ = tiny_run
        allowed = ("is excluded by design", "excluded_experiments",
                   "out_of_scope_for_this_frozen_analysis",
                   "high_prevalence_different_regime")
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".csv"}:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if any(marker in line for marker in allowed):
                    continue
                assert "evia" not in line.lower(), f"{path}: {line[:120]}"

    def test_no_evia_appears_as_data(self, tiny_run):
        _, root, _ = tiny_run
        for relative in ("selected_blocks.parquet",):
            frame = pd.read_parquet(root / relative)
            for column in ("source_experiment", "target_experiment", "direction"):
                assert not frame[column].astype(str).str.contains("evia").any()
        oof = pd.read_parquet(root / fsr.OOF_PREDICTIONS_DIRNAME)
        assert not oof["direction"].astype(str).str.contains("evia").any()

    def test_manifest_exposes_the_predictions_as_one_logical_dataset(self, tiny_run):
        _, root, _ = tiny_run
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        logical = manifest["logical_datasets"][fsr.OOF_PREDICTIONS_DIRNAME]
        assert logical["kind"] == "partitioned_parquet_dataset"
        assert len(logical["parts"]) == 6
        assert logical["dataset_sha256"]

    def test_manifest_hashes_match_the_files_on_disk(self, tiny_run):
        _, root, _ = tiny_run
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = root / entry["path"]
            assert path.is_file()
            assert fsr.sha256_file(path) == entry["sha256"]

    def test_every_written_path_is_inside_the_namespace(self, tiny_run):
        result, root, output_root = tiny_run
        diagnostics = fsr.diagnostics_root(output_root)
        written = [p for p in diagnostics.rglob("*") if p.is_file()]
        assert written
        assert all(str(p).startswith(str(root)) for p in written)

    def test_analysis_id_matches_the_directory(self, tiny_run):
        result, root, _ = tiny_run
        assert root.name == result["analysis_id"]
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        assert fsr.compute_analysis_id(
            config["scientific_configuration"]) == result["analysis_id"]


# =============================================================================
# Write safety: dry-run, resume, force
# =============================================================================
class TestWriteSafety:
    def test_dry_run_writes_nothing_and_fits_nothing(
            self, synthetic_experiments_root, unhashed_experiments, tmp_path):
        output_root = tmp_path / "outputs"
        result = fsr.run_analysis(
            experiments=list(SYNTHETIC_EXPERIMENTS), dry_run=True,
            output_root=output_root, experiments_root=synthetic_experiments_root)
        assert result["ran"] is False
        assert result["dry_run"] is True
        assert result["fit_performed"] is False
        assert result["files_written"] == []
        assert result["earth_engine_used"] is False
        assert not output_root.exists()

    def test_dry_run_reports_the_planned_layout_and_fit_budget(
            self, synthetic_experiments_root, unhashed_experiments, tmp_path):
        result = fsr.run_analysis(
            experiments=list(SYNTHETIC_EXPERIMENTS), dry_run=True,
            output_root=tmp_path / "outputs",
            experiments_root=synthetic_experiments_root)
        assert result["directed_pair_count"] == 6
        assert "config.json" in result["planned_outputs"]
        assert result["expected_fit_accounting"]["unique_fits"] == 3642

    def test_dry_run_still_enforces_the_hash_gate(
            self, monkeypatch, synthetic_experiments_root, tmp_path):
        monkeypatch.setattr(fsr, "CANONICAL_STEP8A_SHA256",
                            {e: "0" * 64 for e in SYNTHETIC_EXPERIMENTS})
        with pytest.raises(FewShotRecoveryError):
            fsr.run_analysis(experiments=list(SYNTHETIC_EXPERIMENTS), dry_run=True,
                             output_root=tmp_path / "outputs",
                             experiments_root=synthetic_experiments_root)

    def test_existing_namespace_is_not_overwritten(self, mutable_run,
                                                   synthetic_experiments_root):
        result, root, output_root = mutable_run
        with pytest.raises(FewShotRecoveryError, match="already exists"):
            fsr.run_analysis(
                experiments=list(SYNTHETIC_EXPERIMENTS),
                output_root=output_root, experiments_root=synthetic_experiments_root)

    def test_force_quarantines_and_never_deletes(self, mutable_run):
        result, root, output_root = mutable_run
        marker = root / "config.json"
        assert marker.is_file()
        destination = fsr.quarantine_namespace(result["analysis_id"], output_root)
        assert destination is not None
        assert not root.exists()
        quarantined = Path(destination)
        assert quarantined.is_dir()
        assert (quarantined / "config.json").is_file()
        assert fsr.QUARANTINE_DIRNAME in destination

    def test_quarantine_of_a_missing_namespace_is_a_no_op(self, tmp_path):
        assert fsr.quarantine_namespace("a" * 64, tmp_path / "outputs") is None

    def test_atomic_write_leaves_no_temporary_file(self, tmp_path):
        target = tmp_path / "nested" / "doc.json"
        fsr._atomic_write_text(target, '{"a": 1}\n')
        assert target.read_text(encoding="utf-8") == '{"a": 1}\n'
        assert not list(tmp_path.rglob(".*tmp"))

    def test_atomic_write_replaces_in_place(self, tmp_path):
        target = tmp_path / "doc.json"
        fsr._atomic_write_text(target, "first\n")
        fsr._atomic_write_text(target, "second\n")
        assert target.read_text(encoding="utf-8") == "second\n"
        assert len(list(tmp_path.iterdir())) == 1

    def test_atomic_parquet_write_leaves_no_temporary_file(self, tmp_path):
        target = tmp_path / "frame.parquet"
        fsr._atomic_write_parquet(target, pd.DataFrame({"a": [1, 2]}))
        assert pd.read_parquet(target)["a"].tolist() == [1, 2]
        assert not list(tmp_path.rglob(".*tmp"))

    def test_writing_outside_the_namespace_fails_closed(self, tmp_path):
        root = tmp_path / "namespace"
        root.mkdir()
        with pytest.raises(FewShotRecoveryError, match="outside the analysis namespace"):
            fsr.assert_inside_namespace(tmp_path / "elsewhere.json", root)


class TestResume:
    def test_complete_stage_is_reusable(self, mutable_run):
        result, root, output_root = mutable_run
        state = fsr.verify_stage_complete(result["analysis_id"], "plan", output_root)
        assert state["complete"] is True

    def test_missing_marker_is_not_reusable(self, mutable_run):
        result, root, output_root = mutable_run
        (root / "stages" / "plan.json").unlink()
        state = fsr.verify_stage_complete(result["analysis_id"], "plan", output_root)
        assert state["complete"] is False
        assert "no stage marker" in state["reason"]

    def test_hash_drift_makes_a_stage_unreusable(self, mutable_run):
        result, root, output_root = mutable_run
        (root / "target_block_inventory.csv").write_text("tampered\n", encoding="utf-8")
        state = fsr.verify_stage_complete(result["analysis_id"], "plan", output_root)
        assert state["complete"] is False
        assert "hash drift" in state["reason"]

    def test_deleted_artifact_makes_a_stage_unreusable(self, mutable_run):
        result, root, output_root = mutable_run
        (root / "selected_blocks.parquet").unlink()
        state = fsr.verify_stage_complete(result["analysis_id"], "plan", output_root)
        assert state["complete"] is False

    def test_partial_direction_partition_is_rejected(self, mutable_run):
        """A partition present on disk but not hash-bound is never accepted."""
        result, root, output_root = mutable_run
        directions = [fsr.direction_token(s, t)
                      for s, t in fsr.directed_pairs(list(SYNTHETIC_EXPERIMENTS))]
        target = directions[0]
        path = root / fsr.OOF_PREDICTIONS_DIRNAME / f"part-{target}.parquet"
        truncated = pd.read_parquet(path).head(5)
        fsr._atomic_write_parquet(path, truncated)
        assert fsr.verify_direction_partition(result["analysis_id"], target,
                                              output_root) is False

    def test_untracked_partition_is_rejected(self, mutable_run):
        result, root, output_root = mutable_run
        assert fsr.verify_direction_partition(
            result["analysis_id"], "ghost_to_nowhere", output_root) is False

    def test_complete_partition_is_accepted(self, mutable_run):
        result, root, output_root = mutable_run
        direction = fsr.direction_token(*fsr.directed_pairs(
            list(SYNTHETIC_EXPERIMENTS))[0])
        assert fsr.verify_direction_partition(result["analysis_id"], direction,
                                              output_root) is True

    def test_stage_prerequisite_is_enforced(self, synthetic_experiments_root,
                                            unhashed_experiments, tmp_path):
        with pytest.raises(FewShotRecoveryError, match="requires a complete"):
            fsr.run_analysis(
                experiments=list(SYNTHETIC_EXPERIMENTS),
                from_stage="summarize", to_stage="summarize",
                output_root=tmp_path / "outputs",
                experiments_root=synthetic_experiments_root)


# =============================================================================
# Runner passthrough
# =============================================================================
class TestRunnerPassthrough:
    def test_runner_forwards_every_argument(self):
        from unittest.mock import patch
        from scripts import run_few_shot_recovery as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(experiments=["a", "b"], from_stage="plan", to_stage="fit",
                        dry_run=True, resume=True, force=False,
                        output_root="/tmp/o", experiments_root="/tmp/e")
        kwargs = mocked.call_args.kwargs
        assert kwargs["experiments"] == ["a", "b"]
        assert kwargs["from_stage"] == "plan"
        assert kwargs["to_stage"] == "fit"
        assert kwargs["dry_run"] is True
        assert kwargs["resume"] is True
        assert str(kwargs["output_root"]) == "/tmp/o"
        assert str(kwargs["experiments_root"]) == "/tmp/e"

    def test_runner_defaults_roots_to_none(self):
        from unittest.mock import patch
        from scripts import run_few_shot_recovery as runner

        with patch.object(runner, "run_analysis", return_value={"ran": False}) as mocked:
            runner.main(dry_run=True)
        kwargs = mocked.call_args.kwargs
        assert kwargs["output_root"] is None
        assert kwargs["experiments_root"] is None
        assert kwargs["from_stage"] == "plan"
        assert kwargs["to_stage"] == "summarize"

    def test_runner_parser_exposes_the_required_flags(self):
        from scripts import run_few_shot_recovery as runner

        args = runner.build_parser().parse_args([
            "--from-stage", "fit", "--to-stage", "summarize", "--dry-run",
            "--resume", "--force", "--output-root", "/tmp/o",
            "--experiments-root", "/tmp/e",
        ])
        assert args.from_stage == "fit"
        assert args.to_stage == "summarize"
        assert args.dry_run and args.resume and args.force

    def test_runner_rejects_an_unknown_stage(self):
        from scripts import run_few_shot_recovery as runner

        with pytest.raises(SystemExit):
            runner.build_parser().parse_args(["--from-stage", "not-a-stage"])


# =============================================================================
# Validator
# =============================================================================
class TestValidator:
    def test_dry_run_validation_passes_on_the_frozen_contract(self):
        from scripts import validate_few_shot_recovery as validator

        payload = validator.run_validation(dry_run=True)
        assert payload["validator_mode"] == "dry-run"
        assert payload["counts"]["FAIL"] == 0, [
            check for check in payload["checks"] if check["status"] == "FAIL"]

    def test_validator_reports_a_missing_namespace(self, tmp_path):
        from scripts import validate_few_shot_recovery as validator

        payload = validator.run_validation(
            analysis_id="a" * 64, output_root=tmp_path / "outputs")
        assert payload["overall_status"] == "FAIL"

    def test_validator_passes_on_a_produced_artifact(self, tiny_run,
                                                     synthetic_experiments_root):
        from scripts import validate_few_shot_recovery as validator

        result, root, output_root = tiny_run
        payload = validator.run_validation(
            analysis_id=result["analysis_id"],
            experiments=list(SYNTHETIC_EXPERIMENTS),
            output_root=output_root,
            experiments_root=synthetic_experiments_root)
        failures = [check for check in payload["checks"] if check["status"] == "FAIL"]
        # The synthetic run has no frozen 10-cell ceiling anchor and a reduced
        # budget/repeat grid, so the fit-total and ceiling checks are expected to
        # skip or fail; every structural check must pass.
        structural = [check for check in failures
                      if not check["check_id"].startswith(("FSR-35", "FSR-43", "FSR-24"))]
        assert not structural, structural
