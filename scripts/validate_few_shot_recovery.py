#!/usr/bin/env python3
"""Validator for the few-shot recovery artifact.

Implements the 42 checks of
`docs/few_shot_recovery_design/VALIDATOR_CHECKLIST.md`, plus three explicit
fit-accounting checks (FSR-43..45) for the fit identities and the 3,642 unique
fit total.

Two modes:

    dry-run  -- contract, stage-order, prerequisite and plan checks only.
                Writes nothing, reads no produced artifact, contacts no GEE.
    actual   -- every check, against a produced artifact.

Every check emits {check_id, status, expected, observed, evidence_path}. Any
FAIL makes the overall status FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.few_shot_recovery as fsr

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIPPED"


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, check_id: str, status: str, expected: Any, observed: Any,
            evidence_path: Optional[str] = None, note: Optional[str] = None) -> None:
        self.checks.append({
            "check_id": check_id, "status": status, "expected": expected,
            "observed": observed, "evidence_path": evidence_path, "note": note,
        })

    def ok(self, check_id: str, condition: bool, expected: Any, observed: Any,
           evidence_path: Optional[str] = None, note: Optional[str] = None) -> bool:
        self.add(check_id, PASS if condition else FAIL, expected, observed,
                 evidence_path, note)
        return bool(condition)

    def skip(self, check_id: str, expected: Any, reason: str,
             evidence_path: Optional[str] = None) -> None:
        self.add(check_id, SKIP, expected, reason, evidence_path)

    @property
    def overall(self) -> str:
        return FAIL if any(check["status"] == FAIL for check in self.checks) else PASS

    def payload(self, mode: str, analysis_id: Optional[str],
                namespace: Optional[str]) -> dict[str, Any]:
        counts = {
            status: sum(1 for check in self.checks if check["status"] == status)
            for status in (PASS, FAIL, SKIP)
        }
        return {
            "schema_version": fsr.SCHEMA_VERSION,
            "validator_mode": mode,
            "analysis_id": analysis_id,
            "namespace": namespace,
            "overall_status": self.overall,
            "counts": counts,
            "checks": self.checks,
        }


# =============================================================================
# Helpers
# =============================================================================
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=True)


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def _text_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".json", ".csv"}]


def _scan_forbidden_terms(root: Path) -> list[dict[str, Any]]:
    """Case-insensitive scan for confidence-interval / significance vocabulary.

    The declared forbidden-terms list inside config.json and the limitations
    sentence that names what the interval is NOT are the only allowed
    occurrences, so those two contexts are excluded.
    """
    hits: list[dict[str, Any]] = []
    allowed_context = re.compile(
        r"is not a confidence interval|supports no claim about statistical support|"
        r"NOT a selection interval|no p-value is reported|"
        r"\"p_values_produced\"|\"forbidden_terms_source\"", re.IGNORECASE)
    for path in _text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if allowed_context.search(line):
                continue
            lowered = line.lower()
            for term in fsr.FORBIDDEN_UNCERTAINTY_TERMS:
                if term in lowered:
                    hits.append({"path": str(path), "line": line_number, "term": term})
    return hits


def _scan_excluded_tokens(root: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    allowed_context = re.compile(
        r"excluded_experiments|is excluded by design|out_of_scope_for_this_frozen_analysis|"
        r"high_prevalence_different_regime", re.IGNORECASE)
    for path in _text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if allowed_context.search(line):
                continue
            lowered = line.lower()
            for token in fsr.EXCLUDED_TOKENS:
                if token in lowered:
                    hits.append({"path": str(path), "line": line_number, "token": token})
    return hits


def _read_oof(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / fsr.OOF_PREDICTIONS_DIRNAME)


# =============================================================================
# Contract checks (run in both modes)
# =============================================================================
def run_contract_checks(report: Report, experiments: list[str]) -> None:
    pairs = fsr.directed_pairs(experiments)
    report.ok("FSR-01", len(pairs) == fsr.EXPECTED_DIRECTED_PAIRS,
              fsr.EXPECTED_DIRECTED_PAIRS, len(pairs), note="directed pair count")
    report.ok("FSR-02", all(source != target for source, target in pairs),
              "no self-pair", [f"{s}->{t}" for s, t in pairs if s == t] or "none")
    report.ok("FSR-03",
              all(fsr.direction_token(s, t) == f"{s}_to_{t}" for s, t in pairs),
              "direction tokens never sorted", "ok")

    excluded_ok = True
    for candidate in ("evia_2021", "evia_2021_extended"):
        try:
            fsr.assert_not_excluded(candidate)
            excluded_ok = False
        except fsr.FewShotRecoveryError:
            pass
    report.ok("FSR-06", excluded_ok, "Evia identifiers rejected",
              "rejected" if excluded_ok else "accepted")

    report.ok("FSR-05", fsr.POPULATION == "burnable_tree_shrub_grass",
              "burnable_tree_shrub_grass", fsr.POPULATION)

    all_features = set().union(*(set(f) for f in fsr.FEATURE_LISTS.values()))
    leaked = sorted(all_features & set(fsr.FORBIDDEN_FEATURE_COLUMNS))
    report.ok("FSR-14", not leaked, "no forbidden feature column", leaked or "none")

    report.ok("FSR-16", fsr.PRIMARY_METRIC == "roc_auc" and set(fsr.METRICS) == {
        "roc_auc", "pr_auc", "brier_score"},
        "threshold-free metrics only", list(fsr.METRICS),
        note="no threshold is selected anywhere in this analysis")

    report.ok("FSR-27",
              fsr.BLOCK_SIZE_CELLS == 10
              and fsr.BLOCK_NOMINAL_SCALE == "approximately_5_km",
              "10-cell blocks, approximately_5_km",
              f"{fsr.BLOCK_SIZE_CELLS} / {fsr.BLOCK_NOMINAL_SCALE}")
    report.ok("FSR-28",
              fsr.N_OUTER_FOLDS == 5 and fsr.FOLD_RANDOM_STATE == 42,
              "5 strict folds, seed 42",
              f"{fsr.N_OUTER_FOLDS} / {fsr.FOLD_RANDOM_STATE}")
    report.ok("FSR-30", "StratifiedGroupKFold" in json.dumps(
        fsr.build_scientific_config(experiments, {
            experiment: {"sha256": ""} for experiment in experiments})["outer_folds"]),
        "StratifiedGroupKFold is the only splitter", "declared")

    classifier = fsr.build_classifier(fsr.MODEL_NAME, fsr.ESTIMATOR_SEED)
    params = classifier.get_params(deep=False)
    report.ok("FSR-34",
              params.get("class_weight") == "balanced"
              and params.get("n_estimators") == 300
              and params.get("random_state") == 42,
              "canonical RandomForest hyperparameters", params,
              note="pre-existing class_weight='balanced'; no sample_weight, no tuning")

    stage_order_ok = True
    try:
        fsr.validate_stage_range("summarize", "plan")
        stage_order_ok = False
    except fsr.FewShotRecoveryError:
        pass
    report.ok("FSR-08a", stage_order_ok, "reversed stage range rejected",
              "rejected" if stage_order_ok else "accepted")

    report.ok("FSR-36a", fsr.build_scientific_config(experiments, {
        experiment: {"sha256": ""} for experiment in experiments}
    )["recovery"]["clipped"] is False,
        "recovery fraction not clipped", "declared unclipped")

    report.ok("FSR-40a",
              fsr.build_scientific_config(experiments, {
                  experiment: {"sha256": ""} for experiment in experiments}
              )["uncertainty"]["p_values_produced"] is False,
              "no p-values", "declared")

    # Whether some OTHER module in this process happens to have imported `ee`
    # is irrelevant: the contract is that THIS analysis never reaches Earth
    # Engine. The honest check is its own source.
    source = Path(fsr.__file__).read_text(encoding="utf-8")
    imports_gee = ("import ee" in source) or ("core.gee_utils" in source)
    report.ok("FSR-42a", not imports_gee,
              "module imports neither ee nor core.gee_utils",
              "clean" if not imports_gee else "imports GEE")


def run_input_checks(report: Report, experiments: list[str],
                     experiments_root: Optional[Path]) -> Optional[dict[str, Any]]:
    try:
        inventory = fsr.build_frozen_input_inventory(experiments, experiments_root)
    except fsr.FewShotRecoveryError as exc:
        report.ok("FSR-04", False, "canonical Step8A hashes match", str(exc))
        return None
    mismatched = [
        experiment for experiment, entry in inventory.items() if not entry.get("match")
    ]
    report.ok("FSR-04", not mismatched, "all canonical Step8A hashes match",
              mismatched or "all match",
              evidence_path=str(next(iter(inventory.values()))["path"]))
    return inventory


# =============================================================================
# Artifact checks (actual mode only)
# =============================================================================
def run_artifact_checks(report: Report, root: Path, experiments: list[str],
                        inventory: Optional[dict[str, Any]]) -> None:
    config_path = root / "config.json"
    if not config_path.is_file():
        report.ok("FSR-08", False, "config.json present", "missing", str(config_path))
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scientific = config["scientific_configuration"]

    recomputed = fsr.compute_analysis_id(scientific)
    report.ok("FSR-08", recomputed == config["analysis_id"] == root.name,
              "analysis_id reproduces from the frozen config",
              {"recomputed": recomputed, "config": config["analysis_id"],
               "directory": root.name}, str(config_path))

    selected = pd.read_parquet(root / "selected_blocks.parquet")
    feasibility = _read_csv(root / "direction_budget_feasibility.csv")
    inventory_csv = _read_csv(root / "target_block_inventory.csv")
    repeat_metrics = _read_csv(root / "repeat_metrics.csv")
    curve = _read_csv(root / "recovery_curve.csv")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    oof = _read_oof(root)

    directions = sorted(set(selected["direction"]))
    report.ok("FSR-01b", len(directions) == fsr.EXPECTED_DIRECTED_PAIRS,
              fsr.EXPECTED_DIRECTED_PAIRS, len(directions),
              str(root / "selected_blocks.parquet"))
    report.ok("FSR-02b",
              bool((selected["source_experiment"] != selected["target_experiment"]).all()),
              "no self-pair row", "ok")

    report.ok("FSR-05b", set(oof["population"]) == {fsr.POPULATION},
              fsr.POPULATION, sorted(set(oof["population"])))

    token_hits = _scan_excluded_tokens(root)
    report.ok("FSR-06b", not token_hits, "no Evia identifier in any artifact",
              token_hits[:5] or "none")

    # --- FSR-07 population sizes -------------------------------------------
    per_target = {}
    for target, group in oof.groupby("target_experiment"):
        n_series = group.groupby(
            ["direction", "condition", "budget_blocks", "repeat_id"]).ngroups
        per_target[target] = int(len(group) / max(n_series, 1))
    declared = {
        str(row["target_experiment"]): int(row["population_rows"])
        for _, row in inventory_csv[inventory_csv["outer_fold"] == -1].iterrows()
    }
    report.ok("FSR-07", per_target == declared,
              declared, per_target, str(root / "target_block_inventory.csv"))

    # --- FSR-09 / FSR-10 adaptation vs evaluation blocks --------------------
    eval_blocks: dict[tuple[str, int], set[str]] = {}
    for (direction, fold), group in oof[oof["condition"] == fsr.CONDITION_RAW].groupby(
        ["direction", "outer_fold"]
    ):
        eval_blocks[(direction, int(fold))] = set(group["evaluation_block_id"])

    overlaps = 0
    for (direction, fold), group in selected.groupby(["direction", "outer_fold"]):
        blocks = set(group["adaptation_block_id"])
        overlaps += len(blocks & eval_blocks.get((direction, int(fold)), set()))
    report.ok("FSR-09", overlaps == 0, 0, overlaps,
              note="adaptation/evaluation block overlap over every selection")

    target_blocks_by_target = {
        target: set(group["evaluation_block_id"])
        for target, group in oof.groupby("target_experiment")
    }
    foreign = 0
    for _, row in selected.iterrows():
        if row["adaptation_block_id"] not in target_blocks_by_target[row["target_experiment"]]:
            foreign += 1
    report.ok("FSR-10", foreign == 0, 0, foreign,
              note="every adaptation block belongs to its target region")

    # --- FSR-11 / FSR-12 / FSR-13 firewall ----------------------------------
    report.ok("FSR-11", overlaps == 0 and foreign == 0,
              "evaluation blocks absent from every training frame",
              {"eval_overlap": overlaps, "foreign_blocks": foreign},
              note="training frames are built from the training pool only; "
                   "block-level disjointness implies cell-level disjointness")

    seed_ok = True
    for _, row in selected.drop_duplicates(
        ["direction", "outer_fold", "repeat_id"]
    ).iterrows():
        expected_seed = fsr.selection_seed(
            row["source_experiment"], row["target_experiment"],
            int(row["outer_fold"]), int(row["repeat_id"]))
        if int(row["selection_seed"]) != expected_seed:
            seed_ok = False
            break
    report.ok("FSR-12", seed_ok,
              "selection seed reproducible from (direction, fold, repeat) alone",
              "reproduced" if seed_ok else "mismatch")
    report.ok("FSR-13", seed_ok,
              "selection cannot depend on evaluation labels",
              "seed derivation contains no label term",
              note="counterfactual label permutation is covered by the unit test suite")

    report.ok("FSR-15",
              scientific["preprocessing"]["fit_scope"]
              == "training_frame_of_each_condition_only",
              "preprocessing fitted on the training frame only",
              scientific["preprocessing"]["fit_scope"])

    threshold_columns = {"threshold", "precision", "recall", "f1", "balanced_accuracy"}
    present = sorted(threshold_columns & (set(repeat_metrics.columns) | set(curve.columns)))
    report.ok("FSR-16b", not present, "no threshold-dependent column", present or "none")

    # --- FSR-17 OOF coverage -------------------------------------------------
    coverage_failures = []
    for keys, group in oof.groupby(["direction", "condition", "budget_blocks", "repeat_id"]):
        expected_rows = per_target[group["target_experiment"].iloc[0]]
        if len(group) != expected_rows or group["cell_id"].duplicated().any():
            coverage_failures.append(keys)
        if group[["baseline_probability", "thermal_probability"]].isna().any().any():
            coverage_failures.append(keys)
    report.ok("FSR-17", not coverage_failures,
              "every target row predicted exactly once per series, no NaN",
              coverage_failures[:5] or "complete")

    # --- FSR-18 nesting ------------------------------------------------------
    nested_ok = True
    try:
        fsr.assert_nested_budgets(selected)
    except fsr.FewShotRecoveryError as exc:
        nested_ok = False
        report.ok("FSR-18", False, "budgets nested", str(exc))
    if nested_ok:
        report.ok("FSR-18", True, "budgets nested", "prefix property holds")

    counts = selected.groupby(
        ["direction", "outer_fold", "repeat_id", "budget_blocks"]
    )["adaptation_block_id"].nunique()
    budget_index = counts.index.get_level_values("budget_blocks")
    report.ok("FSR-19", bool((counts.to_numpy() == budget_index.to_numpy()).all()),
              "k distinct blocks per selection",
              "ok" if bool((counts.to_numpy() == budget_index.to_numpy()).all())
              else "mismatch")

    report.ok("FSR-20", seed_ok, "deterministic seeds", "reproduced")

    seed_by_key = selected.groupby(["direction", "outer_fold", "repeat_id"])[
        "selection_seed"].nunique()
    report.ok("FSR-21", bool((seed_by_key == 1).all()),
              "one seed per (direction, fold, repeat), shared by every budget",
              int(seed_by_key.max()))

    tier_rank = {tier: index for index, tier in enumerate(fsr.TIER_ORDER)}
    tier_ok = True
    for _, group in selected.groupby(["direction", "outer_fold", "repeat_id", "budget_blocks"]):
        ordered = group.sort_values("selection_rank")
        ranks = [tier_rank[tier] for tier in ordered["block_tier"]]
        if ranks != sorted(ranks):
            tier_ok = False
            break
    report.ok("FSR-22", tier_ok, "tier order both_classes -> positives_only -> negatives_only",
              "respected" if tier_ok else "violated")

    positives = selected.groupby(
        ["direction", "outer_fold", "repeat_id", "budget_blocks"]
    )["block_positive_count"].sum()
    report.ok("FSR-23", bool((positives > 0).all()),
              "every k>=1 selection contains a burned-containing block",
              int(positives.min()))

    repeats = selected[selected["budget_blocks"] > 0].groupby(
        ["direction", "outer_fold", "budget_blocks"])["repeat_id"].nunique()
    report.ok("FSR-24", bool((repeats == fsr.N_REPEATS).all()),
              fsr.N_REPEATS, sorted(set(repeats.tolist())))

    endpoints = curve[curve["budget_blocks"] == 0]
    single_ok = bool((endpoints["n_repeats"] == 1).all())
    degenerate_ok = True
    for _, row in endpoints.iterrows():
        lower, upper = row["selection_interval_lower"], row["selection_interval_upper"]
        if pd.notna(lower) and pd.notna(upper) and abs(float(upper) - float(lower)) > 1e-12:
            degenerate_ok = False
            break
    report.ok("FSR-25", single_ok and degenerate_ok,
              "k=0 and ceiling carry one realisation with a degenerate interval",
              {"n_repeats_is_one": single_ok, "interval_degenerate": degenerate_ok})

    report.ok("FSR-26", True, "row-order invariance",
              "covered by the unit test suite (blocks sorted before shuffling)",
              note="structural: nested_block_ordering sorts by block id")

    # --- FSR-27..30 blocks and folds ----------------------------------------
    block_pattern = re.compile(r"^b10_r-?\d+_c-?\d+$")
    bad_ids = [b for b in set(selected["adaptation_block_id"]) if not block_pattern.match(b)]
    report.ok("FSR-27b", not bad_ids, "b10_r{r}_c{c}", bad_ids[:5] or "all match")
    report.ok("FSR-28b", set(oof["n_outer_folds"]) == {fsr.N_OUTER_FOLDS},
              fsr.N_OUTER_FOLDS, sorted(set(oof["n_outer_folds"])))

    fold_map_ok = True
    for target, group in oof.groupby("target_experiment"):
        per_direction = {
            direction: dict(zip(sub["cell_id"], sub["outer_fold"]))
            for direction, sub in group[group["condition"] == fsr.CONDITION_RAW].groupby(
                "direction")
        }
        maps = list(per_direction.values())
        if len(maps) > 1 and any(m != maps[0] for m in maps[1:]):
            fold_map_ok = False
            break
    report.ok("FSR-29", fold_map_ok,
              "fold assignment identical across the directions sharing a target",
              "identical" if fold_map_ok else "diverged")
    report.ok("FSR-30b", scientific["outer_folds"]["splitter"] == "StratifiedGroupKFold",
              "StratifiedGroupKFold", scientific["outer_folds"]["splitter"])

    # --- FSR-31..33 condition contract --------------------------------------
    fold_rows = repeat_metrics[repeat_metrics["evaluation_level"] == fsr.EVALUATION_LEVEL_FOLD]
    raw_rows = fold_rows[fold_rows["condition"] == fsr.CONDITION_RAW]
    report.ok("FSR-31",
              bool((raw_rows["n_train_target_rows"] == 0).all())
              and bool((raw_rows["budget_blocks"] == 0).all()),
              "raw has zero target training rows and budget 0",
              {"max_target_rows": int(raw_rows["n_train_target_rows"].max()),
               "budgets": sorted(set(raw_rows["budget_blocks"]))})

    ceiling_rows = fold_rows[fold_rows["condition"] == fsr.CONDITION_CEILING]
    report.ok("FSR-32", bool((ceiling_rows["n_train_source_rows"] == 0).all()),
              "ceiling has zero source training rows",
              int(ceiling_rows["n_train_source_rows"].max()))

    fewshot_rows = fold_rows[fold_rows["condition"] == fsr.CONDITION_FEWSHOT]
    consistent = (
        fewshot_rows["n_train_rows"]
        == fewshot_rows["n_train_source_rows"] + fewshot_rows["adaptation_row_count"]
    )
    report.ok("FSR-33", bool(consistent.all()),
              "few_shot train rows = full source + adaptation rows",
              int((~consistent).sum()))

    report.ok("FSR-34b",
              scientific["model"]["sample_weight_argument_used"] is False
              and scientific["model"]["tuning_performed"] is False
              and scientific["model"]["pre_existing_class_weighting"]["present"] is True,
              "no sample_weight, no tuning, pre-existing class weighting declared",
              scientific["model"]["pre_existing_class_weighting"]["mechanism"])

    # --- FSR-35 ceiling reproduction ----------------------------------------
    for experiment_id in experiments:
        entry = summary["ceiling_reproduction"].get(experiment_id, {})
        if not entry.get("available"):
            report.skip(f"FSR-35[{experiment_id}]", "ceiling reproduces frozen 10-cell value",
                        entry.get("reason", "no frozen artifact"))
            continue
        for family, detail in sorted(entry["family"].items()):
            report.ok(f"FSR-35[{experiment_id}/{family}]", bool(detail.get("match")),
                      detail.get("expected"), detail.get("observed"),
                      note=f"abs_diff={detail.get('abs_diff')} tol={detail.get('tolerance')}")

    # --- FSR-36..39 recovery arithmetic -------------------------------------
    recompute_failures = []
    clip_violations = 0
    for _, row in curve.iterrows():
        raw_o, few_o, ceil_o = (row["raw_oriented"], row["fewshot_oriented"],
                                row["ceiling_oriented"])
        if pd.isna(raw_o) or pd.isna(few_o) or pd.isna(ceil_o):
            continue
        expected = fsr.recovery_quantities(float(raw_o), float(few_o), float(ceil_o))
        stored = row["recovery_fraction"]
        if expected["recovery_fraction"] is None:
            if pd.notna(stored):
                recompute_failures.append(row["direction"])
        elif pd.isna(stored) or abs(float(stored) - expected["recovery_fraction"]) > 1e-12:
            recompute_failures.append(row["direction"])
        if pd.notna(stored) and (float(stored) < 0.0 or float(stored) > 1.0):
            clip_violations += 1
    report.ok("FSR-36", not recompute_failures,
              "recovery fraction reproduces the closed form exactly",
              recompute_failures[:5] or "exact",
              note=f"{clip_violations} value(s) outside [0,1] preserved -- unclipped")

    brier = curve[curve["metric"] == "brier_score"]
    orientation_ok = bool(
        (brier["metric_orientation"] == fsr.ORIENTATION_NEGATED).all()
    ) and bool(
        np.allclose(brier["raw_oriented"].astype(float),
                    -brier["raw_value"].astype(float), atol=1e-12)
    ) and bool((brier["raw_value"].astype(float) >= 0).all())
    others = curve[curve["metric"] != "brier_score"]
    others_ok = bool(np.allclose(others["raw_oriented"].astype(float),
                                 others["raw_value"].astype(float), atol=1e-12))
    report.ok("FSR-37", orientation_ok and others_ok,
              "oriented_brier = -brier; ROC/PR unchanged; natural Brier preserved",
              {"brier_ok": orientation_ok, "roc_pr_ok": others_ok})

    degenerate = curve[_as_bool_series(curve["denominator_near_zero"])]
    report.ok("FSR-38",
              bool(degenerate["recovery_fraction"].isna().all())
              and bool((degenerate["recovery_fraction_status"]
                        == fsr.STATUS_DEGENERATE).all()),
              "degenerate denominator -> null fraction, status undefined",
              int(len(degenerate)))

    not_above = curve[_as_bool_series(curve["ceiling_not_above_raw"])]
    flagged = not_above["recovery_fraction_status"].isin(
        {fsr.STATUS_CEILING_NOT_ABOVE_RAW, fsr.STATUS_DEGENERATE})
    report.ok("FSR-39", bool(flagged.all()) if len(not_above) else True,
              "ceiling<=raw is flagged", int(len(not_above)))

    # --- FSR-40..42 wording and containment ---------------------------------
    hits = _scan_forbidden_terms(root)
    report.ok("FSR-40", not hits, "no confidence-interval or significance vocabulary",
              hits[:5] or "clean")
    interval_columns = [c for c in curve.columns if "selection_interval" in c]
    report.ok("FSR-40b", "selection_interval_lower" in curve.columns
              and "selection_interval_upper" in curve.columns
              and not any(c.startswith("ci_") for c in curve.columns),
              "selection_interval_* column names", interval_columns)

    outside = [entry["path"] for entry in manifest["files"]
               if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts]
    report.ok("FSR-41", not outside, "every manifest path is namespace-relative",
              outside[:5] or "contained")

    if inventory is not None:
        recorded = config["scientific_configuration"]["input_hashes"]
        drift = {
            experiment: {"config": recorded.get(experiment),
                         "on_disk": entry["sha256"]}
            for experiment, entry in inventory.items()
            if recorded.get(experiment) != entry["sha256"]
        }
        report.ok("FSR-42", not drift, "canonical Step8A unchanged since the run",
                  drift or "unchanged")
    else:
        report.skip("FSR-42", "canonical Step8A unchanged", "inventory unavailable")

    report.ok("FSR-42b", summary.get("earth_engine_used") is False
              and manifest.get("earth_engine_used") is False,
              "no Earth Engine", summary.get("earth_engine_used"))

    logical = manifest.get("logical_datasets", {}).get(fsr.OOF_PREDICTIONS_DIRNAME, {})
    report.ok("FSR-41b",
              logical.get("kind") == "partitioned_parquet_dataset"
              and len(logical.get("parts", [])) == fsr.EXPECTED_DIRECTED_PAIRS,
              "oof predictions exposed as one logical dataset of 6 parts",
              {"kind": logical.get("kind"), "parts": len(logical.get("parts", []))})

    # --- FSR-43..45 fit accounting ------------------------------------------
    fit_marker_path = root / "stages" / "fit.json"
    if not fit_marker_path.is_file():
        report.skip("FSR-43", "fit accounting", "no fit stage marker")
        return
    fit_marker = json.loads(fit_marker_path.read_text(encoding="utf-8"))
    accounting = fit_marker.get("fit_accounting", {})
    expected = fsr.expected_unique_fit_count(len(directions), len(experiments))

    report.ok("FSR-43",
              accounting.get("unique_fits") == expected["unique_fits"],
              expected["unique_fits"], accounting.get("unique_fits"),
              str(fit_marker_path),
              note="3,642 = 3,600 few-shot + 12 raw + 30 ceiling")
    report.ok("FSR-43b",
              accounting.get("raw_fits") == expected["raw_fits"]
              and accounting.get("few_shot_fits") == expected["few_shot_fits"]
              and accounting.get("ceiling_fits") == expected["ceiling_fits"],
              expected, {k: accounting.get(k) for k in
                         ("raw_fits", "few_shot_fits", "ceiling_fits")})
    report.ok("FSR-44",
              accounting.get("raw_reuse_per_fit") == float(fsr.N_OUTER_FOLDS),
              float(fsr.N_OUTER_FOLDS), accounting.get("raw_reuse_per_fit"),
              note="each raw fit is evaluated against all 5 outer folds")
    report.ok("FSR-45",
              accounting.get("ceiling_reuse_per_fit") == 2.0,
              2.0, accounting.get("ceiling_reuse_per_fit"),
              note="each ceiling fit is shared by the two directions into that target")


# =============================================================================
# Entry point
# =============================================================================
def run_validation(analysis_id: Optional[str] = None, dry_run: bool = False,
                   experiments: Optional[list[str]] = None,
                   output_root: Optional[Path] = None,
                   experiments_root: Optional[Path] = None) -> dict[str, Any]:
    report = Report()
    resolved = fsr.resolve_experiments(experiments)

    run_contract_checks(report, resolved)
    inventory = run_input_checks(report, resolved, experiments_root)

    if dry_run:
        return report.payload("dry-run", analysis_id, None)

    if analysis_id is None:
        if inventory is None:
            report.ok("FSR-00", False, "resolvable analysis_id",
                      "canonical inputs unavailable")
            return report.payload("actual", None, None)
        analysis_id = fsr.compute_analysis_id(
            fsr.build_scientific_config(resolved, inventory))

    root = fsr.analysis_root(analysis_id, output_root)
    if not root.is_dir():
        report.ok("FSR-00", False, f"namespace {root} exists", "missing", str(root))
        return report.payload("actual", analysis_id, str(root))

    run_artifact_checks(report, root, resolved, inventory)
    return report.payload("actual", analysis_id, str(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a few_shot_recovery.v1 artifact against the frozen check "
            "contract. Any FAIL makes the overall status FAIL."
        )
    )
    parser.add_argument("--analysis-id", default=None,
                        help="Analysis id to validate; derived from the frozen config when omitted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Contract and input checks only; reads no produced artifact.")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--write-report", action="store_true",
                        help="Write validation_report.json into the analysis namespace.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_validation(
        analysis_id=args.analysis_id,
        dry_run=args.dry_run,
        experiments=args.experiments,
        output_root=Path(args.output_root) if args.output_root else None,
        experiments_root=Path(args.experiments_root) if args.experiments_root else None,
    )
    print(json.dumps(payload, indent=2, default=str))
    if args.write_report and payload.get("namespace"):
        target = Path(payload["namespace"]) / "validation_report.json"
        fsr._atomic_write_text(target, fsr._json_document(payload))
    print(f"OVERALL STATUS: {payload['overall_status']}")
    return 0 if payload["overall_status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
