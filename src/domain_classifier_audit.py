"""Generic, multi-experiment PAIRWISE domain-classifier (covariate-
separability) diagnostic.

SCIENTIFIC PURPOSE
-------------------
For every unordered pair of resolved experiments, measure how well the two
regions can be distinguished using PREDICTOR FEATURES ALONE (never burned
labels, transfer predictions, coordinates, or experiment identity). The
classifier target is region/domain identity, not burned-area risk. This is
a covariate-separability diagnostic: it does NOT establish causality and
does NOT prove that covariate shift is the sole cause of any observed
cross-region transfer success or failure.

MANDATORY PRE-IMPLEMENTATION AUDIT (see deliverable summary for details)
-------------------------------------------------------------------------
An exhaustive repo-wide search (working tree, all git history, all
branches) found NO existing domain-classifier implementation or result
anywhere in this repository -- the previously reported "~1.000 AUC" for
Manavgat_2021<->Bejis_2022 could not be matched to any artifact. Per
explicit instruction, no fallback "legacy_comparable_domain_auc" is
invented: `legacy_method_available` is always False and the legacy fields
are always null in every output this module writes. The spatial-block
result (`spatial_block_domain_auc`) is the sole, primary result.

CODE-PATH DISCIPLINE
---------------------
This module does not rerun or modify Step8/Step9A-G/Step10. It only reads
the frozen Step8A 500 m modeling datasets (read-only) and reuses shared
infrastructure rather than reimplementing it:
    - canonical label-analysis eligibility + registry resolution:
      src.burned_pattern_audit (resolve_analysis_eligible_mask,
      resolve_experiments, canonical_step8a_path, ExperimentResolution)
    - canonical Step9 feature contract + forbidden-column list:
      src.step9a_audit_cross_region_inputs (SHARED_THERMAL_MODEL_FEATURES,
      FORBIDDEN_MODEL_COLUMNS, PRIMARY_POPULATIONS)
    - classifier/pipeline construction + spatial-block grouped CV:
      src.step8b_train_baseline_vs_thermal_model (build_pipeline,
      build_classifier, add_spatial_block_id, make_spatial_folds,
      check_no_forbidden_features)
    - block scale + bootstrap configuration: same values as Step9G
      (src.step9g_univariate_feature_auc_direction_reversal)
    - canonical JSON / SHA-256 / git commit: src.step8_large_block_robustness

No experiment ID is hard-coded anywhere in this module.
"""
from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.paths import PROJECT_ROOT
from src.burned_pattern_audit import (
    ExperimentResolution,
    canonical_step8a_path,
    dataset_schema_columns,
    resolve_analysis_eligible_mask,
    resolve_experiments as resolve_experiments_generic,
)
from src.step8_large_block_robustness import canonical_json, sha256_bytes, sha256_file, _git_commit
from src.step8b_train_baseline_vs_thermal_model import (
    CATEGORICAL_FEATURES,
    build_classifier,
    build_pipeline,
    check_no_forbidden_features,
    add_spatial_block_id,
    make_spatial_folds,
)
from src.step9a_audit_cross_region_inputs import (
    FORBIDDEN_MODEL_COLUMNS,
    PRIMARY_POPULATIONS,
    SHARED_THERMAL_MODEL_FEATURES,
)
import src.step9g_univariate_feature_auc_direction_reversal as step9g

# =============================================================================
# Frozen scientific/schema constants
# =============================================================================
ANALYSIS_SCHEMA_VERSION = "domain_classifier_audit.v1"

PRIMARY_POPULATION = PRIMARY_POPULATIONS[0]  # "burnable_tree_shrub_grass"
BURNABLE_MASK_COLUMN = PRIMARY_POPULATION

# No historical/legacy domain-classifier method exists anywhere in this
# repository (confirmed via exhaustive working-tree + full git-history +
# all-branches search). Never fabricated as a fallback -- always null/false.
LEGACY_METHOD_AVAILABLE = False

# Reused verbatim from the canonical Step9 feature contract
# (src/step9a_audit_cross_region_inputs.py) -- "expected candidates ...
# Step8 baseline and thermal predictors". No new/competing feature list is
# invented since no legacy contract exists to match.
DOMAIN_CLASSIFIER_FEATURES = tuple(SHARED_THERMAL_MODEL_FEATURES)
FEATURE_SET_ID = "step9_shared_baseline_thermal_v1"

MODEL_NAME = "random_forest"

# Reused verbatim from Step9G ("reuse the same block construction used by
# Step9G where possible").
BLOCK_SIZE_CELLS = step9g.BLOCK_SIZE_CELLS
NOMINAL_BLOCK_SCALE = step9g.NOMINAL_BLOCK_SCALE
N_SPLITS = 5
RANDOM_SEED = step9g.BOOTSTRAP_SEED  # 42
BOOTSTRAP_REPLICATES = step9g.BOOTSTRAP_REPLICATES  # 1000
BOOTSTRAP_SEED = step9g.BOOTSTRAP_SEED  # 42
CI_LOWER_PCT = step9g.CI_LOWER_PCT
CI_UPPER_PCT = step9g.CI_UPPER_PCT
MIN_VALID_REPLICATES = 900

REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        list(DOMAIN_CLASSIFIER_FEATURES) + ["row_500m", "col_500m", BURNABLE_MASK_COLUMN, "burned"]
    )
)

# Never permitted as predictors, by explicit semantic identity (union of the
# canonical Step9 forbidden-column list plus a few defensive extras named in
# the task -- geometry/model-prediction columns do not exist in Step8A
# today, but are listed so a future schema change still fails loudly).
NEVER_PREDICTOR_SEMANTIC_IDENTITIES = tuple(
    dict.fromkeys(
        list(FORBIDDEN_MODEL_COLUMNS) + [
            "geometry", "block_id", "large_block_id", "domain_block_id",
            "prediction", "predicted_probability", "transfer_prediction",
        ]
    )
)

SCIENTIFIC_LIMITATIONS = (
    "The domain classifier is a covariate-separability diagnostic, not a "
    "burned-area prediction model.",
    "It does not establish causality and does not prove that covariate "
    "shift is the sole cause of cross-region transfer success or failure.",
    "A high domain-classifier AUC does not necessarily cause transfer "
    "failure, and a low domain-classifier AUC does not guarantee "
    "successful transfer.",
    "This diagnostic does not claim statistical significance.",
    "No historical/legacy domain-classifier method exists in this "
    "repository; the spatial-block result is the sole primary result, not "
    "a comparison against a prior frozen value.",
)

PROHIBITED_ACTIONS = (
    "modify Step8A",
    "rerun Step8 modeling",
    "rerun transfer models",
    "modify Step9E or Step9G",
    "run or modify Step10",
    # Registry-driven, not AOI-specific: any experiment whose registry record
    # carries variant_status='legacy_superseded' (e.g. the superseded legacy
    # North Evia 2021 AOI variant) is barred from this canonical run; only its
    # `superseded_by` successor may enter. Enforced -- not merely declared --
    # by resolve_experiments(), which fails closed on superseded IDs instead
    # of silently dropping them. Successor experiments that are themselves
    # variant_status='canonical' (the extended North Evia AOI among them)
    # remain fully permitted.
    "use a legacy/superseded experiment_id (variant_status="
    "'legacy_superseded') in this canonical run instead of the canonical "
    "successor named by its registry 'superseded_by' pointer",
    "use burned labels for classifier fitting",
    "use row-random results as though they were spatially robust",
    "relax methodology after seeing results",
    "invent a fallback legacy method and call it comparable",
)

#: The ONE canonical output root of this analysis family. Every other path in
#: this module is derived from it through `resolve_layout()`; no output path
#: string is spelled out a second time here or in the runner.
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "domain_classifier_audit"

#: Legacy FLAT layout, kept as the default for direct calls to `analyze_pair()`
#: / `run_comparison()` so existing programmatic callers and their
#: monkeypatches keep working unchanged.
PAIRS_OUTPUT_ROOT = OUTPUT_ROOT / "pairs"
COMPARISON_OUTPUT_DIR = OUTPUT_ROOT / "comparison"


@dataclass(frozen=True)
class AuditLayout:
    """Resolved output paths for one run. The single path authority.

    Different analysis SCOPES get their own namespace UNDER `root` rather than
    a sibling root beside it. Sibling roots are what produced the hand-renamed
    `domain_classifier_audit_archive_*` directory: with no `--output-root` and
    a hardcoded default, preserving a previous result meant renaming the whole
    root by hand.
    """

    root: Path
    pairs: Path
    comparison: Path
    scope: Optional[str] = None


def default_layout() -> AuditLayout:
    """The legacy flat layout, read from the module constants at call time."""
    return AuditLayout(
        root=OUTPUT_ROOT,
        pairs=PAIRS_OUTPUT_ROOT,
        comparison=COMPARISON_OUTPUT_DIR,
        scope=None,
    )


def scope_key(resolution: ExperimentResolution) -> str:
    """Deterministic namespace name for one analysis scope.

    Delegates to `burned_pattern_audit.scope_key`, which both families share so
    the canonical-cohort rule cannot drift between them: a selection resolving
    to exactly the current `--all-enabled` cohort is named `all_enabled`
    however it was requested; any other selection keeps its own sorted name.
    """
    from src.burned_pattern_audit import scope_key as generic_scope_key

    return generic_scope_key(resolution)


def resolve_layout(
    output_root: Optional[Path] = None, scope: Optional[str] = None,
) -> AuditLayout:
    """Resolve the output layout for a run. The ONLY place paths are built."""
    if output_root is None and scope is None:
        return default_layout()
    # Implicit base is the PARENT of the legacy pairs directory, not
    # `OUTPUT_ROOT`: callers that redirect this analysis override the leaf
    # constants, and deriving from `OUTPUT_ROOT` would escape their sandbox.
    base = Path(output_root) if output_root is not None else PAIRS_OUTPUT_ROOT.parent
    namespace = base / scope if scope else base
    return AuditLayout(
        root=base,
        pairs=namespace / "pairs",
        comparison=namespace / "comparison",
        scope=scope,
    )


class DomainClassifierAuditError(SystemExit):
    """Fail-fast error for the domain-classifier audit."""


# =============================================================================
# Experiment resolution + pair generation (reuses burned_pattern_audit's
# generic registry-driven resolver; no experiment ID hard-coded).
# =============================================================================
def resolve_experiments(
    experiments: Optional[list[str]] = None, all_enabled: bool = False,
) -> ExperimentResolution:
    return resolve_experiments_generic(experiments=experiments, all_enabled=all_enabled)


def generate_pairs(resolved_ids: tuple[str, ...]) -> list[tuple[str, str]]:
    """Every unordered pair among resolved_ids, canonically sorted, each
    appearing exactly once."""
    return list(itertools.combinations(sorted(resolved_ids), 2))


def pair_output_dir(
    experiment_a: str, experiment_b: str, layout: Optional[AuditLayout] = None,
) -> Path:
    a, b = sorted((experiment_a, experiment_b))
    return (layout or default_layout()).pairs / f"{a}__{b}"


# =============================================================================
# Leakage / feature-contract audit (static -- does not require data)
# =============================================================================
def leakage_audit() -> dict[str, Any]:
    """Every predictor column vs the fixed exclusion list, plus an explicit
    'never predictor' semantic-identity check. Fails loudly if the fixed
    feature contract itself contains a forbidden column (defensive; should
    never trigger given the frozen SHARED_THERMAL_MODEL_FEATURES list)."""
    included = list(DOMAIN_CLASSIFIER_FEATURES)
    leaked = set(included) & set(NEVER_PREDICTOR_SEMANTIC_IDENTITIES)
    if leaked:
        raise DomainClassifierAuditError(
            f"Forbidden/leaked column(s) present in the domain-classifier feature contract: {sorted(leaked)}."
        )
    return {
        "included_predictors": included,
        "categorical_predictors": [f for f in included if f in CATEGORICAL_FEATURES],
        "numeric_predictors": [f for f in included if f not in CATEGORICAL_FEATURES],
        "never_predictor_semantic_identities": list(NEVER_PREDICTOR_SEMANTIC_IDENTITIES),
        "feature_set_id": FEATURE_SET_ID,
        "feature_set_source": "src.step9a_audit_cross_region_inputs.SHARED_THERMAL_MODEL_FEATURES",
    }


# =============================================================================
# Population resolution (canonical eligibility + Step9 primary population)
# =============================================================================
def resolve_population(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Canonical analysis-eligible rows (see src.burned_pattern_audit;
    excludes pre-label-burned rows for experiments configured with
    exclude_pre_label_burns, e.g. Mugla) restricted to the frozen Step9
    primary population burnable_tree_shrub_grass. Includes BOTH burned and
    unburned eligible cells -- burned status is never used to filter here."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DomainClassifierAuditError(
            f"'{experiment_id}': canonical Step8A dataset is missing required column(s): {missing}."
        )
    eligible_mask = resolve_analysis_eligible_mask(df)
    valid_grid = df["row_500m"].notna() & df["col_500m"].notna()
    population_mask = eligible_mask & valid_grid & (df[BURNABLE_MASK_COLUMN] == True)  # noqa: E712
    population = df.loc[population_mask].reset_index(drop=True)
    if population[["row_500m", "col_500m"]].duplicated().any():
        raise DomainClassifierAuditError(f"'{experiment_id}': duplicate (row_500m, col_500m) in analysis population.")
    return population


def assign_domain_blocks(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Namespaces spatial-block IDs by experiment (id_prefix=experiment_id)
    so blocks from different AOIs can never collide when concatenated."""
    return add_spatial_block_id(
        df, BLOCK_SIZE_CELLS, column_name="domain_block_id", id_prefix=experiment_id,
    )


def build_combined_frame(pop_a: pd.DataFrame, pop_b: pd.DataFrame, experiment_a: str, experiment_b: str) -> pd.DataFrame:
    """domain 0 = experiment_a (canonical-sorted first), domain 1 =
    experiment_b. The `domain` column is the ONLY place experiment
    identity enters the modeling frame; it becomes the classifier TARGET,
    never a predictor."""
    a = assign_domain_blocks(pop_a, experiment_a).assign(domain=0, _source_experiment_id=experiment_a)
    b = assign_domain_blocks(pop_b, experiment_b).assign(domain=1, _source_experiment_id=experiment_b)
    combined = pd.concat([a, b], ignore_index=True)
    return combined


# =============================================================================
# OOF fitting (spatial-block grouped CV; preprocessing fit inside each fold)
# =============================================================================
def fit_oof_predictions(combined: pd.DataFrame) -> dict[str, Any]:
    y = combined["domain"].to_numpy()
    groups = combined["domain_block_id"].to_numpy()
    features = list(DOMAIN_CLASSIFIER_FEATURES)
    check_no_forbidden_features(features)

    folds, n_splits_used = make_spatial_folds(
        y=y, groups=groups, n_splits_requested=N_SPLITS, random_state=RANDOM_SEED,
        min_positive_folds=2, strict=True,
    )

    oof_probs = np.full(len(combined), np.nan)
    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_blocks, test_blocks = set(groups[train_idx]), set(groups[test_idx])
        overlap = train_blocks & test_blocks
        if overlap:
            raise DomainClassifierAuditError(f"Fold {fold_id}: spatial block(s) {sorted(overlap)} appear in both train and test.")

        pipeline = build_pipeline(features, MODEL_NAME, RANDOM_SEED)
        X_train = combined.iloc[train_idx][features]
        y_train = y[train_idx]
        pipeline.fit(X_train, y_train)

        X_test = combined.iloc[test_idx][features]
        proba = pipeline.predict_proba(X_test)
        class1_idx = list(pipeline.classes_).index(1)
        fold_probs = proba[:, class1_idx]
        oof_probs[test_idx] = fold_probs

        y_test = y[test_idx]
        fold_auc = float(roc_auc_score(y_test, fold_probs)) if len(set(y_test)) == 2 else None
        fold_rows.append({
            "fold_id": fold_id,
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "n_train_domain_0": int((y_train == 0).sum()), "n_train_domain_1": int((y_train == 1).sum()),
            "n_test_domain_0": int((y_test == 0).sum()), "n_test_domain_1": int((y_test == 1).sum()),
            "train_blocks": len(train_blocks), "test_blocks": len(test_blocks),
            "block_overlap": len(overlap),
            "fold_auc": fold_auc,
        })

    if np.isnan(oof_probs).any():
        raise DomainClassifierAuditError("Not every row received an out-of-fold prediction (OOF coverage violation).")
    if not np.all((oof_probs >= 0.0) & (oof_probs <= 1.0)) or not np.all(np.isfinite(oof_probs)):
        raise DomainClassifierAuditError("Domain-classifier OOF probabilities are not all finite and within [0, 1].")

    overall_auc = float(roc_auc_score(y, oof_probs))
    fold_aucs = [r["fold_auc"] for r in fold_rows if r["fold_auc"] is not None]

    return {
        "oof_probs": oof_probs,
        "fold_rows": fold_rows,
        "n_splits_used": n_splits_used,
        "overall_oof_auc": overall_auc,
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "estimator_name": MODEL_NAME,
        "estimator_params": build_classifier(MODEL_NAME, RANDOM_SEED).get_params(deep=False),
    }


# =============================================================================
# Paired two-domain spatial-block bootstrap (blocks, never rows)
# =============================================================================
def block_bootstrap_domain_auc(
    combined: pd.DataFrame, oof_probs: np.ndarray,
    n_replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y = combined["domain"].to_numpy()
    blocks = combined["domain_block_id"].to_numpy()
    block_to_indices: dict[str, np.ndarray] = {
        block: np.where(blocks == block)[0] for block in np.unique(blocks)
    }
    domain0_blocks = sorted(np.unique(blocks[y == 0]))
    domain1_blocks = sorted(np.unique(blocks[y == 1]))

    rng = np.random.default_rng(seed)
    replicate_aucs: list[float] = []
    valid, invalid = 0, 0
    for _ in range(n_replicates):
        sampled0 = rng.choice(domain0_blocks, size=len(domain0_blocks), replace=True)
        sampled1 = rng.choice(domain1_blocks, size=len(domain1_blocks), replace=True)
        idx = np.concatenate(
            [block_to_indices[b] for b in sampled0] + [block_to_indices[b] for b in sampled1]
        )
        y_rep = y[idx]
        p_rep = oof_probs[idx]
        if len(set(y_rep)) < 2:
            invalid += 1
            continue
        replicate_aucs.append(float(roc_auc_score(y_rep, p_rep)))
        valid += 1

    if replicate_aucs:
        ci_low = float(np.percentile(replicate_aucs, CI_LOWER_PCT))
        ci_high = float(np.percentile(replicate_aucs, CI_UPPER_PCT))
    else:
        ci_low = ci_high = None

    return {
        "replicates_requested": n_replicates,
        "valid_replicates": valid,
        "invalid_replicates": invalid,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seed": seed,
        "method": "paired two-domain spatial-block bootstrap; blocks sampled with replacement independently within each domain; all rows per sampled block retained",
    }


# =============================================================================
# Manifest / analysis ID
# =============================================================================
def _package_versions() -> dict[str, str]:
    names = {"numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn"}
    return {key: importlib.metadata.version(pkg) for key, pkg in names.items()}


def scientific_configuration(resolved_experiment_ids: tuple[str, ...], input_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "resolved_experiment_ids": sorted(resolved_experiment_ids),
        "primary_population": PRIMARY_POPULATION,
        "eligibility_definition": (
            "src.burned_pattern_audit.resolve_analysis_eligible_mask: uses Step8A's "
            "own 'analysis_eligible' column (= NOT pre_label_burn_excluded) when "
            "present -- excluding pre-label-burned rows for any experiment configured "
            "with exclude_pre_label_burns -- and "
            "falls back to 'every row eligible' when the column is absent (no "
            "exclude_pre_label_burns configured for that experiment)."
        ),
        "feature_contract": leakage_audit(),
        "preprocessing": {
            "numeric": "SimpleImputer(strategy='median'), fit on the training fold only; no scaling; no per-region z-score normalization",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore'), fit on the training fold only",
            "source": "src.step8b_train_baseline_vs_thermal_model.build_pipeline (reused unmodified)",
        },
        "evaluation_method": {
            "description": (
                "Spatial-block grouped out-of-fold evaluation: StratifiedGroupKFold "
                "over deterministic 10-cell blocks, namespaced per experiment so "
                "blocks from different AOIs cannot collide; strict mode enforces "
                "zero train/test block overlap, both domain classes present on both "
                "sides of every fold, and full one-time OOF coverage."
            ),
            "source": "src.step8b_train_baseline_vs_thermal_model.make_spatial_folds(strict=True)",
        },
        "block_size_cells": BLOCK_SIZE_CELLS,
        "nominal_block_scale": NOMINAL_BLOCK_SCALE,
        "block_namespacing": "add_spatial_block_id(..., id_prefix=experiment_id) -- blocks from different AOIs never collide",
        "n_splits": N_SPLITS,
        "random_seed": RANDOM_SEED,
        "estimator_name": MODEL_NAME,
        "estimator_params": build_classifier(MODEL_NAME, RANDOM_SEED).get_params(deep=False),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "ci_lower_pct": CI_LOWER_PCT, "ci_upper_pct": CI_UPPER_PCT,
            "method": "paired two-domain spatial-block bootstrap; blocks (not rows) sampled with replacement independently within each domain",
        },
        "legacy_method_available": LEGACY_METHOD_AVAILABLE,
        "legacy_method_provenance": (
            "No domain-classifier implementation, artifact, manifest, test, or "
            "result was found anywhere in this repository after an exhaustive "
            "search of the working tree, full git commit history (git log --all "
            "-S), and all branches. A previously reported near-1.0 AUC for one "
            "specific pair could not be matched to any artifact. No fallback "
            "method is invented or labeled comparable; "
            "legacy_comparable_domain_auc is always null."
        ),
        "input_step8a_sha256": dict(sorted(input_hashes.items())),
        "package_versions": _package_versions(),
        "prohibited_actions": list(PROHIBITED_ACTIONS),
        "limitations": list(SCIENTIFIC_LIMITATIONS),
    }


def build_analysis_id(resolved_experiment_ids: tuple[str, ...], input_hashes: dict[str, str]) -> str:
    content = scientific_configuration(resolved_experiment_ids, input_hashes)
    return sha256_bytes(canonical_json(content).encode("utf-8"))


def _existing_manifest_analysis_id(output_dir: Path) -> Optional[str]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text()).get("analysis_id")
    except (json.JSONDecodeError, OSError):
        return None


def _guard_force(output_dir: Path, analysis_id: str, force: bool, label: str) -> None:
    existing = _existing_manifest_analysis_id(output_dir)
    if existing is not None and existing != analysis_id and not force:
        raise DomainClassifierAuditError(
            f"{label}: existing output at {output_dir} was produced by a different "
            f"analysis_id ({existing} != {analysis_id}). Use --force to overwrite."
        )


# =============================================================================
# Per-pair orchestration
# =============================================================================
def analyze_pair(
    experiment_a: str, experiment_b: str, dry_run: bool = False, force: bool = False,
    layout: Optional[AuditLayout] = None,
) -> dict[str, Any]:
    """`layout=None` keeps the legacy flat default, so existing callers are
    unaffected."""
    layout = layout or default_layout()
    a, b = sorted((experiment_a, experiment_b))
    pair_id = f"{a}__{b}"
    output_dir = pair_output_dir(a, b, layout)
    input_path_a = canonical_step8a_path(a)
    input_path_b = canonical_step8a_path(b)
    planned_paths = {
        "domain_classifier_metrics_json": str(output_dir / "domain_classifier_metrics.json"),
        "domain_classifier_fold_metrics_csv": str(output_dir / "domain_classifier_fold_metrics.csv"),
        "domain_classifier_oof_predictions_parquet": str(output_dir / "domain_classifier_oof_predictions.parquet"),
        "domain_classifier_bootstrap_json": str(output_dir / "domain_classifier_bootstrap.json"),
        "domain_classifier_report_md": str(output_dir / "domain_classifier_report.md"),
        "manifest_json": str(output_dir / "manifest.json"),
    }

    for exp_id, path in ((a, input_path_a), (b, input_path_b)):
        if not path.is_file():
            raise DomainClassifierAuditError(f"Missing canonical Step8A dataset for '{exp_id}': {path}.")

    if dry_run:
        columns_a = dataset_schema_columns(input_path_a)
        columns_b = dataset_schema_columns(input_path_b)
        missing_a = [c for c in REQUIRED_COLUMNS if c not in columns_a]
        missing_b = [c for c in REQUIRED_COLUMNS if c not in columns_b]
        if missing_a:
            raise DomainClassifierAuditError(f"'{a}': canonical Step8A dataset is missing required column(s): {missing_a}.")
        if missing_b:
            raise DomainClassifierAuditError(f"'{b}': canonical Step8A dataset is missing required column(s): {missing_b}.")
        hash_a = sha256_file(input_path_a)
        hash_b = sha256_file(input_path_b)
        return {
            "ran": False, "dry_run": True,
            "pair_id": pair_id, "experiment_a": a, "experiment_b": b,
            "domain_0": a, "domain_1": b,
            "input_paths": {a: str(input_path_a), b: str(input_path_b)},
            "input_sha256": {a: hash_a, b: hash_b},
            "scientific_configuration": scientific_configuration((a, b), {a: hash_a, b: hash_b}),
            "planned_output_paths": planned_paths,
        }

    hash_a_before = sha256_file(input_path_a)
    hash_b_before = sha256_file(input_path_b)
    df_a = pd.read_parquet(input_path_a)
    df_b = pd.read_parquet(input_path_b)

    pop_a = resolve_population(df_a, a)
    pop_b = resolve_population(df_b, b)
    if len(pop_a) == 0 or len(pop_b) == 0:
        raise DomainClassifierAuditError(f"'{pair_id}': empty analysis population for one or both experiments.")

    combined = build_combined_frame(pop_a, pop_b, a, b)
    fit_result = fit_oof_predictions(combined)
    oof_probs = fit_result["oof_probs"]
    bootstrap_result = block_bootstrap_domain_auc(combined, oof_probs)

    input_hashes = {a: hash_a_before, b: hash_b_before}
    analysis_id = build_analysis_id((a, b), input_hashes)
    _guard_force(output_dir, analysis_id, force, label=f"domain-classifier-audit[{pair_id}]")

    n_domain_0 = int((combined["domain"] == 0).sum())
    n_domain_1 = int((combined["domain"] == 1).sum())
    n_blocks_domain_0 = int(combined.loc[combined["domain"] == 0, "domain_block_id"].nunique())
    n_blocks_domain_1 = int(combined.loc[combined["domain"] == 1, "domain_block_id"].nunique())

    missing_counts = {feature: int(combined[feature].isna().sum()) for feature in DOMAIN_CLASSIFIER_FEATURES}
    burned_counts = {
        a: {"n_burned": int((pop_a["burned"] == 1).sum()), "n_unburned": int((pop_a["burned"] == 0).sum())},
        b: {"n_burned": int((pop_b["burned"] == 1).sum()), "n_unburned": int((pop_b["burned"] == 0).sum())},
    }

    manifest = {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "experiment_a": a, "experiment_b": b,
        "domain_0": a, "domain_1": b,
        "input_paths": {a: str(input_path_a), b: str(input_path_b)},
        "input_sha256": input_hashes,
        "output_paths": planned_paths,
        "scientific_configuration": scientific_configuration((a, b), input_hashes),
    }

    metrics = {
        "analysis_id": analysis_id,
        "pair_id": pair_id,
        "experiment_a": a, "experiment_b": b,
        "domain_0": a, "domain_1": b,
        "n_domain_0": n_domain_0, "n_domain_1": n_domain_1,
        "n_blocks_domain_0": n_blocks_domain_0, "n_blocks_domain_1": n_blocks_domain_1,
        "n_splits": fit_result["n_splits_used"],
        "zero_block_overlap": all(r["block_overlap"] == 0 for r in fit_result["fold_rows"]),
        "estimator_name": fit_result["estimator_name"],
        "estimator_params": fit_result["estimator_params"],
        "feature_count": len(DOMAIN_CLASSIFIER_FEATURES),
        "feature_set_id": FEATURE_SET_ID,
        "missing_value_counts": missing_counts,
        "burned_accounting": burned_counts,
        "fold_auc_mean": fit_result["fold_auc_mean"],
        "fold_auc_std": fit_result["fold_auc_std"],
        "spatial_block_domain_auc": fit_result["overall_oof_auc"],
        "spatial_block_ci_low": bootstrap_result["ci_low"],
        "spatial_block_ci_high": bootstrap_result["ci_high"],
        "valid_bootstrap_replicates": bootstrap_result["valid_replicates"],
        "total_bootstrap_replicates": bootstrap_result["replicates_requested"],
        "legacy_method_available": LEGACY_METHOD_AVAILABLE,
        "legacy_comparable_domain_auc": None,
        "legacy_comparable_ci_low": None,
        "legacy_comparable_ci_high": None,
        "legacy_evaluation_type": None,
        "result_status": "spatial_block_computed_no_legacy_precedent",
        "limitations": list(SCIENTIFIC_LIMITATIONS),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "domain_classifier_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    pd.DataFrame(fit_result["fold_rows"]).to_csv(output_dir / "domain_classifier_fold_metrics.csv", index=False)

    oof_df = pd.DataFrame({
        "row_500m": combined["row_500m"].astype(int),
        "col_500m": combined["col_500m"].astype(int),
        "domain": combined["domain"].astype(int),
        "domain_block_id": combined["domain_block_id"],
        "oof_probability_domain_1": oof_probs,
    })
    oof_df.to_parquet(output_dir / "domain_classifier_oof_predictions.parquet", index=False)
    (output_dir / "domain_classifier_bootstrap.json").write_text(json.dumps(bootstrap_result, indent=2, default=str))
    (output_dir / "domain_classifier_report.md").write_text(render_pair_markdown(metrics, manifest))

    hash_a_after = sha256_file(input_path_a)
    hash_b_after = sha256_file(input_path_b)
    if hash_a_after != hash_a_before or hash_b_after != hash_b_before:
        raise DomainClassifierAuditError(f"'{pair_id}': Step8A input hash changed during the audit run (expected read-only).")

    return {
        "ran": True, "dry_run": False,
        "pair_id": pair_id, "experiment_a": a, "experiment_b": b,
        "analysis_id": analysis_id, "output_dir": str(output_dir),
        "metrics": metrics, "manifest": manifest,
    }


# =============================================================================
# Markdown rendering
# =============================================================================
def render_pair_markdown(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        f"# Domain-classifier audit -- {metrics['experiment_a']} vs {metrics['experiment_b']}",
        "",
        f"analysis_id: `{metrics['analysis_id']}`  ",
        f"created_at: {manifest['created_at']}  ",
        f"domain_0 = {metrics['domain_0']}, domain_1 = {metrics['domain_1']}",
        "",
        "## Spatial-block result (primary)",
        "",
        f"- spatial_block_domain_auc: **{metrics['spatial_block_domain_auc']}**",
        f"- 95% spatial-block bootstrap CI: [{metrics['spatial_block_ci_low']}, {metrics['spatial_block_ci_high']}]",
        f"- fold AUC mean/std: {metrics['fold_auc_mean']} / {metrics['fold_auc_std']}",
        f"- n_splits: {metrics['n_splits']}, zero_block_overlap: {metrics['zero_block_overlap']}",
        f"- n_domain_0={metrics['n_domain_0']} (blocks={metrics['n_blocks_domain_0']}), "
        f"n_domain_1={metrics['n_domain_1']} (blocks={metrics['n_blocks_domain_1']})",
        f"- estimator: {metrics['estimator_name']} {metrics['estimator_params']}",
        f"- feature_set_id: {metrics['feature_set_id']} ({metrics['feature_count']} features)",
        "",
        "## Legacy-comparable result",
        "",
        f"- legacy_method_available: {metrics['legacy_method_available']}",
        "- No historical/legacy domain-classifier method exists anywhere in this "
        "repository (confirmed by exhaustive working-tree and git-history audit); "
        "legacy_comparable_domain_auc is null, not fabricated.",
        "",
        "## Burned-label accounting (input-accounting only; never used to fit the classifier)",
        "",
        f"- {metrics['experiment_a']}: {metrics['burned_accounting'][metrics['experiment_a']]}",
        f"- {metrics['experiment_b']}: {metrics['burned_accounting'][metrics['experiment_b']]}",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {line}" for line in metrics["limitations"]]
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Multi-experiment comparison
# =============================================================================
def comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_a": metrics["experiment_a"], "experiment_b": metrics["experiment_b"],
        "domain_0": metrics["domain_0"], "domain_1": metrics["domain_1"],
        "n_domain_0": metrics["n_domain_0"], "n_domain_1": metrics["n_domain_1"],
        "n_blocks_domain_0": metrics["n_blocks_domain_0"], "n_blocks_domain_1": metrics["n_blocks_domain_1"],
        "legacy_method_available": metrics["legacy_method_available"],
        "legacy_comparable_domain_auc": metrics["legacy_comparable_domain_auc"],
        "legacy_comparable_ci_low": metrics["legacy_comparable_ci_low"],
        "legacy_comparable_ci_high": metrics["legacy_comparable_ci_high"],
        "legacy_evaluation_type": metrics["legacy_evaluation_type"],
        "spatial_block_domain_auc": metrics["spatial_block_domain_auc"],
        "spatial_block_ci_low": metrics["spatial_block_ci_low"],
        "spatial_block_ci_high": metrics["spatial_block_ci_high"],
        "valid_bootstrap_replicates": metrics.get("valid_bootstrap_replicates"),
        "total_bootstrap_replicates": metrics.get("total_bootstrap_replicates"),
        "estimator_name": metrics["estimator_name"],
        "feature_set_id": metrics["feature_set_id"],
        "feature_count": metrics["feature_count"],
        "zero_block_overlap": metrics["zero_block_overlap"],
        "result_status": metrics["result_status"],
    }


def render_comparison_markdown(resolution: ExperimentResolution, results: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> str:
    rows_by_pair = {(r["metrics"]["experiment_a"], r["metrics"]["experiment_b"]): r["metrics"] for r in results.values()}
    lines = [
        "# Multi-AOI domain-classifier comparison",
        "",
        f"analysis_id: `{manifest['analysis_id']}` (order-invariant)  ",
        f"created_at: {manifest['created_at']}  ",
        f"resolved_experiment_ids: {sorted(resolution.resolved_ids)}",
        "",
        "## 1. Domain-classifier AUC per pair",
        "",
    ]
    for (a, b), m in sorted(rows_by_pair.items()):
        lines.append(
            f"- {a}–{b}: spatial_block_domain_auc = **{m['spatial_block_domain_auc']}** "
            f"[{m['spatial_block_ci_low']}, {m['spatial_block_ci_high']}] "
            f"(legacy_method_available={m['legacy_method_available']})"
        )
    lines.append("")

    lines.append("## 2. Is each pair also nearly perfectly domain-separable?")
    lines.append("")
    lines.append(
        "Generic per-pair separability check (no pair is singled out by name in "
        "this implementation; every resolved pair is evaluated identically):"
    )
    lines.append("")
    for (a, b), m in sorted(rows_by_pair.items()):
        auc = m["spatial_block_domain_auc"]
        lines.append(
            f"- {a}–{b}: spatial_block_domain_auc = {auc} [{m['spatial_block_ci_low']}, {m['spatial_block_ci_high']}]. "
            + (
                f"{a} and {b} remain highly separable in feature space; if this pair also "
                "transfers comparatively well elsewhere in this comparison, that would mean "
                "domain separability alone is insufficient to explain the observed transfer "
                "pattern for this pair."
                if auc is not None and auc >= 0.9
                else f"{a} and {b} are not highly separable under this method."
            )
        )
    lines.append("")

    lines.append("## 3. Does pair ordering support covariate distance alone explaining transfer success?")
    lines.append("")
    lines.append(
        "This diagnostic reports domain-classifier separability only. It does not "
        "claim causality, does not claim domain AUC is a formal geographic or "
        "climatic distance, does not claim a high domain AUC necessarily causes "
        "transfer failure, does not claim a low domain AUC guarantees successful "
        "transfer, and does not claim statistical significance."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines += [f"- {line}" for line in SCIENTIFIC_LIMITATIONS]
    lines.append("")
    return "\n".join(lines)


def run_comparison(
    resolution: ExperimentResolution, dry_run: bool, force: bool,
    layout: Optional[AuditLayout] = None,
) -> dict[str, Any]:
    layout = layout or default_layout()
    comparison_dir = layout.comparison
    pairs = generate_pairs(resolution.resolved_ids)
    planned_paths = {
        "multi_aoi_domain_classifier_comparison_json": str(comparison_dir / "multi_aoi_domain_classifier_comparison.json"),
        "multi_aoi_domain_classifier_comparison_csv": str(comparison_dir / "multi_aoi_domain_classifier_comparison.csv"),
        "multi_aoi_domain_classifier_comparison_md": str(comparison_dir / "multi_aoi_domain_classifier_comparison.md"),
        "manifest_json": str(comparison_dir / "manifest.json"),
    }

    if dry_run:
        pair_plans = {f"{a}__{b}": analyze_pair(a, b, dry_run=True, force=force, layout=layout) for a, b in pairs}
        return {
            "ran": False, "dry_run": True,
            "resolved_experiment_ids": list(resolution.resolved_ids),
            "generated_pairs": [f"{a}__{b}" for a, b in pairs],
            "pair_plans": pair_plans,
            "output_root": str(comparison_dir),
            "planned_output_paths": planned_paths,
        }

    pair_results: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for a, b in pairs:
        result = analyze_pair(a, b, dry_run=False, force=force, layout=layout)
        pair_results[result["pair_id"]] = result
        input_hashes.update(result["manifest"]["input_sha256"])

    analysis_id = build_analysis_id(resolution.resolved_ids, input_hashes)
    _guard_force(comparison_dir, analysis_id, force, label="domain-classifier-audit[comparison]")

    manifest = {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "requested_experiment_ids": list(resolution.requested_ids),
        "resolved_experiment_ids": sorted(resolution.resolved_ids),
        "selection_mode": resolution.selection_mode,
        "generated_pairs": [f"{a}__{b}" for a, b in pairs],
        "input_sha256": input_hashes,
        "output_paths": planned_paths,
        "pair_output_paths": {pid: r["manifest"]["output_paths"] for pid, r in pair_results.items()},
        "scientific_configuration": scientific_configuration(resolution.resolved_ids, input_hashes),
        "pair_analysis_ids": {pid: r["analysis_id"] for pid, r in pair_results.items()},
    }

    comparison_rows = [comparison_row(r["metrics"]) for r in pair_results.values()]
    comparison_df = pd.DataFrame(comparison_rows)

    comparison_json = {
        "analysis_id": analysis_id,
        "created_at": manifest["created_at"],
        "resolved_experiment_ids": sorted(resolution.resolved_ids),
        "rows": comparison_rows,
        "limitations": list(SCIENTIFIC_LIMITATIONS),
    }

    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "multi_aoi_domain_classifier_comparison.json").write_text(json.dumps(comparison_json, indent=2, default=str))
    comparison_df.to_csv(comparison_dir / "multi_aoi_domain_classifier_comparison.csv", index=False)
    (comparison_dir / "multi_aoi_domain_classifier_comparison.md").write_text(
        render_comparison_markdown(resolution, pair_results, manifest)
    )
    (comparison_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    return {
        "ran": True, "dry_run": False,
        "resolved_experiment_ids": sorted(resolution.resolved_ids),
        "generated_pairs": [f"{a}__{b}" for a, b in pairs],
        "analysis_id": analysis_id,
        "output_dir": str(comparison_dir),
        "pair_analysis_ids": manifest["pair_analysis_ids"],
    }


def run_analysis(
    experiments: Optional[list[str]] = None, all_enabled: bool = False,
    dry_run: bool = False, force: bool = False,
    output_root: Optional[Path] = None, scope: Optional[str] = None,
) -> dict[str, Any]:
    """Top-level entry point: resolve experiments, generate all unordered
    pairs, then run (or dry-run) each pair and the multi-experiment
    comparison in one pass.

    `output_root` overrides the canonical root. `scope` names the namespace
    under it; when left None it is DERIVED from the resolved selection, so two
    different analysis scopes get two namespaces under the one canonical root
    instead of two sibling roots beside it. Pass `scope=""` to opt back into
    the legacy flat layout."""
    resolution = resolve_experiments(experiments=experiments, all_enabled=all_enabled)
    layout = resolve_layout(
        output_root, scope_key(resolution) if scope is None else (scope or None),
    )
    return run_comparison(resolution, dry_run=dry_run, force=force, layout=layout)
