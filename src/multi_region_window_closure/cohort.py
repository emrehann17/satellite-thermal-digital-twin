"""Common-cohort and fold contracts (Structure A), plus feasibility inventory.

Structure A -- verified from the frozen Manavgat implementation -- is the exact
`cell_id` INTERSECTION of the three variants' complete-case, in-population,
uncensored rows. Label and static-feature disagreements are FAILURES, never
silent removals, so `removed_label_mismatch` and
`removed_static_invariance_failure` are structurally always zero.

This module contains no model fitting. It builds the contract, the accounting
and the feasibility inventory that the fit stage is gated on.

Design reference: docs/multi_region_window_closure_design/COHORT_FEASIBILITY.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS,
    MODEL_FAMILIES,
    MultiRegionWindowClosureError,
    N_FOLDS,
    PRIMARY_POPULATION,
    VARIANTS,
    frozen_model_configuration,
)

COHORT_RULE = (
    "exact cell_id intersection of the analysis-eligible, primary-population, "
    "valid-grid rows of every variant, after removing the shared pre-label "
    "censored cells and any row missing a required feature-union value"
)

COHORT_INVENTORY_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "variant", "initial_rows",
    "removed_not_valid_for_modeling", "removed_outside_primary_population",
    "removed_prelabel_censor", "removed_missing_required_feature_union",
    "removed_variant_only_keys", "removed_label_mismatch",
    "removed_static_invariance_failure", "final_common_cohort_rows",
    "final_positive_rows", "final_negative_rows", "prevalence",
    "cohort_hash", "duplicate_cell_ids", "feasibility_pass", "failure_reason",
)

FOLD_MAPPING_COLUMNS: tuple[str, ...] = (
    "analysis_id", "aoi", "cell_id", "grid_id", "block_id", "fold_id",
    "y_true", "cohort_hash", "fold_mapping_hash",
)

#: The fourteen gates the `cohort-feasibility` stage runs before any fit.
FEASIBILITY_CHECKS: tuple[str, ...] = (
    "canonical_cohort_row_count",
    "canonical_positive_count",
    "canonical_prevalence",
    "shifted_variant_feature_completeness",
    "common_complete_rows_across_variants",
    "dropped_row_difference_between_variants",
    "label_invariance",
    "static_predictor_invariance",
    "grid_cell_identity_invariance",
    "duplicate_row_or_grid_id",
    "both_classes_in_every_fold",
    "minimum_positive_and_block_count",
    "high_prevalence_metric_and_fold_effect",
    "partial_aoi_or_variant_risk",
)


# =============================================================================
# Cohort hashing
# =============================================================================
def cohort_hash(cell_ids: Sequence[Any]) -> str:
    """SHA-256 over the sorted cell-id list.

    Must be byte-identical for the three variants of one AOI, and different
    between AOIs.
    """
    from src.step8_large_block_robustness import canonical_json, sha256_bytes

    ordered = sorted(str(c) for c in cell_ids)
    return sha256_bytes(canonical_json(ordered).encode("utf-8"))


def fold_mapping_hash(assignment: dict[Any, int]) -> str:
    """SHA-256 over the sorted `{cell_id: fold_id}` map.

    Mirrors `build_shared_spatial_folds`'s `assignment_sha256` exactly, so a
    mapping produced by the frozen helper hashes identically here.
    """
    from src.step8_large_block_robustness import canonical_json, sha256_bytes

    ordered = {str(k): int(v) for k, v in sorted(assignment.items(), key=lambda kv: str(kv[0]))}
    return sha256_bytes(canonical_json(ordered).encode("utf-8"))


# =============================================================================
# Cohort intersection (pure; operates on already-loaded frames)
# =============================================================================
def build_common_cohort(
    frames_by_variant: dict[str, Any],
    censored_cell_ids: Optional[Sequence[Any]] = None,
    feature_union: Optional[Sequence[str]] = None,
    label_column: str = "burned",
    key_column: str = "cell_id",
    static_columns: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Structure-A intersection cohort for ONE AOI.

    Fails closed -- rather than dropping rows -- on label or static-feature
    disagreement between variants, because those indicate a broken invariance
    assumption, not a data-quality issue.
    """
    import numpy as np
    import pandas as pd

    variants = sorted(frames_by_variant)
    if sorted(variants) != sorted(VARIANTS):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: VARIANT_SET_MISMATCH -- got {variants}, expected "
            f"{list(VARIANTS)}."
        )
    censored = {str(c) for c in (censored_cell_ids or ())}
    features = list(feature_union or ())

    stats: dict[str, dict[str, int]] = {}
    eligible: dict[str, Any] = {}
    for name in variants:
        frame = frames_by_variant[name]
        initial = int(len(frame))
        work = frame
        for flag in ("valid_for_modeling", "analysis_eligible"):
            if flag in work.columns:
                work = work.loc[work[flag].astype(bool)]
        after_valid = int(len(work))
        if PRIMARY_POPULATION in work.columns:
            work = work.loc[work[PRIMARY_POPULATION].astype(bool)]
        after_population = int(len(work))
        work = work.loc[~work[key_column].astype(str).isin(censored)]
        after_censor = int(len(work))
        present = [f for f in features if f in work.columns]
        if present:
            work = work.loc[~work[present].isna().any(axis=1)]
        after_features = int(len(work))
        eligible[name] = (
            work.sort_values(key_column, kind="mergesort").reset_index(drop=True)
        )
        stats[name] = {
            "initial_rows": initial,
            "removed_not_valid_for_modeling": initial - after_valid,
            "removed_outside_primary_population": after_valid - after_population,
            "removed_prelabel_censor": after_population - after_censor,
            "removed_missing_required_feature_union": after_censor - after_features,
            "after_features": after_features,
        }

    key_sets = [set(f[key_column].astype(str)) for f in eligible.values()]
    common_ids = sorted(set.intersection(*key_sets)) if key_sets else []
    if not common_ids:
        raise MultiRegionWindowClosureError(
            "BLOCKER: VARIANT_COHORT_MISMATCH -- the exact common cohort is "
            "empty; no comparison is possible."
        )

    common = {
        name: frame.loc[frame[key_column].astype(str).isin(common_ids)]
                   .sort_values(key_column, kind="mergesort").reset_index(drop=True)
        for name, frame in eligible.items()
    }
    anchor_name = variants[0]
    anchor = common[anchor_name]
    if anchor[key_column].duplicated().any():
        raise MultiRegionWindowClosureError(
            "BLOCKER: DUPLICATE_COHORT_ROW -- the common cohort carries "
            f"duplicate {key_column} values."
        )

    compare_static = [
        c for c in (static_columns or ())
        if all(c in common[name].columns for name in variants)
    ]
    for name in variants[1:]:
        other = common[name]
        if not np.array_equal(
            anchor[key_column].astype(str).to_numpy(),
            other[key_column].astype(str).to_numpy(),
        ):
            raise MultiRegionWindowClosureError(
                "BLOCKER: VARIANT_COHORT_MISMATCH -- cohort key order differs "
                f"between '{anchor_name}' and '{name}'."
            )
        if label_column in anchor.columns and label_column in other.columns:
            if not np.array_equal(
                anchor[label_column].astype(int).to_numpy(),
                other[label_column].astype(int).to_numpy(),
            ):
                raise MultiRegionWindowClosureError(
                    "BLOCKER: LABEL_INVARIANCE_VIOLATED -- the same cell "
                    f"carries different labels in '{anchor_name}' and '{name}'."
                )
        for column in compare_static:
            left, right = anchor[column], other[column]
            if pd.api.types.is_float_dtype(left) and pd.api.types.is_float_dtype(right):
                equal = np.isclose(
                    left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"),
                    rtol=0.0, atol=1e-9, equal_nan=True,
                )
            else:
                equal = (left.to_numpy() == right.to_numpy()) | (
                    left.isna().to_numpy() & right.isna().to_numpy()
                )
            if not bool(equal.all()):
                raise MultiRegionWindowClosureError(
                    "BLOCKER: STATIC_INVARIANCE_VIOLATED -- column "
                    f"'{column}' differs between '{anchor_name}' and '{name}'."
                )

    labels = anchor[label_column].astype(int).to_numpy()
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: FOLD_CLASS_INFEASIBILITY -- the common cohort carries a "
            f"single class (positives={positives}, negatives={negatives})."
        )
    digest = cohort_hash(common_ids)
    rows = int(len(anchor))
    return {
        "cell_ids": common_ids,
        "frames": common,
        "cohort_hash": digest,
        "final_common_cohort_rows": rows,
        "final_positive_rows": positives,
        "final_negative_rows": negatives,
        "prevalence": float(positives / rows) if rows else None,
        "per_variant": {
            name: {
                **stats[name],
                "removed_variant_only_keys": stats[name]["after_features"] - rows,
                # Structurally zero: a mismatch raises above rather than
                # removing, so a non-zero value here would itself be a defect.
                "removed_label_mismatch": 0,
                "removed_static_invariance_failure": 0,
            }
            for name in variants
        },
        "cohort_rule": COHORT_RULE,
    }


def cohort_inventory_rows(
    aoi: str, cohort: dict[str, Any],
) -> list[dict[str, Any]]:
    """`cohort_inventory.csv` rows for one AOI: one per variant, three total."""
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        stats = cohort["per_variant"][variant]
        out.append({
            "aoi": aoi,
            "variant": variant,
            "initial_rows": stats["initial_rows"],
            "removed_not_valid_for_modeling": stats["removed_not_valid_for_modeling"],
            "removed_outside_primary_population": stats["removed_outside_primary_population"],
            "removed_prelabel_censor": stats["removed_prelabel_censor"],
            "removed_missing_required_feature_union": stats[
                "removed_missing_required_feature_union"
            ],
            "removed_variant_only_keys": stats["removed_variant_only_keys"],
            "removed_label_mismatch": stats["removed_label_mismatch"],
            "removed_static_invariance_failure": stats["removed_static_invariance_failure"],
            "final_common_cohort_rows": cohort["final_common_cohort_rows"],
            "final_positive_rows": cohort["final_positive_rows"],
            "final_negative_rows": cohort["final_negative_rows"],
            "prevalence": cohort["prevalence"],
            "cohort_hash": cohort["cohort_hash"],
            "duplicate_cell_ids": 0,
            "feasibility_pass": True,
            "failure_reason": None,
        })
    return out


def assert_cohort_accounting(rows: Sequence[dict[str, Any]]) -> None:
    """Row-level accounting must reconcile, and invariance counters must be zero."""
    for row in rows:
        removals = (
            row["removed_not_valid_for_modeling"]
            + row["removed_outside_primary_population"]
            + row["removed_prelabel_censor"]
            + row["removed_missing_required_feature_union"]
            + row["removed_variant_only_keys"]
        )
        if row["initial_rows"] - removals != row["final_common_cohort_rows"]:
            raise MultiRegionWindowClosureError(
                "BLOCKER: COHORT_ACCOUNTING_INCONSISTENT -- "
                f"{row['aoi']}/{row['variant']}: {row['initial_rows']} - "
                f"{removals} != {row['final_common_cohort_rows']}."
            )
        for field in ("removed_label_mismatch", "removed_static_invariance_failure"):
            if row[field]:
                raise MultiRegionWindowClosureError(
                    f"BLOCKER: {field.upper()} -- {row['aoi']}/{row['variant']} "
                    f"records {row[field]}; invariance failures raise rather "
                    "than remove, so a non-zero value indicates a defect."
                )
        if row["final_positive_rows"] + row["final_negative_rows"] != row[
            "final_common_cohort_rows"
        ]:
            raise MultiRegionWindowClosureError(
                "BLOCKER: COHORT_ACCOUNTING_INCONSISTENT -- positives + "
                f"negatives != rows for {row['aoi']}/{row['variant']}."
            )

    by_aoi: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_aoi.setdefault(row["aoi"], []).append(row)
    for aoi, aoi_rows in sorted(by_aoi.items()):
        if len(aoi_rows) != len(VARIANTS):
            raise MultiRegionWindowClosureError(
                f"BLOCKER: MISSING_VARIANT -- {aoi} has {len(aoi_rows)} cohort "
                f"rows, expected {len(VARIANTS)}."
            )
        digests = {r["cohort_hash"] for r in aoi_rows}
        if len(digests) != 1:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: VARIANT_COHORT_MISMATCH -- {aoi} variants do not "
                f"share one cohort_hash: {sorted(digests)}."
            )


def assert_distinct_cohorts_across_aois(rows: Sequence[dict[str, Any]]) -> None:
    """Different AOIs must not share a cohort hash -- that would mean reuse."""
    seen: dict[str, str] = {}
    for row in rows:
        digest, aoi = row["cohort_hash"], row["aoi"]
        if digest in seen and seen[digest] != aoi:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: FOLD_HASH_COLLISION -- {seen[digest]} and {aoi} "
                "share a cohort_hash; one AOI's cohort was reused for another."
            )
        seen[digest] = aoi


# =============================================================================
# Fold contract
# =============================================================================
def assert_fold_mapping(
    aoi: str,
    assignment: dict[Any, int],
    labels_by_cell: dict[Any, int],
    block_by_cell: Optional[dict[Any, Any]] = None,
    n_folds: int = N_FOLDS,
) -> dict[str, Any]:
    """One shared fold assignment per AOI. Fails closed on every violation."""
    if not assignment:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: FOLD_ASSIGNMENT_INCOMPLETE -- {aoi} has no fold mapping."
        )
    folds = sorted({int(v) for v in assignment.values()})
    if folds != list(range(n_folds)):
        raise MultiRegionWindowClosureError(
            f"BLOCKER: FOLD_ASSIGNMENT_INCOMPLETE -- {aoi} fold ids {folds}, "
            f"expected {list(range(n_folds))}."
        )
    missing = sorted(set(labels_by_cell) - set(assignment))
    if missing:
        raise MultiRegionWindowClosureError(
            f"BLOCKER: FOLD_ASSIGNMENT_INCOMPLETE -- {aoi}: {len(missing)} "
            "cohort row(s) received no validation fold."
        )
    per_fold: dict[int, dict[int, int]] = {f: {0: 0, 1: 0} for f in folds}
    for cell, fold in assignment.items():
        label = int(labels_by_cell[cell])
        per_fold[int(fold)][1 if label else 0] += 1
    for fold, counts in sorted(per_fold.items()):
        if counts[0] == 0 or counts[1] == 0:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: FOLD_CLASS_INFEASIBILITY -- {aoi} fold {fold} has "
                f"positives={counts[1]}, negatives={counts[0]}; every required "
                "evaluation fold must contain both classes."
            )
    if block_by_cell:
        blocks_per_fold: dict[int, set] = {}
        for cell, fold in assignment.items():
            blocks_per_fold.setdefault(int(fold), set()).add(block_by_cell[cell])
        split = sorted({
            block
            for a in blocks_per_fold for b in blocks_per_fold if a < b
            for block in blocks_per_fold[a] & blocks_per_fold[b]
        })
        if split:
            raise MultiRegionWindowClosureError(
                f"BLOCKER: FOLD_REOPTIMISED -- {aoi} spatial block(s) "
                f"{split[:6]} are split across validation folds."
            )
    return {
        "aoi": aoi,
        "fold_count": len(folds),
        "fold_mapping_hash": fold_mapping_hash(assignment),
        "rows_per_fold": {f: sum(per_fold[f].values()) for f in folds},
        "positives_per_fold": {f: per_fold[f][1] for f in folds},
        "negatives_per_fold": {f: per_fold[f][0] for f in folds},
        "unique_block_count": (
            len(set(block_by_cell.values())) if block_by_cell else None
        ),
    }


# =============================================================================
# Feasibility inventory (read-only; no fit)
# =============================================================================
def canonical_feasibility_inventory(
    aois: Sequence[str] = ACTUAL_AOIS,
    experiments_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Read-only canonical-cohort feasibility evidence, per AOI.

    Reads only the canonical Step8A parquet -- an input this analysis already
    hashes -- and applies the eligibility, population and feature-completeness
    filters. It performs no fit and writes nothing.
    """
    import pandas as pd

    from src.multi_region_window_closure.inputs import canonical_step8a_path

    config = frozen_model_configuration()
    features = _feature_union()
    out: dict[str, Any] = {}
    for aoi in aois:
        path = canonical_step8a_path(aoi, experiments_root)
        if not path.is_file():
            out[aoi] = {
                "aoi": aoi, "available": False, "path": str(path),
                "failure_reason": "canonical Step8A dataset is missing",
            }
            continue
        frame = pd.read_parquet(path)
        work = frame
        for flag in ("valid_for_modeling", "analysis_eligible"):
            if flag in work.columns:
                work = work.loc[work[flag].astype(bool)]
        eligible_rows = int(len(work))
        if PRIMARY_POPULATION in work.columns:
            work = work.loc[work[PRIMARY_POPULATION].astype(bool)]
        population_rows = int(len(work))
        present = [f for f in features if f in work.columns]
        absent = sorted(set(features) - set(work.columns))
        complete = work.loc[~work[present].isna().any(axis=1)] if present else work
        rows = int(len(complete))
        positives = int(complete["burned"].astype(int).sum()) if "burned" in complete else 0
        out[aoi] = {
            "aoi": aoi,
            "available": True,
            "path": str(path),
            "step8a_rows": int(len(frame)),
            "eligible_rows": eligible_rows,
            "population_rows": population_rows,
            "complete_case_rows": rows,
            "complete_case_positives": positives,
            "complete_case_negatives": rows - positives,
            "prevalence": float(positives / rows) if rows else None,
            "missing_feature_columns": absent,
            "min_positives_required": config["min_positives"],
            "meets_min_positives": positives >= config["min_positives"],
            "n_folds": config["n_splits"],
            "failure_reason": None,
        }
    return out


def _feature_union() -> list[str]:
    from src.multi_region_window_closure.contract import feature_registry

    return list(feature_registry()["feature_union"])


def feasibility_blockers(inventory: dict[str, Any]) -> list[str]:
    """Human-readable feasibility blockers, empty when the gate would pass."""
    problems: list[str] = []
    for aoi, record in sorted(inventory.items()):
        if not record.get("available"):
            problems.append(f"{aoi}: {record.get('failure_reason')}")
            continue
        if record["missing_feature_columns"]:
            problems.append(
                f"{aoi}: canonical Step8A is missing feature column(s) "
                f"{record['missing_feature_columns']}"
            )
        if not record["meets_min_positives"]:
            problems.append(
                f"{aoi}: {record['complete_case_positives']} positives < "
                f"min_positives {record['min_positives_required']}"
            )
        if record["complete_case_negatives"] <= 0:
            problems.append(f"{aoi}: no negative rows in the complete-case cohort")
    return problems


def expected_oof_rows(cohort_rows_by_aoi: dict[str, int]) -> int:
    """`sum_aoi (N_aoi x variants x model families)`."""
    return sum(
        int(n) * len(VARIANTS) * len(MODEL_FAMILIES)
        for n in cohort_rows_by_aoi.values()
    )
