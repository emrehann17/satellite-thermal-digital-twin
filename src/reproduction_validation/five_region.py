"""Task 1: independent reproduction check of the frozen five-region analysis.

WITHIN-REGION
-------------
For each cohort experiment the frozen Step8A parquet is re-loaded and the
canonical Step8B spatial-block CV is re-executed through Step8B's OWN
functions (`filter_valid_for_modeling`, `add_spatial_block_id`,
`build_population_masks`, `train_population`) with the frozen constants
STEP8B_N_SPLITS / STEP8B_SPATIAL_BLOCK_SIZE_CELLS / STEP8B_RANDOM_SEED /
STEP8B_MIN_POSITIVES_PER_POPULATION and model "random_forest". The resulting
ROC-AUC/PR-AUC for the PRIMARY population is compared against the frozen
`step8b_model_comparison_metrics.json`.

CORAL TRANSFER
--------------
For each of the 5*4 = 20 directed transfers the canonical Step10B
`generate_predictions_for_direction` is re-executed (raw_source_only,
regionwise_zscore, coral_after_regionwise_zscore -- label-blind, target X
only), then evaluated with Step10C's own `build_aligned_direction_frame` +
`compute_point_metrics`. Results are compared against the frozen
`step10_metrics.json` point metrics of the corresponding pair namespace.

No canonical output is written or modified. Target labels are loaded only by
Step10C's evaluation path, exactly as the frozen pipeline already does.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from core.paths import PROJECT_ROOT
from src.reproduction_validation.common import (
    ReproductionValidationError,
    relative_to_root,
    sha256_file,
)

SCHEMA_VERSION = "reproduction_check.v1"
OUTPUT_NAMESPACE = "reproduction_check"

# Metric families compared. Both are produced by the canonical evaluation
# functions; ROC-AUC is the manuscript-blocking one.
COMPARED_METRICS = ("roc_auc", "pr_auc")


# =============================================================================
# Frozen-value readers (read-only)
# =============================================================================
def frozen_within_region_path(experiment_id: str) -> Path:
    from src.step9a_audit_cross_region_inputs import get_experiment_output_root

    return (
        get_experiment_output_root(experiment_id)
        / "step8b"
        / "step8b_model_comparison_metrics.json"
    )


def read_frozen_within_region(experiment_id: str, primary_population: str) -> dict:
    path = frozen_within_region_path(experiment_id)
    if not path.exists():
        raise ReproductionValidationError(
            f"Frozen Step8B metrics missing for '{experiment_id}': {path}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    population_metrics = (payload.get("population_metrics") or {}).get(primary_population)
    if population_metrics is None:
        raise ReproductionValidationError(
            f"Frozen Step8B metrics for '{experiment_id}' carry no "
            f"'{primary_population}' population block: {path}."
        )
    return {
        "path": relative_to_root(path),
        "sha256": sha256_file(path),
        "baseline": population_metrics.get("overall_baseline") or {},
        "thermal": population_metrics.get("overall_thermal") or {},
    }


def _synthesis_declared_pair_namespaces() -> dict[frozenset, Path]:
    """Map each unordered cohort pair to the Step10 namespace that the FROZEN
    five-AOI synthesis manifest actually consumed.

    Several unordered pairs exist on disk under BOTH directory orderings
    (e.g. `bejis_2022__mugla_2021` and `mugla_2021__bejis_2022`), each from a
    separate Step10 execution with its own analysis_id. Probing the filesystem
    would therefore be ambiguous. The frozen synthesis manifest is the record
    of which artefact the five-region analysis of record was built from, so it
    -- not the directory listing -- resolves the pair.
    """
    from src.reproduction_validation.common import FIVE_REGION_SYNTHESIS_DIR

    manifest_path = FIVE_REGION_SYNTHESIS_DIR / "multi_aoi_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: dict[frozenset, Path] = {}
    for entry in manifest.get("resolved_inputs", []):
        if entry.get("family") != "step10_pair":
            continue
        key = frozenset((entry["source_experiment_id"], entry["target_experiment_id"]))
        namespace = (PROJECT_ROOT / entry["path"]).parent
        if key in mapping and mapping[key] != namespace:
            raise ReproductionValidationError(
                f"Frozen synthesis manifest declares two Step10 namespaces for "
                f"pair {sorted(key)}: {mapping[key]} and {namespace}."
            )
        mapping[key] = namespace
    return mapping


def resolve_frozen_pair_metrics_path(experiment_a: str, experiment_b: str) -> Path:
    """`step10_metrics.json` of the Step10 namespace the frozen five-region
    synthesis consumed for this unordered pair."""
    mapping = _synthesis_declared_pair_namespaces()
    key = frozenset((experiment_a, experiment_b))
    if key not in mapping:
        raise ReproductionValidationError(
            "The frozen five-AOI synthesis manifest declares no Step10 pair "
            f"namespace for {{{experiment_a}, {experiment_b}}}."
        )
    path = mapping[key] / "step10_metrics.json"
    if not path.exists():
        raise ReproductionValidationError(
            f"Frozen Step10 metrics missing: {relative_to_root(path)}."
        )
    return path


def read_frozen_direction_point_metrics(source_id: str, target_id: str) -> dict:
    path = resolve_frozen_pair_metrics_path(source_id, target_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    direction = f"{source_id}_to_{target_id}"
    point_metrics = (payload.get("point_metrics") or {}).get(direction)
    if point_metrics is None:
        raise ReproductionValidationError(
            f"Frozen Step10 metrics {relative_to_root(path)} carry no direction "
            f"'{direction}'. Present: {sorted((payload.get('point_metrics') or {}))}."
        )
    return {
        "path": relative_to_root(path),
        "sha256": sha256_file(path),
        "analysis_id": payload.get("analysis_id"),
        "point_metrics": point_metrics,
    }


# =============================================================================
# Independent re-execution through canonical functions
# =============================================================================
def reproduce_within_region(experiment_id: str, primary_population: str) -> dict:
    """Re-run the canonical Step8B within-region spatial-block CV."""
    from core.config import (
        STEP8B_MIN_POSITIVES_PER_POPULATION,
        STEP8B_N_SPLITS,
        STEP8B_RANDOM_SEED,
        STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
    )
    import pandas as pd

    from src.step8b_train_baseline_vs_thermal_model import (
        add_spatial_block_id,
        build_population_masks,
        filter_valid_for_modeling,
        train_population,
    )
    from src.step9a_audit_cross_region_inputs import resolve_step8a_dataset_path

    dataset_path = resolve_step8a_dataset_path(experiment_id)
    df = pd.read_parquet(dataset_path)
    df = filter_valid_for_modeling(df)
    df = add_spatial_block_id(df, STEP8B_SPATIAL_BLOCK_SIZE_CELLS)

    masks = build_population_masks(df)
    if primary_population not in masks:
        raise ReproductionValidationError(
            f"Population '{primary_population}' is not a Step8B population mask."
        )
    df_pop = df.loc[masks[primary_population]].reset_index(drop=True)

    result = train_population(
        df_pop,
        primary_population,
        STEP8B_N_SPLITS,
        STEP8B_RANDOM_SEED,
        "random_forest",
        STEP8B_MIN_POSITIVES_PER_POPULATION,
    )
    if result is None or result.get("skipped"):
        raise ReproductionValidationError(
            f"Canonical Step8B training skipped '{experiment_id}' / "
            f"'{primary_population}': {None if result is None else result.get('reason')}"
        )
    return {
        "baseline": result["overall_baseline"],
        "thermal": result["overall_thermal"],
        "n_splits_used": result.get("n_splits_used"),
        "row_count": int(len(df_pop)),
        "config": {
            "n_splits_requested": STEP8B_N_SPLITS,
            "spatial_block_size_cells": STEP8B_SPATIAL_BLOCK_SIZE_CELLS,
            "random_seed": STEP8B_RANDOM_SEED,
            "min_positives_per_population": STEP8B_MIN_POSITIVES_PER_POPULATION,
            "model": "random_forest",
            "population": primary_population,
        },
    }


def reproduce_pair_directions(experiment_a: str, experiment_b: str) -> dict:
    """Re-run canonical Step10B adaptation + Step10C evaluation for BOTH
    directed transfers of one unordered pair.

    Returns {direction: point_metrics} using the same nested
    method -> model_family -> metric layout as the frozen step10_metrics.json.
    """
    from src.step10b_label_blind_adaptation import (
        generate_predictions_for_direction,
        strip_target_to_label_blind,
    )
    from src.step10c_paired_evaluation_bootstrap import (
        build_aligned_direction_frame,
        compute_point_metrics,
    )
    from src.step9b_run_cross_region_transfer import load_step8a_dataset

    frames = {
        experiment_a: load_step8a_dataset(experiment_a),
        experiment_b: load_step8a_dataset(experiment_b),
    }

    out: dict = {}
    for source_id, target_id in ((experiment_a, experiment_b), (experiment_b, experiment_a)):
        target_x = strip_target_to_label_blind(frames[target_id])
        predictions, _adaptation_stats = generate_predictions_for_direction(
            frames[source_id], target_x, source_id, target_id,
        )
        direction = f"{source_id}_to_{target_id}"
        merged = build_aligned_direction_frame(predictions, direction, source_id, target_id)
        out[direction] = compute_point_metrics(merged)
    return out


# =============================================================================
# Comparison assembly
# =============================================================================
def _difference(reproduced, frozen) -> float | None:
    if reproduced is None or frozen is None:
        return None
    return abs(float(reproduced) - float(frozen))


def _max_case(comparisons: list[dict], metric: str = "roc_auc") -> tuple[float | None, dict | None]:
    best_value: float | None = None
    best_case: dict | None = None
    for row in comparisons:
        if row.get("metric") != metric:
            continue
        diff = row.get("absolute_difference")
        if diff is None:
            continue
        if best_value is None or diff > best_value:
            best_value = diff
            best_case = row
    return best_value, best_case


def run_five_region_reproduction_check(
    *, coral_method: str = "coral_after_regionwise_zscore", output_root: Path | None = None,
) -> dict:
    from core.step10_shared import ADAPTATION_METHODS, MODEL_FAMILIES, PRIMARY_POPULATION
    from src.step10c_paired_evaluation_bootstrap import (
        RAW_REPRODUCTION_TOLERANCE,
        WITHIN_REGION_REPRODUCTION_TOLERANCE,
    )
    from src.reproduction_validation.common import (
        DIAGNOSTICS_ROOT,
        environment,
        git_commit,
        git_status_short,
        resolve_frozen_five_region_cohort,
        step8a_dataset_record,
        utc_now,
        write_json,
    )

    if coral_method not in ADAPTATION_METHODS:
        raise ReproductionValidationError(
            f"'{coral_method}' is not a canonical Step10 adaptation method "
            f"({list(ADAPTATION_METHODS)})."
        )

    output_root = Path(output_root) if output_root else DIAGNOSTICS_ROOT / OUTPUT_NAMESPACE

    failures: list[str] = []
    notes: list[str] = []

    git_status_before = git_status_short()

    cohort_info = resolve_frozen_five_region_cohort()
    cohort = cohort_info["cohort"]

    # --- Frozen input hashes -------------------------------------------------
    input_hashes: dict[str, dict] = {}
    for experiment_id in cohort:
        record = step8a_dataset_record(experiment_id)
        input_hashes[experiment_id] = record
        if not record["hash_agrees_with_provenance"]:
            failures.append(
                f"Step8A SHA-256 for '{experiment_id}' does not agree with the "
                f"recorded provenance: observed={record['observed_sha256']}, "
                f"recorded={record['recorded_sha256_values']}."
            )

    # --- 1B: within-region ---------------------------------------------------
    within_comparisons: list[dict] = []
    for experiment_id in cohort:
        frozen = read_frozen_within_region(experiment_id, PRIMARY_POPULATION)
        reproduced = reproduce_within_region(experiment_id, PRIMARY_POPULATION)
        for model_family in MODEL_FAMILIES:
            for metric in COMPARED_METRICS:
                frozen_value = frozen[model_family].get(metric)
                reproduced_value = reproduced[model_family].get(metric)
                within_comparisons.append({
                    "experiment_id": experiment_id,
                    "model_family": model_family,
                    "metric": metric,
                    "population": PRIMARY_POPULATION,
                    "frozen_value": frozen_value,
                    "reproduced_value": reproduced_value,
                    "absolute_difference": _difference(reproduced_value, frozen_value),
                    "frozen_artifact": frozen["path"],
                    "frozen_artifact_sha256": frozen["sha256"],
                    "step8a_sha256": input_hashes[experiment_id]["observed_sha256"],
                    "n_splits_used": reproduced["n_splits_used"],
                    "row_count": reproduced["row_count"],
                })
                if frozen_value is None or reproduced_value is None:
                    failures.append(
                        f"within-region comparison incomplete for {experiment_id}/"
                        f"{model_family}/{metric}."
                    )

    expected_within = len(cohort) * len(MODEL_FAMILIES) * len(COMPARED_METRICS)
    if len(within_comparisons) != expected_within:
        failures.append(
            f"expected {expected_within} within-region comparisons, produced "
            f"{len(within_comparisons)}."
        )

    within_max, within_case = _max_case(within_comparisons, "roc_auc")

    # --- 1C: CORAL transfer --------------------------------------------------
    expected_directions = [
        (source_id, target_id)
        for source_id, target_id in itertools.permutations(sorted(cohort), 2)
    ]
    coral_comparisons: list[dict] = []
    observed_directions: set[tuple[str, str]] = set()
    missing_directions: list[str] = []

    for experiment_a, experiment_b in itertools.combinations(sorted(cohort), 2):
        reproduced_pair = reproduce_pair_directions(experiment_a, experiment_b)
        for source_id, target_id in ((experiment_a, experiment_b), (experiment_b, experiment_a)):
            direction = f"{source_id}_to_{target_id}"
            try:
                frozen = read_frozen_direction_point_metrics(source_id, target_id)
            except ReproductionValidationError as exc:
                missing_directions.append(direction)
                failures.append(str(exc))
                continue
            observed_directions.add((source_id, target_id))
            reproduced_metrics = reproduced_pair[direction]
            for model_family in MODEL_FAMILIES:
                for metric in COMPARED_METRICS:
                    frozen_value = (
                        (frozen["point_metrics"].get(coral_method) or {})
                        .get(model_family, {})
                        .get(metric)
                    )
                    reproduced_value = reproduced_metrics[coral_method][model_family].get(metric)
                    coral_comparisons.append({
                        "direction": direction,
                        "source_experiment_id": source_id,
                        "target_experiment_id": target_id,
                        "adaptation_method": coral_method,
                        "model_family": model_family,
                        "metric": metric,
                        "population": PRIMARY_POPULATION,
                        "frozen_value": frozen_value,
                        "reproduced_value": reproduced_value,
                        "absolute_difference": _difference(reproduced_value, frozen_value),
                        "frozen_artifact": frozen["path"],
                        "frozen_artifact_sha256": frozen["sha256"],
                        "frozen_analysis_id": frozen["analysis_id"],
                        "source_step8a_sha256": input_hashes[source_id]["observed_sha256"],
                        "target_step8a_sha256": input_hashes[target_id]["observed_sha256"],
                    })
                    if frozen_value is None or reproduced_value is None:
                        failures.append(
                            f"CORAL comparison incomplete for {direction}/"
                            f"{model_family}/{metric}."
                        )

    if len(observed_directions) != len(expected_directions):
        failures.append(
            f"expected {len(expected_directions)} directed CORAL transfers, "
            f"reproduced {len(observed_directions)}; missing: "
            f"{sorted(set('%s_to_%s' % d for d in expected_directions) - set('%s_to_%s' % d for d in observed_directions))}"
        )

    coral_max, coral_case = _max_case(coral_comparisons, "roc_auc")

    # Several unordered pairs carry a second, superseded Step10 namespace with
    # the reversed directory ordering and its own analysis_id. Record which
    # ones, so the choice of frozen reference is auditable.
    duplicate_namespaces = {}
    for experiment_a, experiment_b in itertools.combinations(sorted(cohort), 2):
        alternates = [
            relative_to_root(candidate)
            for candidate in (
                PROJECT_ROOT / "outputs" / "cross_region" / f"{first}__{second}" / "step10" / "step10_metrics.json"
                for first, second in ((experiment_a, experiment_b), (experiment_b, experiment_a))
            )
            if candidate.exists()
        ]
        if len(alternates) > 1:
            duplicate_namespaces[f"{experiment_a}|{experiment_b}"] = {
                "on_disk": alternates,
                "used": relative_to_root(resolve_frozen_pair_metrics_path(experiment_a, experiment_b)),
            }
    if duplicate_namespaces:
        notes.append(
            "Some unordered pairs carry two Step10 namespaces on disk (reversed "
            "directory ordering, separate analysis_id). The frozen five-AOI "
            "synthesis manifest -- not the directory listing -- selected the "
            "reference artefact; see coral_transfer.duplicate_pair_namespaces."
        )

    # --- Repository tolerance ------------------------------------------------
    tolerance = {
        "source": "src/step10c_paired_evaluation_bootstrap.py",
        "within_region_reproduction_tolerance": WITHIN_REGION_REPRODUCTION_TOLERANCE,
        "raw_reproduction_tolerance": RAW_REPRODUCTION_TOLERANCE,
        "criterion": (
            "absolute difference <= tolerance; this is the repository's own "
            "pre-existing Step10C fail-fast reproduction criterion, applied "
            "here unchanged. It was NOT chosen for this report."
        ),
        "within_region_roc_auc_within_tolerance": (
            within_max is not None and within_max <= WITHIN_REGION_REPRODUCTION_TOLERANCE
        ),
        "coral_roc_auc_within_tolerance": (
            coral_max is not None and coral_max <= RAW_REPRODUCTION_TOLERANCE
        ),
    }

    git_status_after = git_status_short()
    if git_status_before != git_status_after:
        notes.append(
            "git status changed during execution; see "
            "working_tree.status_before / status_after."
        )

    status = "PASS" if not failures else "FAIL"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_meaning": (
            "PASS means: the frozen five-region cohort resolved unambiguously, "
            "every expected within-region and directed CORAL comparison was "
            "produced, all frozen Step8A hashes agree with the recorded "
            "provenance, and the independent re-execution completed. It is a "
            "technical-completion status; the achieved numerical differences "
            "are reported separately and are also evaluated against the "
            "repository's own pre-existing 1e-6 Step10C tolerance."
        ),
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "environment": environment(),
        "cohort": cohort,
        "cohort_resolution": cohort_info,
        "primary_population": PRIMARY_POPULATION,
        "input_hashes": input_hashes,
        "tolerance_criterion": tolerance,
        "within_region": {
            "reproduction_method": (
                "src.step8b_train_baseline_vs_thermal_model.train_population "
                "re-executed on the frozen Step8A parquet"
            ),
            "expected_comparisons": expected_within,
            "observed_comparisons": len(within_comparisons),
            "comparisons": within_comparisons,
            "max_abs_roc_auc_difference": within_max,
            "max_difference_case": within_case,
        },
        "coral_transfer": {
            "reproduction_method": (
                "src.step10b_label_blind_adaptation.generate_predictions_for_direction "
                "+ src.step10c_paired_evaluation_bootstrap.build_aligned_direction_frame"
                " / compute_point_metrics"
            ),
            "adaptation_method": coral_method,
            "expected_directed_directions": len(expected_directions),
            "observed_directed_directions": len(observed_directions),
            "missing_directions": missing_directions,
            "duplicate_pair_namespaces": duplicate_namespaces,
            "comparisons": coral_comparisons,
            "max_abs_roc_auc_difference": coral_max,
            "max_difference_case": coral_case,
        },
        "working_tree": {
            "status_before": git_status_before,
            "status_after": git_status_after,
            "changed_during_execution": git_status_before != git_status_after,
        },
        "failures": failures,
        "notes": notes,
    }

    report_path = write_json(output_root / "reproduction_check.json", payload)
    payload["_report_path"] = relative_to_root(report_path)
    return payload
