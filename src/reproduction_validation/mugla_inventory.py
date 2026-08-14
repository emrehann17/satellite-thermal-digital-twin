"""Task 2: READ-ONLY output inventory for the Muglas 2021 <-> 2022 transfer pair.

This module RUNS NOTHING. It discovers the frozen artefacts that already
exist locally for both directed transfers

    mugla_2021               -> mugla_2022_event_relative
    mugla_2022_event_relative -> mugla_2021

verifies the two frozen Step8A parquets against the manuscript author's
expected SHA-256 digests, checks that the frozen Step9/Step10 provenance
records reference exactly those digests, extracts the ROC-AUC point estimates
and the canonical 95% target spatial-block bootstrap CIs at full precision,
and compares them against the author's reference values.

No model is fitted, no bootstrap is drawn, no file outside
`outputs/diagnostics/mugla_transfer_inventory/` is written.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.paths import PROJECT_ROOT
from src.reproduction_validation.common import relative_to_root, sha256_file

SCHEMA_VERSION = "mugla_transfer_inventory.v1"
OUTPUT_NAMESPACE = "mugla_transfer_inventory"

EXPERIMENT_A = "mugla_2021"
EXPERIMENT_B = "mugla_2022_event_relative"

DIRECTION_A_TO_B = f"{EXPERIMENT_A}_to_{EXPERIMENT_B}"
DIRECTION_B_TO_A = f"{EXPERIMENT_B}_to_{EXPERIMENT_A}"
EXPECTED_DIRECTIONS = (DIRECTION_A_TO_B, DIRECTION_B_TO_A)

# Step8A SHA-256 digests supplied by the manuscript author. Verified, never
# regenerated: a mismatch is reported, not repaired.
EXPECTED_STEP8A_SHA256 = {
    EXPERIMENT_A: "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
    EXPERIMENT_B: "7c545f4da8fa7f8973575400862595d6ef85a8d55b9b25bc16fc54aba23a1d52",
}

# Manuscript-author reference values (Python 3.12.3, scikit-learn 1.9.0,
# pandas 3.0.5, numpy 2.5.2). COMPARISON ONLY.
AUTHOR_REFERENCE = {
    DIRECTION_A_TO_B: {
        "baseline_roc_auc": {"point": 0.642, "ci": [0.606, 0.674]},
        "thermal_roc_auc": {"point": 0.559, "ci": [0.513, 0.604]},
        "delta_roc_auc": {"point": -0.082, "ci": [-0.127, -0.040]},
    },
    DIRECTION_B_TO_A: {
        "baseline_roc_auc": {"point": 0.581, "ci": [0.566, 0.598]},
        "thermal_roc_auc": {"point": 0.670, "ci": [0.654, 0.685]},
        "delta_roc_auc": {"point": 0.089, "ci": [0.072, 0.104]},
    },
}
AUTHOR_REFERENCE_ENVIRONMENT = {
    "python": "3.12.3", "scikit_learn": "1.9.0",
    "pandas": "3.0.5", "numpy": "2.5.2",
}

# Artefacts that must all be present for a direction to count as COMPLETE.
REQUIRED_ARTEFACTS = (
    ("step9a", "cross_region_input_audit.json"),
    ("step9b", "cross_region_transfer_metrics.json"),
    ("step9b", "cross_region_transfer_predictions.parquet"),
    ("step9c", "cross_region_bootstrap_metrics.json"),
    ("step9d", "final_cross_region_report.json"),
)


def pair_root() -> Path:
    return PROJECT_ROOT / "outputs" / "cross_region" / f"{EXPERIMENT_A}__{EXPERIMENT_B}"


# =============================================================================
# Discovery
# =============================================================================
def discover_artefacts() -> dict:
    """Discover every local artefact that names either directed transfer.

    Nothing about the directory layout is assumed: the cross-region tree, the
    diagnostics tree and the archives tree are all scanned for directories
    naming the pair, and every file under them is inventoried.
    """
    roots = []
    for search_root in (
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "archives",
    ):
        if not search_root.exists():
            continue
        for candidate in sorted(search_root.rglob("*")):
            if not candidate.is_dir():
                continue
            name = candidate.name
            if EXPERIMENT_A in name and EXPERIMENT_B in name:
                roots.append(candidate)

    # Drop nested duplicates (keep the outermost matching directory).
    top_roots: list[Path] = []
    for candidate in roots:
        if not any(candidate != other and other in candidate.parents for other in roots):
            top_roots.append(candidate)

    inventory = {}
    for root in top_roots:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        inventory[relative_to_root(root)] = {
            "file_count": len(files),
            "files": [relative_to_root(path) for path in files],
        }
    return inventory


# =============================================================================
# Frozen Step8A verification
# =============================================================================
def verify_step8a_inputs() -> dict:
    from src.step9a_audit_cross_region_inputs import resolve_step8a_dataset_path

    records = {}
    mismatches = []
    for experiment_id, expected in EXPECTED_STEP8A_SHA256.items():
        path = resolve_step8a_dataset_path(experiment_id)
        exists = Path(path).exists()
        observed = sha256_file(path) if exists else None
        match = observed == expected
        records[experiment_id] = {
            "path": relative_to_root(path),
            "exists": exists,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "match": match,
            "size_bytes": Path(path).stat().st_size if exists else None,
        }
        if not match:
            mismatches.append(
                f"{experiment_id}: expected {expected}, observed {observed}"
            )
    return {"records": records, "mismatches": mismatches, "all_match": not mismatches}


def verify_provenance_binding(step9b_payload: dict, step9c_payload: dict) -> dict:
    """Confirm the frozen Step9B metrics reference exactly the verified
    Step8A digests, and that Step9C's bootstrap consumed the Step9B
    predictions file that is on disk now."""
    resolved = step9b_payload.get("resolved_inputs") or {}
    bindings = {}
    problems = []
    for experiment_id, expected in EXPECTED_STEP8A_SHA256.items():
        record = resolved.get(experiment_id) or {}
        recorded = record.get("dataset_sha256")
        bindings[experiment_id] = {
            "recorded_in_step9b": recorded,
            "expected": expected,
            "match": recorded == expected,
            "dataset_path": record.get("dataset_path"),
            "step8a_manifest_sha256": record.get("step8a_manifest_sha256"),
        }
        if recorded != expected:
            problems.append(
                f"Step9B provenance for '{experiment_id}' records {recorded}, "
                f"expected {expected}."
            )

    predictions_path = step9c_payload.get("predictions_path")
    recorded_predictions_sha = step9c_payload.get("predictions_sha256")
    observed_predictions_sha = (
        sha256_file(Path(predictions_path)) if predictions_path else None
    )
    predictions_binding = {
        "path": predictions_path,
        "recorded_sha256": recorded_predictions_sha,
        "observed_sha256": observed_predictions_sha,
        "match": recorded_predictions_sha == observed_predictions_sha,
    }
    if not predictions_binding["match"]:
        problems.append(
            "Step9C records a Step9B predictions SHA-256 that does not match the "
            f"file on disk ({recorded_predictions_sha} vs {observed_predictions_sha})."
        )

    return {
        "step8a_bindings": bindings,
        "step9b_predictions_binding": predictions_binding,
        "problems": problems,
        "consistent": not problems,
    }


# =============================================================================
# Value extraction
# =============================================================================
def extract_results(primary_population: str) -> dict:
    root = pair_root()
    step9b = json.loads(
        (root / "step9b" / "cross_region_transfer_metrics.json").read_text(encoding="utf-8")
    )
    step9c = json.loads(
        (root / "step9c" / "cross_region_bootstrap_metrics.json").read_text(encoding="utf-8")
    )

    results: dict = {}
    for record in step9b.get("results", []):
        if record.get("population") != primary_population or record.get("skipped"):
            continue
        direction = record["transfer_direction"]
        results[direction] = {
            "population": primary_population,
            "source_experiment_id": record["source_experiment_id"],
            "target_experiment_id": record["target_experiment_id"],
            "source_cell_count": record["source_cell_count"],
            "target_cell_count": record["target_cell_count"],
            "target_positive_count": record["target_positive_count"],
            "target_negative_count": record["target_negative_count"],
            "baseline_roc_auc": record["baseline_metrics"]["roc_auc"],
            "thermal_roc_auc": record["thermal_metrics"]["roc_auc"],
            "baseline_pr_auc": record["baseline_metrics"]["pr_auc"],
            "thermal_pr_auc": record["thermal_metrics"]["pr_auc"],
            "delta_roc_auc": record["delta_metrics"]["delta_auc"],
        }

    for group in step9c.get("groups", []):
        if group.get("population") != primary_population:
            continue
        direction = group["transfer_direction"]
        if direction not in results:
            continue
        intervals = group.get("confidence_intervals") or {}
        results[direction]["n_successful_replicates"] = group.get("n_successful_replicates")
        results[direction]["n_target_blocks"] = group.get("n_target_blocks")
        for metric in ("baseline_roc_auc", "thermal_roc_auc", "delta_roc_auc"):
            block = intervals.get(metric) or {}
            results[direction][f"{metric}_ci_95"] = [block.get("ci_2_5"), block.get("ci_97_5")]
            results[direction][f"{metric}_bootstrap_mean"] = block.get("mean")
        results[direction]["delta_roc_auc_interpretation"] = (
            (intervals.get("delta_roc_auc") or {}).get("interpretation")
        )
    return {"step9b": step9b, "step9c": step9c, "results": results}


def check_completeness() -> dict:
    root = pair_root()
    present = {}
    missing = []
    for stage, filename in REQUIRED_ARTEFACTS:
        path = root / stage / filename
        present[f"{stage}/{filename}"] = {
            "path": relative_to_root(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
        }
        if not path.exists():
            missing.append(relative_to_root(path))
    return {"required_artefacts": present, "missing": missing, "complete": not missing}


# =============================================================================
# Comparison
# =============================================================================
def compare_with_reference(results: dict) -> tuple[list[dict], list[str]]:
    comparisons: list[dict] = []
    problems: list[str] = []

    for direction, metrics in AUTHOR_REFERENCE.items():
        if direction not in results:
            problems.append(f"direction '{direction}' has no local primary-population result.")
            continue
        local = results[direction]
        for metric, reference in metrics.items():
            point = local.get(metric)
            bootstrap_mean = local.get(f"{metric}_bootstrap_mean")
            low, high = local.get(f"{metric}_ci_95") or [None, None]

            comparisons.append({
                "direction": direction,
                "metric": metric,
                "quantity": "point_estimate",
                "author_reference": reference["point"],
                "local_value": point,
                "absolute_difference": _difference(point, reference["point"]),
                "local_bootstrap_mean": bootstrap_mean,
                "absolute_difference_vs_bootstrap_mean": _difference(
                    bootstrap_mean, reference["point"]
                ),
            })
            for index, (bound_name, local_bound) in enumerate(
                (("ci_low", low), ("ci_high", high))
            ):
                comparisons.append({
                    "direction": direction,
                    "metric": metric,
                    "quantity": bound_name,
                    "author_reference": reference["ci"][index],
                    "local_value": local_bound,
                    "absolute_difference": _difference(local_bound, reference["ci"][index]),
                })
            if point is None:
                problems.append(f"{direction}/{metric}: no local point estimate.")
    return comparisons, problems


def _difference(local, reference) -> float | None:
    if local is None or reference is None:
        return None
    return abs(float(local) - float(reference))


# =============================================================================
# Report assembly
# =============================================================================
def build_inventory(*, output_root: Path | None = None) -> dict:
    from src.reproduction_validation.common import (
        DIAGNOSTICS_ROOT, environment, git_commit, git_status_short,
        utc_now, write_json,
    )
    from src.step9a_audit_cross_region_inputs import PRIMARY_POPULATIONS

    primary_population = PRIMARY_POPULATIONS[0]
    output_root = Path(output_root) if output_root else DIAGNOSTICS_ROOT / OUTPUT_NAMESPACE

    problems: list[str] = []
    notes: list[str] = []

    artefacts = discover_artefacts()
    inputs = verify_step8a_inputs()
    problems.extend(inputs["mismatches"])

    completeness = check_completeness()
    if not completeness["complete"]:
        problems.append(
            "missing required artefacts: " + ", ".join(completeness["missing"])
        )

    extracted = extract_results(primary_population)
    results = extracted["results"]
    for direction in EXPECTED_DIRECTIONS:
        if direction not in results:
            problems.append(f"direction '{direction}' absent from the frozen Step9B results.")

    provenance = verify_provenance_binding(extracted["step9b"], extracted["step9c"])
    problems.extend(provenance["problems"])

    comparisons, comparison_problems = compare_with_reference(results)
    problems.extend(comparison_problems)

    rerun_required = bool(problems)
    if rerun_required:
        status = "BLOCKED"
    else:
        status = "EXISTING_COMPLETE"
        notes.append(
            "Both directed Muglas transfers already existed locally, complete and "
            "bound to the verified frozen Step8A digests. No model was rerun."
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "mugla_output_status": status,
        "mugla_rerun_required": "YES" if rerun_required else "NO",
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "environment": environment(),
        "author_reference_environment": AUTHOR_REFERENCE_ENVIRONMENT,
        "pair_root": relative_to_root(pair_root()),
        "expected_directions": list(EXPECTED_DIRECTIONS),
        "observed_directions": sorted(results),
        "discovered_artefact_trees": artefacts,
        "required_artefact_completeness": completeness,
        "step8a_input_verification": inputs,
        "provenance_binding": provenance,
        "primary_population": primary_population,
        "frozen_scientific_configuration": {
            "random_seed": extracted["step9b"].get("random_seed"),
            "spatial_cv_n_splits_requested": extracted["step9b"].get(
                "spatial_cv_n_splits_requested"
            ),
            "spatial_block_size_cells": extracted["step9b"].get("spatial_block_size_cells"),
            "spatial_block_definition": extracted["step9b"].get("spatial_block_definition"),
            "model_name": extracted["step9b"].get("model_name"),
            "model_parameters": extracted["step9b"].get("model_parameters"),
            "preprocessing_parameters": extracted["step9b"].get("preprocessing_parameters"),
            "source_only": extracted["step9b"].get("source_only"),
            "bidirectional": extracted["step9b"].get("bidirectional"),
            "baseline_features": extracted["step9b"].get("baseline_features"),
            "thermal_model_features": extracted["step9b"].get("thermal_model_features"),
            "step9b_git_commit": extracted["step9b"].get("git_commit"),
            "step9b_created_at": extracted["step9b"].get("created_at"),
            "bootstrap_replicates": extracted["step9c"].get(
                "n_bootstrap_replicates_requested"
            ),
            "bootstrap_seed": extracted["step9c"].get("random_seed"),
            "bootstrap_unit": extracted["step9c"].get("resampling_unit"),
            "bootstrap_scheme": extracted["step9c"].get("resampling_scheme"),
            "percentile_interval": extracted["step9c"].get("percentile_interval"),
            "step9c_git_commit": extracted["step9c"].get("git_commit"),
            "step9c_created_at": extracted["step9c"].get("created_at"),
        },
        "local_results": results,
        "comparisons": comparisons,
        "working_tree_status": git_status_short(),
        "problems": problems,
        "notes": notes,
    }

    report_path = write_json(output_root / "mugla_transfer_inventory.json", payload)
    payload["_report_path"] = relative_to_root(report_path)
    return payload


def render_markdown(payload: dict) -> str:
    lines = [
        "# Muglas 2021 <-> 2022 transfer output inventory (read-only)",
        "",
        f"- MUGLA_OUTPUT_STATUS: `{payload['mugla_output_status']}`",
        f"- MUGLA_RERUN_REQUIRED: `{payload['mugla_rerun_required']}`",
        f"- pair root: `{payload['pair_root']}`",
        f"- primary population: `{payload['primary_population']}`",
        f"- inspected at git commit: `{payload.get('git_commit')}`",
        "",
        "## Frozen Step8A verification",
        "",
        "| experiment | expected SHA-256 | observed SHA-256 | match |",
        "| --- | --- | --- | --- |",
    ]
    for experiment_id, record in payload["step8a_input_verification"]["records"].items():
        lines.append(
            f"| {experiment_id} | `{record['expected_sha256']}` | "
            f"`{record['observed_sha256']}` | {record['match']} |"
        )
    lines += [
        "",
        "## Comparison against the manuscript-author reference",
        "",
        "| direction | metric | quantity | author reference | local value | abs difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["comparisons"]:
        difference = row["absolute_difference"]
        local = row["local_value"]
        lines.append(
            f"| {row['direction']} | {row['metric']} | {row['quantity']} | "
            f"{row['author_reference']} | "
            f"{'' if local is None else format(local, '.6f')} | "
            f"{'' if difference is None else format(difference, '.6f')} |"
        )
    if payload["problems"]:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in payload["problems"]]
    if payload["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in payload["notes"]]
    return "\n".join(lines) + "\n"
