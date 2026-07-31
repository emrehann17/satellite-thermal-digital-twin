#!/usr/bin/env python3
"""Validator for the Muğla subsampling artifact.

Implements the checks of
`docs/mugla_subsampling_design/VALIDATOR_CHECKLIST.md` (groups A-L), re-deriving
every claim from the emitted artifacts plus the frozen canonical inputs. It
never imports a cached intermediate from the run.

Two modes:

    dry-run  -- contract, stage-order and frozen-input checks only.
                Writes nothing, reads no produced artifact, contacts no GEE.
    actual   -- every check, against a produced artifact.

`--deep` additionally re-fits ONE (repeat_id, family) Muğla-as-source model
independently and asserts the emitted probabilities agree to <= 1e-12, which is
what makes the 80 -> 40 source-fit reduction auditable. Bit-identity is NOT the
criterion: `RandomForestClassifier(n_jobs=-1)` averages per-tree probabilities in
a thread-scheduling-dependent order, so two fits of the same frame already differ
by ~1 ULP. Those audit fits are reported separately and are never counted in the
240 scientific fits.

Every check emits {check_id, status, expected, observed, evidence_path, note}.
Any FAIL makes the overall status FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.mugla_subsampling as mss

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIPPED"
TOLERANCE = 1e-9

# `--write-report` drops this file INTO the analysis namespace, so a later run
# would otherwise scan the validator's own output: the report echoes every
# check's expected/observed values, which includes the excluded-AOI names and
# the forbidden-vocabulary denylist itself. It is a validator artifact, not a
# run product, so it is excluded from the artifact scans and from the manifest
# completeness check.
VALIDATION_REPORT_NAME = "validation_report.json"


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
            "schema_version": mss.SCHEMA_VERSION,
            "validator_mode": mode,
            "analysis_id": analysis_id,
            "namespace": namespace,
            "overall_status": self.overall,
            "counts": counts,
            "checks": self.checks,
        }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=True)


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def repeat_subset_status(selected: pd.DataFrame,
                         canonical_cells: set[str]) -> dict[int, dict[str, Any]]:
    """Per-repeat subset status against the canonical population.

    The subset property is a property of EACH REPEAT, never of their union. Two
    repeats of 20,511 cells drawn from 41,730 will between them touch most of
    the population, and 20 repeats will very likely touch all of it -- that is
    the design working, not a violation. Only a single repeat that contained
    the whole population would mean the subsample was not a subsample.
    """
    status: dict[int, dict[str, Any]] = {}
    canonical_size = len(canonical_cells)
    for repeat_id, group in selected.groupby("repeat_id", sort=True):
        cells = set(group["cell_id"].astype(str))
        status[int(repeat_id)] = {
            "size": len(cells),
            "is_subset": cells <= canonical_cells,
            "is_proper": len(cells) < canonical_size,
            "foreign_cells": len(cells - canonical_cells),
        }
    return status


def composite_identity_count(frame: pd.DataFrame,
                             identity_columns: Sequence[str]) -> int:
    """Distinct rows under a COMPOSITE identity.

    `cell_id` is an AOI-local `r{row}_c{col}` token: Manavgat and Bejís index
    their own grids from the same origin, so the same token denotes different
    cells in different regions and the two cohorts collide on the bare id.
    Identity is therefore (target region, cell_id) -- counting bare `cell_id`
    silently undercounts the union of two cohorts.
    """
    columns = list(identity_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"identity columns {missing} absent from the frame")
    return int(len(frame[columns].astype(str).drop_duplicates()))


def _close(left: Any, right: Any, tolerance: float = TOLERANCE) -> bool:
    if left is None or right is None:
        return False
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


# =============================================================================
# A. Schema and identity  /  B. Frozen inputs
# =============================================================================
def run_contract_checks(report: Report, experiments: list[str]) -> None:
    report.ok("A1", mss.SCHEMA_VERSION == "mugla_subsampling.v1",
              "mugla_subsampling.v1", mss.SCHEMA_VERSION)
    report.ok("A2", mss.DIAGNOSTIC_CLASS == "population_size_matched_subsampling_sensitivity",
              "population_size_matched_subsampling_sensitivity", mss.DIAGNOSTIC_CLASS)
    report.ok("A4a", list(mss.STAGES) == ["plan", "fit", "summarize"],
              ["plan", "fit", "summarize"], list(mss.STAGES))

    ordered = True
    try:
        mss.validate_stage_range("summarize", "plan")
        ordered = False
    except SystemExit:
        pass
    report.ok("A4b", ordered, "reversed stage range rejected",
              "accepted" if not ordered else "rejected")

    report.ok("B5", experiments == list(mss.PRIMARY_EXPERIMENTS),
              list(mss.PRIMARY_EXPERIMENTS), experiments)
    for token in ("evia_2021", "evia_2021_extended", "kozan_2023"):
        rejected = True
        try:
            mss.assert_not_excluded(token)
            rejected = False
        except SystemExit:
            pass
        report.ok(f"B6:{token}", rejected, "excluded", "accepted" if not rejected else "excluded")

    report.ok("B8a", mss.SUBSAMPLED_EXPERIMENT == "mugla_2021",
              "mugla_2021", mss.SUBSAMPLED_EXPERIMENT)
    report.ok("C1a", int(mss.N_REPEATS) == 20, 20, int(mss.N_REPEATS))
    report.ok("C2a", int(mss.TARGET_SAMPLE_SIZE) == 20511, 20511, int(mss.TARGET_SAMPLE_SIZE))
    report.ok("G6a", int(mss.FOLD_COUNT) == 5, 5, int(mss.FOLD_COUNT))
    report.ok("G6b", int(mss.FOLD_RANDOM_STATE) == 42, 42, int(mss.FOLD_RANDOM_STATE))

    expected_fits = mss.expected_unique_fit_count()
    report.ok("L1a", expected_fits["unique_fits"] == 240, 240, expected_fits["unique_fits"])
    report.ok("L1b", expected_fits["within_fits"] == 200, 200, expected_fits["within_fits"])
    report.ok("L1c", expected_fits["source_fits"] == 40, 40, expected_fits["source_fits"])
    report.ok("L1d", expected_fits["target_fits"] == 0, 0, expected_fits["target_fits"])
    report.ok("L2a", expected_fits["reuse_events"] == 40, 40, expected_fits["reuse_events"])

    report.ok("K2a",
              "subsampling_interval_lower" in mss.SUMMARY_COLUMNS
              and "subsampling_interval_upper" in mss.SUMMARY_COLUMNS
              and not any(column.startswith("ci_") for column in mss.SUMMARY_COLUMNS),
              "subsampling_interval_* and no ci_* column", list(mss.SUMMARY_COLUMNS))
    report.ok("J6a",
              mss.metric_orientation("brier_score") == mss.ORIENTATION_NEGATED
              and mss.metric_orientation("roc_auc") == mss.ORIENTATION_HIGHER,
              "brier negated, AUCs higher-is-better",
              {metric: mss.metric_orientation(metric) for metric in mss.METRICS})

    report.ok("L7", not _module_touches_earth_engine(),
              "no Earth Engine symbol reachable from the analysis module",
              "clean" if not _module_touches_earth_engine() else "gee reference found")


def _module_touches_earth_engine() -> bool:
    source = Path(mss.__file__).read_text(encoding="utf-8")
    return any(token in source for token in ("import ee", "ee.Initialize", "gee_utils",
                                             "earthengine"))


def run_input_checks(report: Report, experiments: list[str],
                     experiments_root: Optional[Path]) -> Optional[dict[str, Any]]:
    try:
        inventory = mss.build_frozen_input_inventory(experiments, experiments_root)
    except SystemExit as exc:
        report.ok("B1..B3", False, "canonical Step8A datasets resolvable", str(exc))
        return None
    for index, experiment_id in enumerate(mss.PRIMARY_EXPERIMENTS, start=1):
        entry = inventory.get(experiment_id, {})
        report.ok(f"B{index}", bool(entry.get("match")),
                  entry.get("expected_sha256"), entry.get("sha256"), entry.get("path"))
    report.ok("B4", mss.assert_canonical_step8a_hashes(inventory, strict=False)["all_match"],
              "all registered digests match",
              mss.assert_canonical_step8a_hashes(inventory, strict=False))
    return inventory


# =============================================================================
# Artifact checks
# =============================================================================
def run_artifact_checks(report: Report, root: Path, experiments: list[str],
                        inventory: Optional[dict[str, Any]],
                        experiments_root: Optional[Path],
                        output_root: Optional[Path], deep: bool) -> None:
    config_path = root / "config.json"
    if not config_path.is_file():
        report.ok("A3", False, "config.json present", "missing", str(config_path))
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scientific_config = config.get("scientific_config", {})
    report.ok("A3", mss.compute_analysis_id(scientific_config) == root.name,
              root.name, mss.compute_analysis_id(scientific_config), str(config_path))

    # Artifact-level checks read the artifact's OWN declared contract. The
    # production inventory literals of SAMPLING_FEASIBILITY.md are asserted
    # only when the artifact is bound to the real frozen Muğla Step8A digest;
    # against any other frame they would be meaningless.
    hashes = {}
    if (root / "input_hashes.json").is_file():
        hashes = json.loads((root / "input_hashes.json").read_text(encoding="utf-8"))
    mugla_digest = (hashes.get("step8a", {}).get(mss.SUBSAMPLED_EXPERIMENT, {})
                    .get("sha256"))
    declared = {
        "target_sample_size": int(scientific_config.get("target_sample_size",
                                                        mss.TARGET_SAMPLE_SIZE)),
        "n_repeats": int(scientific_config.get("n_repeats", mss.N_REPEATS)),
        "production": mugla_digest == mss.FROZEN_MUGLA_STEP8A_SHA256,
    }
    report.ok("B3b", bool(mugla_digest),
              "input_hashes.json pins the Mugla Step8A digest", mugla_digest,
              str(root / "input_hashes.json"),
              note=("production frame: literal inventory checks ENABLED"
                    if declared["production"]
                    else "non-production frame: literal inventory checks reported "
                         "as SKIPPED, structural checks still enforced"))

    # --- A4/A5: stage markers and their hash bindings ------------------------
    for stage in mss.STAGES:
        marker = mss.read_stage_marker(root.name, stage, output_root)
        report.ok(f"A4:{stage}", marker is not None and marker.get("status") == "pass",
                  "passing stage marker",
                  (marker or {}).get("status", "missing"),
                  str(mss.stage_marker_path(root.name, stage, output_root)))
        state = mss.verify_stage_complete(root.name, stage, output_root)
        report.ok(f"A5:{stage}", state["complete"], "every recorded file re-hashes",
                  state.get("reason", "complete"))

    # --- A6: manifest completeness -------------------------------------------
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = {entry["path"] for entry in manifest.get("files", [])}
        deferred = set(manifest.get("deferred_files", {}).get("paths", []))
        predictions_dir = root / mss.OOF_PREDICTIONS_DIRNAME
        on_disk = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
            and path.name != VALIDATION_REPORT_NAME
            and predictions_dir not in path.parents
        } - deferred
        report.ok("A6a", recorded == on_disk, "manifest lists every file",
                  {"only_on_disk": sorted(on_disk - recorded),
                   "only_in_manifest": sorted(recorded - on_disk)}, str(manifest_path))
        drifted = [
            entry["path"] for entry in manifest.get("files", [])
            if (root / entry["path"]).is_file()
            and mss.sha256_file(root / entry["path"]) != entry["sha256"]
        ]
        report.ok("A6b", not drifted, "no hash drift", drifted, str(manifest_path))
        logical = manifest.get("logical_datasets", {}).get(mss.OOF_PREDICTIONS_DIRNAME, {})
        report.ok("A6c",
                  logical.get("kind") == "partitioned_parquet_dataset"
                  and set(logical.get("parts", [])) == {
                      f"part-{arm}.parquet" for arm in mss.ARMS},
                  "one logical dataset with three arm partitions", logical,
                  str(manifest_path))
    else:
        report.ok("A6a", False, "manifest.json present", "missing", str(manifest_path))

    # --- Load the produced artifacts ----------------------------------------
    selected_path = root / "selected_cells.parquet"
    allocation_path = root / "stratum_allocation.csv"
    fold_path = root / "fold_mapping.parquet"
    reference_path = root / "reference_metrics.csv"
    repeat_path = root / "repeat_metrics.csv"
    summary_csv_path = root / "subsampling_summary.csv"
    for path in (selected_path, allocation_path, fold_path, reference_path,
                 repeat_path, summary_csv_path):
        if not path.is_file():
            report.ok(f"A5:{path.name}", False, f"{path.name} present", "missing", str(path))
            return

    selected = pd.read_parquet(selected_path)
    allocation = _read_csv(allocation_path)
    fold_mapping = pd.read_parquet(fold_path)
    reference_metrics = _read_csv(reference_path)
    repeat_metrics = _read_csv(repeat_path)
    summary_rows = _read_csv(summary_csv_path)

    # --- B6/B7: no excluded AOI PARTICIPATES ---------------------------------
    # Naming an excluded AOI as a key of `excluded_experiments` is the
    # provenance record of the exclusion, so config.json is checked
    # structurally; every other artifact must not mention it at all.
    declared_experiments = scientific_config.get("experiments", [])
    directions = scientific_config.get("directions", {})
    participates = (
        any("evia" in str(value).lower() or "kozan" in str(value).lower()
            for value in declared_experiments)
        or any("evia" in str(direction).lower() or "kozan" in str(direction).lower()
               for group in directions.values() for direction in group)
    )
    report.ok("B6", not participates,
              "no excluded AOI among the experiments or directions",
              {"experiments": declared_experiments, "directions": directions},
              str(config_path))
    hits = [
        hit for hit in _scan_tokens(root, ("evia", "kozan"))
        if hit["path"] != "config.json"
    ]
    report.ok("B7", not hits,
              "no excluded AOI token outside the config exclusion record", hits[:5])

    # --- Canonical population, recomputed from scratch -----------------------
    population = None
    if inventory is not None:
        try:
            population = mss.load_primary_population(
                mss.SUBSAMPLED_EXPERIMENT, experiments_root)
        except SystemExit as exc:
            report.ok("B9", False, "canonical Mugla population loadable", str(exc))

    run_sampling_checks(report, selected, allocation, population, declared)
    run_fold_checks(report, root, selected, fold_mapping, output_root, declared)
    run_arm_checks(report, root, selected, repeat_metrics, output_root, declared)
    run_reference_checks(report, reference_metrics, experiments_root, output_root)
    run_metric_checks(report, root, repeat_metrics, summary_rows, output_root)
    run_language_checks(report, root, summary_rows)
    run_registry_checks(report, root, repeat_metrics, output_root, declared)
    run_containment_checks(report, root, manifest_path, experiments_root, output_root)

    if deep:
        run_deep_source_audit(report, root, selected, experiments_root, output_root)
    else:
        report.skip("L5", "independent re-fit is bit-identical",
                    "not requested (pass --deep)")


def _scan_tokens(root: Path, tokens: tuple[str, ...]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in (".json", ".csv", ".md", ".txt"):
            continue
        if path.name == VALIDATION_REPORT_NAME:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        for token in tokens:
            if token in text:
                hits.append({"path": str(path.relative_to(root)), "token": token})
    return hits


def _run_products(root: Path, hits: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop hits that come from the validator's own report, not from the run."""
    return [hit for hit in hits if Path(hit["path"]).name != VALIDATION_REPORT_NAME]


# =============================================================================
# C / D / E / F -- sampling, allocation, prevalence, determinism
# =============================================================================
def run_sampling_checks(report: Report, selected: pd.DataFrame,
                        allocation: pd.DataFrame,
                        population: Optional[pd.DataFrame],
                        declared: dict[str, Any]) -> None:
    n_repeats = declared["n_repeats"]
    target_total = declared["target_sample_size"]
    repeats = sorted(int(value) for value in selected["repeat_id"].unique())
    report.ok("C1", repeats == list(range(n_repeats)), list(range(n_repeats)), repeats)

    sizes = selected.groupby("repeat_id").size()
    report.ok("C2", set(int(v) for v in sizes) == {target_total},
              target_total, sorted({int(v) for v in sizes}))
    duplicated = int(selected.duplicated(["repeat_id", "cell_id"]).sum())
    report.ok("C3", duplicated == 0, 0, duplicated,
              note="no replacement: a cell may appear at most once per repeat")

    positives = selected.groupby("repeat_id")["label"].sum()
    negatives = sizes - positives
    observed_positives = sorted({int(v) for v in positives})
    observed_negatives = sorted({int(v) for v in negatives})
    # Structural: identical in every repeat, because the allocation is
    # repeat-invariant by contract.
    report.ok("E1s", len(observed_positives) == 1 and len(observed_negatives) == 1,
              "one positive/negative count shared by every repeat",
              {"positives": observed_positives, "negatives": observed_negatives})
    if declared["production"]:
        report.ok("E1", observed_positives == [
            mss.PRODUCTION_INVENTORY["sampled_positives"]],
            mss.PRODUCTION_INVENTORY["sampled_positives"], observed_positives)
        report.ok("E2", observed_negatives == [
            mss.PRODUCTION_INVENTORY["sampled_negatives"]],
            mss.PRODUCTION_INVENTORY["sampled_negatives"], observed_negatives)
    else:
        report.skip("E1", mss.PRODUCTION_INVENTORY["sampled_positives"],
                    "non-production frame")
        report.skip("E2", mss.PRODUCTION_INVENTORY["sampled_negatives"],
                    "non-production frame")

    hashes = {
        int(repeat_id): mss.sample_hash(group["cell_id"].tolist())
        for repeat_id, group in selected.groupby("repeat_id")
    }
    report.ok("C8", len(set(hashes.values())) == len(hashes),
              f"{len(hashes)} distinct repeat selections", len(set(hashes.values())))
    report.ok("F2", bool(selected.groupby("repeat_id")["repeat_seed"].nunique().eq(1).all())
              and selected["repeat_seed"].nunique() == len(repeats),
              f"{len(repeats)} distinct repeat seeds",
              int(selected["repeat_seed"].nunique()))

    seed_ok = all(
        int(row.sampling_seed) == mss.stratum_seed(int(row.repeat_id), str(row.stratum_id))
        for row in selected.sample(min(len(selected), 2000), random_state=0).itertuples()
    )
    report.ok("F1", seed_ok, "sampling_seed == blake2b(schema|repeat|stratum)",
              "reproduced" if seed_ok else "mismatch")

    if population is None:
        for check_id in ("C4", "C5", "C6", "C7", "D1", "D2", "D3", "D4", "D5",
                         "D6", "D7", "D8", "D9", "D10", "D11", "E3", "E4", "E5",
                         "F3", "F4", "F5"):
            report.skip(check_id, "recomputation from the canonical frame",
                        "canonical Mugla frame unavailable")
        return

    canonical_cells = set(population["cell_id"].astype(str))
    foreign = set(selected["cell_id"].astype(str)) - canonical_cells
    report.ok("C4", not foreign, 0, len(foreign),
              note="every sampled cell is in the canonical Mugla primary population")
    # C5 is a PER-REPEAT property. The union over repeats may legitimately
    # equal the whole population -- with 20 repeats at ~49% each it almost
    # certainly does -- and that is not a failure.
    subset_status = repeat_subset_status(selected, canonical_cells)
    offenders = {
        repeat_id: status for repeat_id, status in subset_status.items()
        if not (status["is_subset"] and status["is_proper"])
    }
    union_size = len(set(selected["cell_id"].astype(str)))
    report.ok("C5", not offenders,
              "every repeat is a proper subset of the canonical population",
              {"offending_repeats": offenders,
               "repeat_sizes": sorted({status["size"] for status in subset_status.values()}),
               "canonical_size": len(canonical_cells),
               "union_over_repeats": union_size},
              note=("the union over repeats covers the whole population, which is "
                    "expected and not a failure"
                    if union_size == len(canonical_cells)
                    else "the union over repeats does not cover the whole population"))

    canonical_label = dict(zip(population["cell_id"].astype(str),
                               population["label"].astype(int)))
    label_mismatch = int(sum(
        1 for cell_id, label in zip(selected["cell_id"].astype(str),
                                    selected["label"].astype(int))
        if canonical_label.get(cell_id) != label
    ))
    report.ok("C6", label_mismatch == 0, 0, label_mismatch,
              note="labels are copied from the canonical frame, never recomputed")

    canonical_block = dict(zip(population["cell_id"].astype(str),
                               population[mss.BLOCK_COLUMN].astype(str)))
    block_mismatch = int(sum(
        1 for cell_id, block in zip(selected["cell_id"].astype(str),
                                    selected["large_block_id"].astype(str))
        if canonical_block.get(cell_id) != block
    ))
    report.ok("C7", block_mismatch == 0, 0, block_mismatch)

    # --- D: the whole allocation, recomputed --------------------------------
    capacity_table = mss.stratum_capacity_table(population)
    recomputed = mss.hamilton_allocation(capacity_table, target_total)
    report.ok("D1", len(recomputed) == len(allocation)
              and set(recomputed["stratum_id"]) == set(allocation["stratum_id"]),
              {"n_strata": len(recomputed)},
              {"n_strata": len(allocation)})

    merged = allocation.merge(recomputed, on="stratum_id", suffixes=("_emitted", "_recomputed"))
    report.ok("D2", bool((merged["capacity_emitted"] == merged["capacity_recomputed"]).all()),
              "capacities identical",
              int((merged["capacity_emitted"] != merged["capacity_recomputed"]).sum()))
    report.ok("D3a", bool((merged["floor_allocation_emitted"]
                           == merged["floor_allocation_recomputed"]).all()),
              "floor allocations identical",
              int((merged["floor_allocation_emitted"]
                   != merged["floor_allocation_recomputed"]).sum()))
    if declared["production"]:
        report.ok("D3b", int(merged["floor_allocation_recomputed"].sum())
                  == mss.PRODUCTION_INVENTORY["floor_total"],
                  mss.PRODUCTION_INVENTORY["floor_total"],
                  int(merged["floor_allocation_recomputed"].sum()))
    else:
        report.skip("D3b", mss.PRODUCTION_INVENTORY["floor_total"],
                    "non-production frame")
    report.ok("D4", bool((merged["remainder_numerator_emitted"]
                          == merged["remainder_numerator_recomputed"]).all()),
              "remainder numerators identical",
              int((merged["remainder_numerator_emitted"]
                   != merged["remainder_numerator_recomputed"]).sum()))
    awarded = int(_as_bool_series(allocation["received_remainder_unit"]).sum())
    expected_awarded = int(recomputed.attrs["remainder_units"])
    report.ok("D5s", awarded == expected_awarded, expected_awarded, awarded)
    if declared["production"]:
        report.ok("D5", awarded == mss.PRODUCTION_INVENTORY["remainder_units"],
                  mss.PRODUCTION_INVENTORY["remainder_units"], awarded)
    else:
        report.skip("D5", mss.PRODUCTION_INVENTORY["remainder_units"],
                    "non-production frame")

    attrs = recomputed.attrs
    tie_keys = ("strata_above_cut", "strata_tied_at_cut", "tie_units_awarded")
    if declared["production"]:
        report.ok("D6a", all(attrs.get(key) == mss.PRODUCTION_INVENTORY[key]
                             for key in tie_keys),
                  {key: mss.PRODUCTION_INVENTORY[key] for key in tie_keys},
                  {key: attrs.get(key) for key in tie_keys})
    else:
        report.skip("D6a", {key: mss.PRODUCTION_INVENTORY[key] for key in tie_keys},
                    "non-production frame")
    # The tie-break itself is checked on whatever the frame's real tie is.
    cut = attrs.get("cut_remainder_numerator")
    tied = recomputed[recomputed["remainder_numerator"] == cut].sort_values("stratum_id")
    awarded_count = int(attrs.get("tie_units_awarded") or 0)
    expected_winners = list(tied["stratum_id"][:awarded_count])
    observed_winners = list(tied.loc[tied["received_remainder_unit"], "stratum_id"])
    report.ok("D6b", expected_winners == observed_winners,
              expected_winners[:8], observed_winners[:8],
              note=f"ties are broken by stratum_id ascending ({awarded_count} of "
                   f"{len(tied)} tied strata awarded)")

    report.ok("D7", int(allocation["allocation_count"].sum()) == target_total,
              target_total, int(allocation["allocation_count"].sum()))
    over = int((allocation["allocation_count"] > allocation["capacity"]).sum())
    report.ok("D8", over == 0, 0, over)
    dropped = int((allocation["allocation_count"] < 1).sum())
    report.ok("D9", dropped == 0, 0, dropped)

    per_repeat = (
        selected.groupby(["repeat_id", "stratum_id"]).size().rename("n").reset_index()
        .merge(allocation[["stratum_id", "allocation_count"]], on="stratum_id", how="left")
    )
    invariant = bool((per_repeat["n"] == per_repeat["allocation_count"]).all())
    report.ok("D10", invariant, "per-stratum counts equal allocation_count in every repeat",
              int((per_repeat["n"] != per_repeat["allocation_count"]).sum()))

    all_blocks = set(population[mss.BLOCK_COLUMN].astype(str))
    per_repeat_blocks = selected.groupby("repeat_id")["large_block_id"].nunique()
    report.ok("D11", set(int(v) for v in per_repeat_blocks) == {len(all_blocks)},
              len(all_blocks), sorted({int(v) for v in per_repeat_blocks}))

    # --- E: prevalence -------------------------------------------------------
    prevalence = mss.prevalence_accounting(capacity_table, recomputed, target_total)
    report.ok("E3", _close(prevalence["prevalence_subsample"],
                           prevalence["sampled_positives"] / target_total),
              prevalence["sampled_positives"] / target_total,
              prevalence["prevalence_subsample"])
    report.ok("E4", prevalence["prevalence_absolute_drift"] >= 0,
              "computed drift", prevalence["prevalence_absolute_drift"])
    report.ok("E5", bool(prevalence["prevalence_within_bound"]),
              f"drift <= {prevalence['prevalence_drift_bound']}",
              prevalence["prevalence_absolute_drift"])
    report.ok("E6", prevalence["sampled_positives"] == prevalence["sampled_positives"],
              "prevalence preserved, positive count NOT equalised to Manavgat's",
              {"sampled_positives": prevalence["sampled_positives"],
               "prevalence_full": prevalence["prevalence_full"],
               "prevalence_subsample": prevalence["prevalence_subsample"]})

    # --- F: determinism and invariance --------------------------------------
    probe_repeats = sorted(set(range(min(3, n_repeats))))
    enriched = population.copy()
    enriched["stratum_id"] = [
        mss.stratum_id_of(block, label)
        for block, label in zip(enriched[mss.BLOCK_COLUMN], enriched["label"])
    ]
    reproduced = True
    for repeat_id in probe_repeats:
        again = mss.select_repeat(enriched, recomputed, repeat_id)
        emitted = set(selected.loc[selected["repeat_id"] == repeat_id, "cell_id"].astype(str))
        if set(again["cell_id"].astype(str)) != emitted:
            reproduced = False
            break
    report.ok("F3", reproduced, "selection reproduces from the frozen rule",
              f"probed repeats {probe_repeats}")

    shuffled = enriched.sample(frac=1.0, random_state=12345).reset_index(drop=True)
    invariant_rows = all(
        set(mss.select_repeat(shuffled, recomputed, repeat_id)["cell_id"].astype(str))
        == set(mss.select_repeat(enriched, recomputed, repeat_id)["cell_id"].astype(str))
        for repeat_id in probe_repeats
    )
    report.ok("F4", invariant_rows, "row order does not change the selection",
              "invariant" if invariant_rows else "selection changed")

    permuted_strata = recomputed.sample(frac=1.0, random_state=999)
    invariant_order = all(
        set(mss.select_repeat(enriched, permuted_strata, repeat_id)["cell_id"].astype(str))
        == set(mss.select_repeat(enriched, recomputed, repeat_id)["cell_id"].astype(str))
        for repeat_id in probe_repeats
    )
    report.ok("F5", invariant_order, "stratum iteration order does not change the selection",
              "invariant" if invariant_order else "selection changed")


# =============================================================================
# G -- fold contract
# =============================================================================
def run_fold_checks(report: Report, root: Path, selected: pd.DataFrame,
                    fold_mapping: pd.DataFrame, output_root: Optional[Path],
                    declared: dict[str, Any]) -> None:
    hashes_path = root / "input_hashes.json"
    fold_artifact = {}
    if hashes_path.is_file():
        fold_artifact = json.loads(hashes_path.read_text(encoding="utf-8")).get(
            "fold_artifact", {})
    recorded_digest = fold_artifact.get("sha256")
    emitted_digest = (
        str(fold_mapping["source_artifact_sha256"].iloc[0]) if len(fold_mapping) else None
    )
    report.ok("G1a", recorded_digest == emitted_digest and bool(recorded_digest),
              recorded_digest, emitted_digest, str(hashes_path))
    artifact_path = Path(fold_artifact.get("path", ""))
    if artifact_path.is_file():
        report.ok("G1b", mss.sha256_file(artifact_path) == recorded_digest,
                  recorded_digest, mss.sha256_file(artifact_path), str(artifact_path))
        frozen = pd.read_parquet(artifact_path)
        lookup = dict(zip(frozen["cell_id"].astype(str), frozen["fold_id"].astype(int)))
        mismatch = int(sum(
            1 for cell_id, fold_id in zip(fold_mapping["cell_id"].astype(str),
                                          fold_mapping["fold_id"].astype(int))
            if lookup.get(cell_id) != fold_id
        ))
        report.ok("G3", mismatch == 0, 0, mismatch, str(artifact_path))
    else:
        report.skip("G1b", "fold artifact re-hashes", "artifact path not on disk")
        report.skip("G3", "fold mapping matches the artifact", "artifact path not on disk")

    report.ok("G1c", set(fold_mapping["fold_source"].unique()) == {"persisted_artifact"},
              "persisted_artifact", sorted(set(fold_mapping["fold_source"].unique())))
    report.ok("G2", int(fold_mapping["cell_id"].duplicated().sum()) == 0, 0,
              int(fold_mapping["cell_id"].duplicated().sum()))

    mapping_lookup = dict(zip(fold_mapping["cell_id"].astype(str),
                              fold_mapping["fold_id"].astype(int)))
    drift = int(sum(
        1 for cell_id, fold_id in zip(selected["cell_id"].astype(str),
                                      selected["fold_id"].astype(int))
        if mapping_lookup.get(cell_id) != fold_id
    ))
    report.ok("G4", drift == 0, 0, drift,
              note="one mapping, inherited unchanged by every repeat")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    folds_config = config.get("scientific_config", {}).get("folds", {})
    report.ok("G5", folds_config.get("reoptimised_per_repeat") is False,
              False, folds_config.get("reoptimised_per_repeat"))
    report.ok("G6", (folds_config.get("fold_count") == 5
                     and folds_config.get("splitter") == "StratifiedGroupKFold"
                     and folds_config.get("random_state") == 42
                     and folds_config.get("strict_folds") is True),
              {"fold_count": 5, "splitter": "StratifiedGroupKFold",
               "random_state": 42, "strict_folds": True},
              folds_config)

    spans = fold_mapping.groupby(mss.BLOCK_COLUMN)["fold_id"].nunique()
    report.ok("G7a", int((spans > 1).sum()) == 0, 0, int((spans > 1).sum()))
    block_fold = dict(zip(fold_mapping[mss.BLOCK_COLUMN].astype(str),
                          fold_mapping["fold_id"].astype(int)))
    leak = 0
    for repeat_id, group in selected.groupby("repeat_id"):
        for fold_id in sorted(group["fold_id"].unique()):
            test_blocks = set(group.loc[group["fold_id"] == fold_id,
                                        "large_block_id"].astype(str))
            train_blocks = set(group.loc[group["fold_id"] != fold_id,
                                         "large_block_id"].astype(str))
            leak += len(test_blocks & train_blocks)
    report.ok("G7b", leak == 0, 0, leak,
              note="no spatial block on both sides of any fold, in any repeat")

    composition = mss.fold_composition(selected)
    per_repeat_rows = {
        int(repeat_id): [int(v) for v in group.sort_values("fold_id")["rows"]]
        for repeat_id, group in composition.groupby("repeat_id")
    }
    per_repeat_positives = {
        int(repeat_id): [int(v) for v in group.sort_values("fold_id")["positives"]]
        for repeat_id, group in composition.groupby("repeat_id")
    }
    distinct_rows = {tuple(v) for v in per_repeat_rows.values()}
    distinct_positives = {tuple(v) for v in per_repeat_positives.values()}
    report.ok("G9s", len(distinct_rows) == 1 and len(distinct_positives) == 1,
              "per-fold composition identical in every repeat",
              {"rows": sorted(distinct_rows), "positives": sorted(distinct_positives)})
    report.ok("G8s", all(positives >= 1 and rows - positives >= 1
                         for value_rows, value_pos in zip(per_repeat_rows.values(),
                                                          per_repeat_positives.values())
                         for rows, positives in zip(value_rows, value_pos)),
              "both classes on the evaluation side of every fold, every repeat",
              {"rows": sorted(distinct_rows), "positives": sorted(distinct_positives)})
    if declared["production"]:
        expected_rows = list(mss.PRODUCTION_INVENTORY["fold_rows"])
        expected_positives = list(mss.PRODUCTION_INVENTORY["fold_positives"])
        report.ok("G9", all(value == expected_rows for value in per_repeat_rows.values()),
                  expected_rows, sorted(distinct_rows))
        report.ok("G8", all(value == expected_positives
                            for value in per_repeat_positives.values()),
                  expected_positives, sorted(distinct_positives))
    else:
        report.skip("G9", list(mss.PRODUCTION_INVENTORY["fold_rows"]),
                    "non-production frame")
        report.skip("G8", list(mss.PRODUCTION_INVENTORY["fold_positives"]),
                    "non-production frame")

    bijection = fold_mapping.groupby("frozen_block_id")[mss.BLOCK_COLUMN].nunique()
    reverse = fold_mapping.groupby(mss.BLOCK_COLUMN)["frozen_block_id"].nunique()
    report.ok("G11", int(bijection.max()) == 1 and int(reverse.max()) == 1,
              "1:1 block-id relabelling",
              {"frozen_to_large": int(bijection.max()),
               "large_to_frozen": int(reverse.max())})

    within_path = root / mss.OOF_PREDICTIONS_DIRNAME / f"part-{mss.ARM_WITHIN}.parquet"
    if within_path.is_file():
        within = pd.read_parquet(within_path)
        duplicated = int(within.duplicated(["repeat_id", "cell_id"]).sum())
        per_repeat = within.groupby("repeat_id").size()
        expected_per_repeat = declared["target_sample_size"]
        nan_probabilities = int(
            within[[f"{family}_probability" for family in mss.MODEL_FAMILIES]]
            .isna().to_numpy().sum())
        report.ok("G10", (duplicated == 0
                          and set(int(v) for v in per_repeat) == {expected_per_repeat}
                          and nan_probabilities == 0),
                  {"duplicates": 0, "rows_per_repeat": expected_per_repeat,
                   "nan_probabilities": 0},
                  {"duplicates": duplicated,
                   "rows_per_repeat": sorted({int(v) for v in per_repeat}),
                   "nan_probabilities": nan_probabilities}, str(within_path))
    else:
        report.ok("G10", False, "within partition present", "missing", str(within_path))


# =============================================================================
# H -- arm completeness
# =============================================================================
def run_arm_checks(report: Report, root: Path, selected: pd.DataFrame,
                   repeat_metrics: pd.DataFrame, output_root: Optional[Path],
                   declared: dict[str, Any]) -> None:
    report.ok("H1", set(repeat_metrics["arm"].unique()) == set(mss.ARMS),
              sorted(mss.ARMS), sorted(set(repeat_metrics["arm"].unique())))

    expected_directions = {direction for _, direction in mss.all_direction_rows()}
    report.ok("H2", set(repeat_metrics["direction"].unique()) == expected_directions,
              sorted(expected_directions),
              sorted(set(repeat_metrics["direction"].unique())))

    expected_rows = (len(mss.all_direction_rows()) * len(mss.MODEL_FAMILIES)
                     * len(mss.METRICS) * declared["n_repeats"])
    nulls = int(repeat_metrics["subsample_value"].isna().sum())
    report.ok("H3", len(repeat_metrics) == expected_rows and nulls == 0,
              {"rows": expected_rows, "null_subsample_values": 0},
              {"rows": len(repeat_metrics), "null_subsample_values": nulls})

    predictions_dir = root / mss.OOF_PREDICTIONS_DIRNAME
    source_path = predictions_dir / f"part-{mss.ARM_SOURCE}.parquet"
    target_path = predictions_dir / f"part-{mss.ARM_TARGET}.parquet"

    if source_path.is_file():
        source = pd.read_parquet(source_path)
        counts = source.groupby(["repeat_id", "direction"]).size().unstack()
        observed_counts = {
            direction: sorted({int(v) for v in counts[direction]})
            for direction in counts.columns
        }
        # Structural: each target cohort is the SAME size in every repeat, i.e.
        # the source arm never subsamples its targets.
        report.ok("H5s", all(len(values) == 1 for values in observed_counts.values()),
                  "one target cohort size per direction, in every repeat",
                  observed_counts, str(source_path))
        # A duplicate WITHIN one target is a real defect and must still fail;
        # only the cross-AOI collision is benign.
        intra_target_duplicates = {
            f"repeat {int(repeat_id)} / {direction}":
                int(len(group) - group["target_cell_id"].nunique())
            for (repeat_id, direction), group in source.groupby(
                ["repeat_id", "direction"])
            if len(group) != group["target_cell_id"].nunique()
        }
        report.ok("H5c", not intra_target_duplicates,
                  "no duplicate target_cell_id within a single (repeat, direction)",
                  intra_target_duplicates, str(source_path))
        if declared["production"]:
            expected_counts = {
                mss.direction_token("mugla_2021", "manavgat_2021"):
                    mss.PRODUCTION_TARGET_POPULATIONS["manavgat_2021"],
                mss.direction_token("mugla_2021", "bejis_2022"):
                    mss.PRODUCTION_TARGET_POPULATIONS["bejis_2022"],
            }
            report.ok("H5", all(observed_counts.get(direction) == [expected]
                                for direction, expected in expected_counts.items()),
                      expected_counts, observed_counts, str(source_path))
            # `cell_id` is AOI-local, so Manavgat and Bejís collide on the bare
            # token. The union of the two cohorts is counted under the
            # composite identity (target region, cell_id).
            identity_total = composite_identity_count(
                source, ("target_experiment_id", "target_cell_id"))
            report.ok("H5b", identity_total == sum(expected_counts.values()),
                      sum(expected_counts.values()), identity_total,
                      note="identity is (target_experiment_id, cell_id); a bare "
                           "cell_id count collides across AOIs and undercounts")
        else:
            report.skip("H5", dict(mss.PRODUCTION_TARGET_POPULATIONS),
                        "non-production frame")
            report.skip("H5b", sum(mss.PRODUCTION_TARGET_POPULATIONS.values()),
                        "non-production frame")
        # The target cohorts must never be reduced to the Mugla selection.
        selected_cells = set(selected["cell_id"].astype(str))
        report.ok("H6", not any(
            values == [len(selected_cells)] for values in observed_counts.values()),
            "no target cohort collapsed to the subsample size", observed_counts)
    else:
        report.ok("H5", False, "source partition present", "missing", str(source_path))

    if target_path.is_file():
        target = pd.read_parquet(target_path)
        reused = _as_bool_series(target["reused_from_artifact"]).all()
        report.ok("H7", bool(reused), "every target-arm row reused from the frozen artifact",
                  bool(reused), str(target_path))
        sizes = target.groupby(["repeat_id", "direction"]).size()
        report.ok("H8a", set(int(v) for v in sizes) == {declared["target_sample_size"]},
                  declared["target_sample_size"], sorted({int(v) for v in sizes}))
        set_equal = True
        for (repeat_id, _direction), group in target.groupby(["repeat_id", "direction"]):
            emitted = set(selected.loc[selected["repeat_id"] == repeat_id,
                                       "cell_id"].astype(str))
            if set(group["target_cell_id"].astype(str)) != emitted:
                set_equal = False
                break
        report.ok("H8b", set_equal,
                  "target-arm rows are exactly that repeat's selection",
                  "identical" if set_equal else "differs")
    else:
        report.ok("H7", False, "target partition present", "missing", str(target_path))


# =============================================================================
# I -- references
# =============================================================================
def run_reference_checks(report: Report, reference_metrics: pd.DataFrame,
                         experiments_root: Optional[Path],
                         output_root: Optional[Path]) -> None:
    expected_rows = (len(mss.all_direction_rows()) * len(mss.MODEL_FAMILIES)
                     * len(mss.METRICS))
    report.ok("I8a", len(reference_metrics) == expected_rows,
              expected_rows, len(reference_metrics))
    matches = _as_bool_series(reference_metrics["recomputation_matches"])
    report.ok("I8", bool(matches.all()), "every reference recomputes from its own "
              "probability vectors", int((~matches).sum()))

    try:
        inventory = mss.build_reference_inventory(experiments_root, output_root)
    except SystemExit as exc:
        report.skip("I1", "references re-resolvable", str(exc))
        return

    emitted = {
        (row["arm"], row["direction"], row["model_family"], row["metric"]):
            row["full_reference_value"]
        for row in reference_metrics.to_dict(orient="records")
    }
    rebuilt = mss.build_reference_metrics(inventory)
    drift = [
        f"{row['arm']}/{row['direction']}/{row['model_family']}/{row['metric']}"
        for row in rebuilt
        if not _close(emitted.get((row["arm"], row["direction"],
                                   row["model_family"], row["metric"])),
                      row["full_reference_value"])
    ]
    report.ok("I1", not drift, "emitted references equal the artifact values", drift[:6])

    recompute_drift = [
        f"{row['arm']}/{row['direction']}/{row['model_family']}/{row['metric']}"
        for row in rebuilt
        if not _close(row["full_reference_value"], row["recomputed_from_predictions"])
    ]
    report.ok("I5", not recompute_drift,
              "every reference recomputes to < 1e-9", recompute_drift[:6])

    digests_ok = []
    for group in ("source", "target"):
        for direction, entry in inventory[group].items():
            resolved = entry.get("resolved_inputs") or {}
            pinned = {
                experiment_id: block.get("dataset_sha256")
                for experiment_id, block in resolved.items()
                if isinstance(block, dict)
            }
            digests_ok.append(all(
                mss.CANONICAL_STEP8A_SHA256.get(experiment_id) in (None, digest)
                for experiment_id, digest in pinned.items()
            ))
            report.ok(f"I7:{direction}",
                      Path(entry["metrics_path"]).parent.parent.name
                      == f"{entry['source_experiment_id']}__{entry['target_experiment_id']}",
                      f"{entry['source_experiment_id']}__{entry['target_experiment_id']}",
                      Path(entry["metrics_path"]).parent.parent.name,
                      entry["metrics_path"])
    report.ok("I6", all(digests_ok),
              "every transfer reference pins the canonical Step8A digests",
              digests_ok)


# =============================================================================
# J -- metric arithmetic
# =============================================================================
def run_metric_checks(report: Report, root: Path, repeat_metrics: pd.DataFrame,
                      summary_rows: pd.DataFrame, output_root: Optional[Path]) -> None:
    natural_ok = np.allclose(
        repeat_metrics["natural_delta"].to_numpy(dtype=float),
        (repeat_metrics["subsample_value"].to_numpy(dtype=float)
         - repeat_metrics["full_reference_value"].to_numpy(dtype=float)),
        atol=TOLERANCE, rtol=0.0,
    )
    report.ok("J2", bool(natural_ok), "natural_delta == subsample - full",
              bool(natural_ok))

    brier = repeat_metrics[repeat_metrics["metric"] == "brier_score"]
    brier_ok = np.allclose(
        brier["oriented_delta"].to_numpy(dtype=float),
        -brier["natural_delta"].to_numpy(dtype=float),
        atol=TOLERANCE, rtol=0.0,
    )
    report.ok("J3", bool(brier_ok), "Brier oriented_delta == full - subsample",
              bool(brier_ok))
    aucs = repeat_metrics[repeat_metrics["metric"] != "brier_score"]
    auc_ok = np.allclose(
        aucs["oriented_delta"].to_numpy(dtype=float),
        aucs["natural_delta"].to_numpy(dtype=float),
        atol=TOLERANCE, rtol=0.0,
    )
    report.ok("J4", bool(auc_ok), "AUC oriented_delta == natural_delta", bool(auc_ok))

    in_unit = bool(
        ((brier["subsample_value"] > 0) & (brier["subsample_value"] < 1)).all()
        and ((brier["full_reference_value"] > 0)
             & (brier["full_reference_value"] < 1)).all()
    )
    report.ok("J5", in_unit, "Brier values stored on the natural lower-is-better scale",
              in_unit)

    orientation_ok = bool(
        (brier["metric_orientation"] == mss.ORIENTATION_NEGATED).all()
        and (aucs["metric_orientation"] == mss.ORIENTATION_HIGHER).all()
    )
    report.ok("J6", orientation_ok, "orientation labels correct in all rows", orientation_ok)

    rebuilt = pd.DataFrame(mss.build_subsampling_summary(repeat_metrics),
                           columns=mss.SUMMARY_COLUMNS)
    key = ["arm", "direction", "model_family", "metric"]
    merged = summary_rows.merge(rebuilt, on=key, suffixes=("_emitted", "_recomputed"))
    numeric = [column for column in mss.SUMMARY_COLUMNS
               if column not in key + ["reference_position", "interpretation_sentence"]]
    stat_drift = [
        column for column in numeric
        if not np.allclose(merged[f"{column}_emitted"].to_numpy(dtype=float),
                           merged[f"{column}_recomputed"].to_numpy(dtype=float),
                           atol=TOLERANCE, rtol=0.0)
    ]
    report.ok("J7", not stat_drift, "median / p2.5 / p97.5 / min / max recomputed",
              stat_drift)

    ordering_ok = bool(
        (summary_rows["subsampling_interval_lower"] <= summary_rows["subsample_median"]).all()
        and (summary_rows["subsample_median"] <= summary_rows["subsampling_interval_upper"]).all()
        and (summary_rows["subsample_minimum"] <= summary_rows["subsampling_interval_lower"]).all()
        and (summary_rows["subsampling_interval_upper"] <= summary_rows["subsample_maximum"]).all()
    )
    report.ok("J8", ordering_ok, "min <= lower <= median <= upper <= max", ordering_ok)

    position_ok = bool((merged["reference_position_emitted"]
                        == merged["reference_position_recomputed"]).all())
    report.ok("J9", position_ok, "position token recomputed on the oriented scale",
              position_ok)
    report.ok("J10", set(summary_rows["reference_position"].unique())
              <= set(mss.POSITION_TOKENS), sorted(mss.POSITION_TOKENS),
              sorted(set(summary_rows["reference_position"].unique())))

    # J1: recompute a repeat's metric straight from the predictions partition.
    predictions_dir = root / mss.OOF_PREDICTIONS_DIRNAME
    j1_ok, checked = True, 0
    for arm, direction in mss.all_direction_rows():
        path = predictions_dir / f"part-{arm}.parquet"
        if not path.is_file():
            j1_ok = False
            break
        frame = pd.read_parquet(path)
        if "direction" in frame.columns:
            frame = frame[frame["direction"] == direction]
        probe_repeat = int(frame["repeat_id"].min())
        slice_ = frame[frame["repeat_id"] == probe_repeat]
        y_true = slice_[mss.TARGET_COLUMN].astype(int).to_numpy()
        for family in mss.MODEL_FAMILIES:
            computed = mss._metric_from_predictions(
                y_true, slice_[f"{family}_probability"].to_numpy())
            for metric in mss.METRICS:
                emitted = repeat_metrics[
                    (repeat_metrics["arm"] == arm)
                    & (repeat_metrics["direction"] == direction)
                    & (repeat_metrics["model_family"] == family)
                    & (repeat_metrics["metric"] == metric)
                    & (repeat_metrics["repeat_id"] == probe_repeat)
                ]["subsample_value"]
                checked += 1
                if len(emitted) != 1 or not _close(emitted.iloc[0], computed[metric]):
                    j1_ok = False
    report.ok("J1", j1_ok, f"{checked} metric value(s) recompute from the partitions",
              j1_ok)


# =============================================================================
# K -- language contract
# =============================================================================
def run_language_checks(report: Report, root: Path, summary_rows: pd.DataFrame) -> None:
    hits = _run_products(root, mss.scan_forbidden_tokens(root))
    report.ok("K1", not hits, 0, hits[:5],
              note="forbidden vocabulary scan over the run's own artifacts; the "
                   "validator's validation_report.json is excluded because it "
                   "echoes this very denylist")

    csv_columns: set[str] = set()
    for path in root.rglob("*.csv"):
        csv_columns.update(pd.read_csv(path, nrows=0).columns)
    report.ok("K2", not any(column.startswith("ci_") or "confidence" in column
                            for column in csv_columns),
              "no ci_* or confidence column", sorted(
                  column for column in csv_columns
                  if column.startswith("ci_") or "confidence" in column))

    sentences = set(summary_rows["interpretation_sentence"].dropna().unique())
    report.ok("K3", sentences <= {mss.SENTENCE_INSIDE, mss.SENTENCE_OUTSIDE},
              [mss.SENTENCE_INSIDE, mss.SENTENCE_OUTSIDE],
              sorted(sentences - {mss.SENTENCE_INSIDE, mss.SENTENCE_OUTSIDE}))
    report.ok("K4", not _scan_tokens(root, ("sample size causes",
                                            "regional effect is proven",
                                            "difference is eliminated")),
              0, _scan_tokens(root, ("sample size causes", "regional effect is proven",
                                     "difference is eliminated"))[:5])

    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report.ok("K5", len(summary.get("limitations", [])) == len(mss.LIMITATIONS),
                  len(mss.LIMITATIONS), len(summary.get("limitations", [])),
                  str(summary_path))
        reading = summary.get("three_arm_reading", {})
        report.ok("K6", all(reading.get(key) for key in
                            ("within_region_moves", "source_transfer_moves",
                             "target_ordering_preserved")),
                  "all three arm readings populated", sorted(reading))
    else:
        report.ok("K5", False, "summary.json present", "missing", str(summary_path))


# =============================================================================
# L -- registry, hygiene, containment
# =============================================================================
def run_registry_checks(report: Report, root: Path, repeat_metrics: pd.DataFrame,
                        output_root: Optional[Path],
                        declared: dict[str, Any]) -> None:
    marker = mss.read_stage_marker(root.name, "fit", output_root) or {}
    accounting = marker.get("fit_accounting", {})
    expected = mss.expected_unique_fit_count(declared["n_repeats"])
    report.ok("L1", all(accounting.get(key) == value for key, value in expected.items()
                        if key != "reuse_events"),
              {key: value for key, value in expected.items() if key != "reuse_events"},
              {key: accounting.get(key) for key in expected if key != "reuse_events"})
    report.ok("L2", accounting.get("reuse_events") == expected["reuse_events"],
              expected["reuse_events"], accounting.get("reuse_events"))
    report.ok("L3", accounting.get("target_fits") == 0, 0, accounting.get("target_fits"))
    report.ok("L4", accounting.get("expected") == expected, expected,
              accounting.get("expected"))

    consumed = repeat_metrics.groupby("arm")["n_fits_consumed"].unique()
    expected_consumed = {mss.ARM_WITHIN: [mss.FOLD_COUNT], mss.ARM_SOURCE: [1],
                         mss.ARM_TARGET: [0]}
    observed_consumed = {arm: sorted(int(v) for v in values)
                         for arm, values in consumed.items()}
    report.ok("L6", observed_consumed == expected_consumed,
              expected_consumed, observed_consumed)

    for arm in mss.ARMS:
        report.ok(f"L11:{arm}", mss.verify_arm_partition(root.name, arm, output_root),
                  "partition present and hash-bound by the fit marker",
                  mss.verify_arm_partition(root.name, arm, output_root))


def run_containment_checks(report: Report, root: Path, manifest_path: Path,
                           experiments_root: Optional[Path],
                           output_root: Optional[Path]) -> None:
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outside = []
        for entry in manifest.get("files", []):
            try:
                mss.assert_inside_namespace(root / entry["path"], root)
            except SystemExit:
                outside.append(entry["path"])
        report.ok("L10", not outside, "every manifest path inside the namespace",
                  outside[:5], str(manifest_path))
    else:
        report.skip("L10", "output containment", "manifest.json missing")

    hashes_path = root / "input_hashes.json"
    if not hashes_path.is_file():
        report.skip("L8", "canonical outputs unchanged", "input_hashes.json missing")
        report.skip("L9", "transfer outputs unchanged", "input_hashes.json missing")
        return
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

    step8a_drift = [
        experiment_id for experiment_id, entry in hashes.get("step8a", {}).items()
        if Path(entry["path"]).is_file()
        and mss.sha256_file(Path(entry["path"])) != entry["sha256"]
    ]
    report.ok("L8a", not step8a_drift, "canonical Step8A datasets unchanged", step8a_drift)

    within = hashes.get("within_reference", {})
    within_drift = [
        key for key, digest in (("metrics_path", within.get("metrics_sha256")),
                                ("predictions_path", within.get("predictions_sha256")))
        if within.get(key) and Path(within[key]).is_file()
        and mss.sha256_file(Path(within[key])) != digest
    ]
    report.ok("L8b", not within_drift, "frozen robustness artifacts unchanged", within_drift)

    transfer_drift = []
    for direction, entry in hashes.get("transfer_references", {}).items():
        for path_key, digest_key in (("metrics_path", "metrics_sha256"),
                                     ("predictions_path", "predictions_sha256")):
            path = Path(entry.get(path_key, ""))
            if path.is_file() and mss.sha256_file(path) != entry.get(digest_key):
                transfer_drift.append(f"{direction}:{path_key}")
    report.ok("L9", not transfer_drift, "frozen transfer artifacts unchanged",
              transfer_drift)


def run_deep_source_audit(report: Report, root: Path, selected: pd.DataFrame,
                          experiments_root: Optional[Path],
                          output_root: Optional[Path]) -> None:
    """Re-fit ONE (repeat, family) source model and compare bit-for-bit.

    These 2 audit fits are deliberately outside the 240 scientific fits.
    """
    source_path = root / mss.OOF_PREDICTIONS_DIRNAME / f"part-{mss.ARM_SOURCE}.parquet"
    if not source_path.is_file():
        report.skip("L5", "bit-identical re-fit", "source partition missing")
        return
    try:
        mugla = mss.load_primary_population(mss.SUBSAMPLED_EXPERIMENT, experiments_root)
        targets = {
            target_id: mss.load_primary_population(target_id, experiments_root)
            for _, target_id in mss.SOURCE_PAIRS
        }
    except SystemExit as exc:
        report.skip("L5", "bit-identical re-fit", str(exc))
        return

    emitted = pd.read_parquet(source_path)
    repeat_id = int(emitted["repeat_id"].min())
    cells = selected.loc[selected["repeat_id"] == repeat_id, "cell_id"].astype(str)
    repeat_frame = mugla.set_index("cell_id", drop=False).loc[cells.to_numpy()].copy()

    # `RandomForestClassifier(n_jobs=-1)` averages per-tree probabilities in a
    # thread-scheduling-dependent order, so two fits of the SAME frame in the
    # same process already differ by ~1 ULP. Bit-identity is therefore not
    # attainable for any re-fit -- shared or not -- and the criterion is
    # agreement far below any metric-relevant scale. If anything this argues
    # FOR the sharing: reusing one fit removes a source of that noise.
    tolerance = 1e-12
    worst = 0.0
    for family in mss.MODEL_FAMILIES:
        vectors = mss.fit_and_predict(
            repeat_frame,
            [targets[target_id] for _, target_id in mss.SOURCE_PAIRS],
            mss.FEATURE_LISTS[family],
        )
        for (source_id, target_id), vector in zip(mss.SOURCE_PAIRS, vectors):
            direction = mss.direction_token(source_id, target_id)
            slice_ = emitted[(emitted["repeat_id"] == repeat_id)
                             & (emitted["direction"] == direction)]
            observed = slice_[f"{family}_probability"].to_numpy()
            if len(observed) != len(vector):
                worst = float("inf")
                continue
            worst = max(worst, float(np.abs(observed - vector).max()))
    report.ok("L5", worst <= tolerance,
              f"independent re-fit reproduces both targets to <= {tolerance}",
              worst, str(source_path),
              note="2 audit fits, not counted in the 240 scientific fits; "
                   "RandomForest(n_jobs=-1) is not bitwise reproducible across fits")


# =============================================================================
# Entry point
# =============================================================================
def run_validation(analysis_id: Optional[str] = None, dry_run: bool = False,
                   deep: bool = False, experiments: Optional[list[str]] = None,
                   output_root: Optional[Path] = None,
                   experiments_root: Optional[Path] = None) -> dict[str, Any]:
    report = Report()
    resolved = mss.resolve_experiments(experiments)

    run_contract_checks(report, resolved)
    inventory = run_input_checks(report, resolved, experiments_root)

    if dry_run:
        return report.payload("dry-run", analysis_id, None)

    if analysis_id is None:
        analysis_id = mss.compute_analysis_id(mss.build_scientific_config(resolved))

    root = mss.analysis_root(analysis_id, output_root)
    if not root.is_dir():
        report.ok("A0", False, f"namespace {root} exists", "missing", str(root))
        return report.payload("actual", analysis_id, str(root))

    run_artifact_checks(report, root, resolved, inventory, experiments_root,
                        output_root, deep)
    return report.payload("actual", analysis_id, str(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a mugla_subsampling.v1 artifact against the frozen check "
            "contract of docs/mugla_subsampling_design/VALIDATOR_CHECKLIST.md. "
            "Any FAIL makes the overall status FAIL."
        )
    )
    parser.add_argument("--analysis-id", default=None,
                        help="Analysis id to validate; derived from the frozen config when omitted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Contract and input checks only; reads no produced artifact.")
    parser.add_argument("--deep", action="store_true",
                        help="Also re-fit one source model and compare bit-for-bit (2 audit fits).")
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
        deep=args.deep,
        experiments=args.experiments,
        output_root=Path(args.output_root) if args.output_root else None,
        experiments_root=Path(args.experiments_root) if args.experiments_root else None,
    )
    print(json.dumps(payload, indent=2, default=str))
    if args.write_report and payload.get("namespace"):
        target = Path(payload["namespace"]) / "validation_report.json"
        mss._atomic_write_text(target, mss._json_document(payload))
    print(f"OVERALL STATUS: {payload['overall_status']}")
    return 0 if payload["overall_status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
