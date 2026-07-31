#!/usr/bin/env python3
"""Validator for the Marginal AoA completion artifact.

Implements the 27 checks of
`docs/marginal_aoa_completion_design/08_validation_and_test_contract.md`.

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
import math
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.marginal_aoa_completion as mac

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIPPED"


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self, check_id: str, status: str, expected: Any, observed: Any,
        evidence_path: Optional[str] = None, note: Optional[str] = None,
    ) -> None:
        self.checks.append({
            "check_id": check_id, "status": status,
            "expected": expected, "observed": observed,
            "evidence_path": evidence_path, "note": note,
        })

    def ok(self, check_id: str, condition: bool, expected: Any, observed: Any,
           evidence_path: Optional[str] = None, note: Optional[str] = None) -> bool:
        self.add(check_id, PASS if condition else FAIL, expected, observed, evidence_path, note)
        return bool(condition)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == FAIL]

    @property
    def skipped(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == SKIP]


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


# =============================================================================
# Dry-run checks: contract only
# =============================================================================
def validate_dry_run(
    output_root: Optional[Path] = None, experiments_root: Optional[Path] = None,
    strict_hashes: bool = True,
) -> Report:
    report = Report()

    mac.validate_feature_contract()
    report.ok(
        "contract.feature_set", len(mac.numeric_features()) == 9,
        "9 numeric predictors + 1 categorical",
        f"{len(mac.numeric_features())} numeric + {len(mac.categorical_features())} categorical",
    )
    report.ok(
        "contract.stage_order", list(mac.STAGES) == [
            "plan", "climate-export", "weighted-predictor-space",
            "climate-distance", "geographic-distance", "compare",
        ],
        "the six preregistered stages in order", list(mac.STAGES),
    )

    bad_range = False
    try:
        mac.validate_stage_range("compare", "plan")
    except SystemExit:
        bad_range = True
    report.ok("contract.stage_range_fail_closed", bad_range,
              "a reversed stage range raises", bad_range)

    unknown = False
    try:
        mac.validate_stage_range("not-a-stage", "compare")
    except SystemExit:
        unknown = True
    report.ok("contract.unknown_stage_fail_closed", unknown,
              "an unknown stage raises", unknown)

    report.ok(
        "contract.climate_four_variables",
        mac.CLIMATE_FEATURE_COUNT == 4 and len(mac.CLIMATE_FEATURES) == 4,
        "exactly 4 climate variables", list(mac.CLIMATE_FEATURES),
    )
    report.ok(
        "contract.no_removed_climate_axes",
        not ({"warm_season_mean_temperature", "warm_season_precipitation"}
             & set(mac.CLIMATE_FEATURES)),
        "warm-season temperature/precipitation are absent", list(mac.CLIMATE_FEATURES),
    )
    report.ok(
        "contract.threshold_primary_is_whisker",
        mac.PRIMARY_THRESHOLD_METHOD.endswith("upper_whisker_v1"),
        "upper whisker is the primary threshold", mac.PRIMARY_THRESHOLD_METHOD,
    )
    report.ok(
        "contract.normaliser_is_pairwise",
        mac.NORMALISER_METHOD == "source_pairwise_mean_distance_v1",
        "mean pairwise source distance", mac.NORMALISER_METHOD,
    )
    report.ok(
        "contract.geodesic_route",
        mac.GEODESIC_IMPLEMENTATION == "geographiclib_wgs84",
        "geographiclib", mac.GEODESIC_IMPLEMENTATION,
    )
    report.ok(
        "contract.primary_transfer_comparison",
        mac.PRIMARY_TRANSFER_COMPARISON == "raw_thermal_roc_auc"
        and mac.PRIMARY_TRANSFER_SELECTION["transfer_state"] == "raw",
        "raw thermal ROC-AUC", mac.PRIMARY_TRANSFER_COMPARISON,
    )

    plan = mac.run_analysis(
        dry_run=True, output_root=output_root, experiments_root=experiments_root,
        strict_hashes=strict_hashes,
    )
    report.ok("dryrun.no_files_written", plan["files_written"] == [],
              "no file written", plan["files_written"])
    report.ok("dryrun.no_gee", not plan["gee_queries_run"] and not plan["gee_exports_run"],
              "no Earth Engine activity",
              {"queries": plan["gee_queries_run"], "exports": plan["gee_exports_run"]})
    report.ok("dryrun.no_model_fit", not plan["model_fit"] and not plan["bootstrap_run"],
              "no model fit, no bootstrap",
              {"model_fit": plan["model_fit"], "bootstrap": plan["bootstrap_run"]})
    report.add("dryrun.prerequisites", PASS, "reported", plan["prerequisites"])
    report.add("dryrun.plan", PASS, "exact plan", {
        "analysis_id": plan["analysis_id"],
        "stages_requested": plan["stages_requested"],
        "output_namespace": plan["output_namespace"],
        "directed_pair_count": plan["directed_pair_count"],
    })
    return report


# =============================================================================
# Actual checks: the 27-check contract
# =============================================================================
def validate_actual(
    analysis_id: str, output_root: Optional[Path] = None,
    experiments_root: Optional[Path] = None, strict_hashes: bool = True,
    geodesic_inverse: Any = None,
) -> Report:
    """Validate a produced artifact.

    `geodesic_inverse` is the same injection point the analysis uses: it lets
    the kilometre-recomputation check run against a synthetic fixture whose
    distances came from an injected inverse. In production it is None, so the
    check recomputes with the real GeographicLib and fails closed when the
    package is absent -- a shipped kilometre value that cannot be independently
    recomputed is a failure, never a skip.
    """
    report = Report()
    root = mac.analysis_root(analysis_id, output_root)
    if not root.is_dir():
        report.ok("00.namespace_exists", False, str(root), "missing")
        return report

    metadata = _load_json(root / "completion_metadata.json")
    summary = _load_csv(root / "weighted_predictor_space" / "directed_pair_summary.csv")
    weights = _load_csv(root / "weighted_predictor_space" / "source_feature_weights.csv")
    thresholds = _load_csv(root / "weighted_predictor_space" / "source_threshold_diagnostics.csv")
    geo_pairs = _load_csv(root / "geographic_distance" / "pairwise_geographic_distance.csv")
    climate_pairs = _load_csv(root / "climate_distance" / "pairwise_climate_distance.csv")
    ranking = _load_csv(root / "comparison" / "ranking_summary.csv")

    # --- 1. canonical Step8A hashes -----------------------------------------
    inventory = mac.build_frozen_input_inventory(mac.CANONICAL_EXPERIMENTS, experiments_root)
    # strict: compare against the frozen production literals.
    # non-strict (injected synthetic root): compare against the hashes the
    # artifact ITSELF recorded. Both are real verifications -- neither is a
    # skip, so a valid artifact never validates with a skipped check.
    if strict_hashes:
        expected_map = dict(mac.CANONICAL_STEP8A_SHA256)
        basis = "frozen canonical literals"
    else:
        expected_map = dict((metadata or {}).get("canonical_step8a_hashes", {}))
        basis = "hashes recorded in completion_metadata.json"
    mismatches = {
        k: {"expected": expected_map.get(k), "observed": v["sha256"]}
        for k, v in inventory.items()
        if not expected_map or v["sha256"] != expected_map.get(k)
    }
    report.ok("1.canonical_step8a_hashes", not mismatches,
              f"Step8A hashes match the {basis}",
              mismatches or f"all four match the {basis}",
              evidence_path=str(root / "completion_metadata.json"))

    # --- 2/3/7. cardinality --------------------------------------------------
    if summary is None:
        report.ok("2.twelve_directed_pairs", False, 12, "directed_pair_summary.csv missing")
    else:
        report.ok("2.twelve_directed_pairs", len(summary) == 12, 12, len(summary))
        keys = list(zip(summary["source_experiment"], summary["target_experiment"]))
        report.ok("3.no_duplicate_pairs", len(set(keys)) == len(keys),
                  "unique (source, target)", f"{len(keys) - len(set(keys))} duplicate(s)")
        report.ok("7.no_self_pairs", all(s != t for s, t in keys),
                  "no diagonal pair", [f"{s}->{t}" for s, t in keys if s == t] or "none")

    # --- 4. weighted DI directional -----------------------------------------
    if summary is not None:
        di = {
            (r["source_experiment"], r["target_experiment"]): r["target_mean_dissimilarity"]
            for _, r in summary.iterrows()
        }
        asymmetric = any(
            di.get((a, b)) != di.get((b, a))
            for a, b in mac.unordered_pairs(mac.CANONICAL_EXPERIMENTS)
        )
        report.ok("4.weighted_di_directional", asymmetric,
                  "at least one A->B != B->A", asymmetric)
        source_constant = True
        for source in mac.CANONICAL_EXPERIMENTS:
            rows = summary[summary["source_experiment"] == source]
            for column in ("source_pairwise_mean_distance", "source_distance_normaliser",
                           "training_di_upper_whisker_threshold"):
                if rows[column].nunique(dropna=False) > 1:
                    source_constant = False
        report.ok("4b.source_quantities_constant_per_source", source_constant,
                  "normaliser and threshold identical across a source's 3 targets",
                  source_constant)

    # --- 5. climate symmetry -------------------------------------------------
    if climate_pairs is None:
        report.ok("5.climate_symmetric", False, "6 unordered climate rows",
                  "climate distance is ABSENT -- a required component of the "
                  "completion analysis is missing")
    else:
        report.ok("5.climate_symmetric", len(climate_pairs) == 6,
                  "6 unordered rows", len(climate_pairs))
        if summary is not None and "climate_distance" in summary.columns:
            lookup = {
                (r["source_experiment"], r["target_experiment"]): r["climate_distance"]
                for _, r in summary.iterrows()
            }
            sym = all(
                lookup.get((a, b)) == lookup.get((b, a))
                for a, b in mac.unordered_pairs(mac.CANONICAL_EXPERIMENTS)
            )
            report.ok("5b.climate_reversed_equal", sym, "exact copy on reversal", sym)

    # --- 6. geographic symmetry ---------------------------------------------
    if geo_pairs is None:
        report.ok("6.geographic_symmetric", False, "6 unordered geographic rows",
                  "geographic distance is ABSENT -- a required component of the "
                  "completion analysis is missing")
    else:
        report.ok("6.geographic_symmetric", len(geo_pairs) == 6,
                  "6 unordered rows", len(geo_pairs))
        report.ok("6b.geodesic_route",
                  set(geo_pairs["geodesic_implementation"]) == {"geographiclib_wgs84"},
                  "geographiclib_wgs84", sorted(set(geo_pairs["geodesic_implementation"])))
        if summary is not None:
            lookup = {
                (r["source_experiment"], r["target_experiment"]):
                    r["centroid_geodesic_distance_km"]
                for _, r in summary.iterrows()
            }
            sym = all(
                lookup.get((a, b)) == lookup.get((b, a))
                for a, b in mac.unordered_pairs(mac.CANONICAL_EXPERIMENTS)
            )
            report.ok("6c.geographic_reversed_equal", sym, "exact copy on reversal", sym)
            no_pop = not any(
                "population_centroid" in c and c != "population_centroid_reported"
                for c in summary.columns
            )
            report.ok("6d.no_population_centroid_column", no_pop,
                      "no *population_centroid* value column", no_pop)

    # --- 8. target labels never read ----------------------------------------
    forbidden = {"burned", "burn_date", "burn_month", "burn_day_of_year", "label_source"}
    leaked: list[str] = []
    for path in sorted(root.rglob("*.csv")):
        frame = _load_csv(path)
        if frame is None:
            continue
        hits = forbidden & set(frame.columns)
        hits |= {c for c in frame.columns if c.startswith("y_prob_")}
        if hits:
            leaked.append(f"{path.relative_to(root)}: {sorted(hits)}")
    report.ok("8.target_labels_never_read", not leaked,
              "no label column in any output", leaked or "none")
    if metadata:
        fw = metadata.get("target_label_firewall", {})
        report.ok("8b.firewall_flags", (
            fw.get("target_label_used") is False
            and fw.get("target_burn_date_used") is False
            and fw.get("target_transfer_metric_used") is False
        ), "all three false", fw, evidence_path="completion_metadata.json")

    # --- 9. source importance provenance ------------------------------------
    inv = _load_json(root / "config" / "feature_importance_inventory.json")
    if inv is None:
        report.ok("9.source_importance_provenance", False,
                  "feature_importance_inventory.json", "missing")
    else:
        ok = (
            inv.get("importance_method") == mac.IMPORTANCE_METHOD
            and inv.get("population_filter") == mac.IMPORTANCE_POPULATION
            and inv.get("model_filter") == mac.IMPORTANCE_MODEL
            and inv.get("model_algorithm") == mac.IMPORTANCE_MODEL_ALGORITHM
            and inv.get("source_label_policy", {}).get("source_label_used") is True
            and inv.get("source_label_policy", {}).get(
                "source_label_read_directly_by_completion_module") is False
        )
        report.ok("9.source_importance_provenance", ok,
                  "impurity Gini, burnable/thermal, source_label_used=true",
                  {k: inv.get(k) for k in ("importance_method", "population_filter", "model_filter")})

    # --- 10/11/12. weights ---------------------------------------------------
    if weights is None:
        report.ok("10.weights_valid", False, "source_feature_weights.csv", "missing")
    else:
        problems: list[str] = []
        for source in sorted(set(weights["source_experiment"])):
            rows = weights[weights["source_experiment"] == source]
            if len(rows) != 10:
                problems.append(f"{source}: {len(rows)} rows, expected 10")
            values = rows["weight"].astype(float)
            if not values.apply(math.isfinite).all():
                problems.append(f"{source}: non-finite weight")
            if (values < 0).any():
                problems.append(f"{source}: negative weight")
            if abs(float(values.sum()) - 1.0) > 1e-9:
                problems.append(f"{source}: weights sum to {float(values.sum())}")
        report.ok("10.weights_valid", not problems,
                  "10 finite non-negative weights summing to 1 per source",
                  problems or "ok")

        zero_declared = summary is not None and all(
            json.loads(v) == sorted(
                weights[(weights["source_experiment"] == s) & (weights["is_zero_weight"])]["feature"]
            )
            for s, v in zip(summary["source_experiment"], summary["zero_weight_features"])
        )
        report.ok("11.zero_weight_policy_truthful", bool(zero_declared),
                  "declared zero-weight set equals the observed one", bool(zero_declared))

        categorical = mac.categorical_feature()
        cat_rows = weights[weights["feature"] == categorical]
        group_ok = True
        for _, row in cat_rows.iterrows():
            contributions = json.loads(row["dummy_level_contributions"] or "{}")
            if abs(sum(contributions.values()) - float(row["raw_importance"])) > 1e-12:
                group_ok = False
        report.ok("12.categorical_handling_truthful", group_ok,
                  "landcover weight equals the sum of its dummy contributions", group_ok)
        if summary is not None:
            report.ok("12b.categorical_policy_id",
                      set(summary["categorical_policy_id"]) == {mac.CATEGORICAL_POLICY_ID},
                      mac.CATEGORICAL_POLICY_ID,
                      sorted(set(summary["categorical_policy_id"])))

    # --- 13. normaliser + threshold -----------------------------------------
    if thresholds is None:
        report.ok("13.normaliser_and_threshold", False,
                  "source_threshold_diagnostics.csv", "missing")
    else:
        problems = []
        if len(thresholds) != 4:
            problems.append(f"{len(thresholds)} rows, expected 4")
        for _, row in thresholds.iterrows():
            if row["normaliser_method"] != mac.NORMALISER_METHOD:
                problems.append(f"{row['source_experiment']}: normaliser_method")
            if bool(row["normaliser_uses_folds"]):
                problems.append(f"{row['source_experiment']}: normaliser_uses_folds is true")
            n = int(row["source_rows_reference"])
            if int(row["n_distinct_source_pairs"]) != n * (n - 1) // 2:
                problems.append(f"{row['source_experiment']}: pair count")
            if int(row["accumulated_pair_count"]) != int(row["n_distinct_source_pairs"]):
                problems.append(f"{row['source_experiment']}: accumulated pair count")
            if float(row["source_distance_normaliser"]) != float(row["source_pairwise_mean_distance"]):
                problems.append(f"{row['source_experiment']}: normaliser != pairwise mean")
            recomputed = min(
                float(row["training_di_max_threshold"]),
                float(row["training_di_q3"]) + 1.5 * float(row["training_di_iqr"]),
            )
            if abs(recomputed - float(row["training_di_upper_whisker_threshold"])) > 1e-12:
                problems.append(f"{row['source_experiment']}: whisker recomputation")
            if row["primary_threshold_method"] != mac.PRIMARY_THRESHOLD_METHOD:
                problems.append(f"{row['source_experiment']}: primary_threshold_method")
            if bool(row["q95_is_operative"]):
                problems.append(f"{row['source_experiment']}: q95_is_operative is true")
            if bool(row["fold_assignment_reads_label"]):
                problems.append(f"{row['source_experiment']}: fold assignment reads label")
        report.ok("13.normaliser_and_threshold", not problems,
                  "pairwise fold-free normaliser; upper-whisker primary; q95 secondary",
                  problems or "ok")

        holdout_tokens = [
            c for c in thresholds.columns
            if "holdout" in c and "normaliser" in c
        ]
        report.ok("13b.no_holdout_normaliser_token", not holdout_tokens,
                  "no holdout-derived normaliser field", holdout_tokens or "none")

    # --- 13c. primary classification used the whisker ------------------------
    parquet = root / "weighted_predictor_space" / "target_cell_dissimilarity.parquet"
    if summary is not None and parquet.is_file():
        cells = pd.read_parquet(parquet)
        problems = []
        for _, row in summary.iterrows():
            subset = cells[
                (cells["source_experiment"] == row["source_experiment"])
                & (cells["target_experiment"] == row["target_experiment"])
            ]
            di = pd.to_numeric(subset["weighted_dissimilarity"], errors="coerce")
            inside = int((di <= float(row["training_di_upper_whisker_threshold"])).sum())
            expected = inside / int(row["target_rows"])
            if abs(expected - float(row["fraction_inside_weighted_aoa"])) > 1e-12:
                problems.append(f"{row['direction']}: inside fraction != whisker result")
        report.ok("13c.primary_classification_uses_whisker", not problems,
                  "inside fraction recomputes against the upper whisker",
                  problems or "ok", evidence_path=str(parquet))

    # --- 14. target-cell cardinality ----------------------------------------
    if summary is not None and parquet.is_file():
        cells = pd.read_parquet(parquet)
        expected_total = int(summary["target_rows"].sum())
        report.ok("14.target_cell_cardinality", len(cells) == expected_total,
                  expected_total, len(cells), evidence_path=str(parquet))
        problems = []
        for _, row in summary.iterrows():
            n = len(cells[
                (cells["source_experiment"] == row["source_experiment"])
                & (cells["target_experiment"] == row["target_experiment"])
            ])
            if n != int(row["target_rows"]):
                problems.append(f"{row['direction']}: {n} != {int(row['target_rows'])}")
            total = (
                float(row["fraction_inside_weighted_aoa"])
                + float(row["fraction_outside_weighted_aoa"])
                + float(row["fraction_not_assessable"])
            )
            if abs(total - 1.0) > 1e-12:
                problems.append(f"{row['direction']}: fractions sum to {total}")
        report.ok("14b.per_pair_counts_and_fractions", not problems,
                  "per-pair counts match and fractions sum to 1", problems or "ok")

    # --- 15/16. climate ------------------------------------------------------
    climate_inv = _load_json(root / "config" / "climate_input_inventory.json")
    if climate_inv is None:
        report.ok("15.climate_contract", False, "climate_input_inventory.json", "missing")
    else:
        contract = climate_inv.get("contract", {})
        ok = (
            contract.get("collection") == mac.CLIMATE_COLLECTION
            and contract.get("reference_period") == f"{mac.CLIMATE_PERIOD_START}/{mac.CLIMATE_PERIOD_END}"
            and contract.get("season_months") == list(mac.CLIMATE_SEASON_MONTHS)
            and contract.get("climate_feature_count") == 4
            and contract.get("climate_features") == list(mac.CLIMATE_FEATURES)
            and contract.get("land_mask") == mac.CLIMATE_LAND_MASK
        )
        report.ok("15.climate_contract", ok,
                  "TerraClimate 1991-2020, months 6-9, 4 variables, native land support",
                  {k: contract.get(k) for k in
                   ("collection", "reference_period", "climate_feature_count", "land_mask")})
        # ERA5 is a DEFERRED sensitivity: the contract records it as an
        # explicit `false` flag. The check is that it is not USED, not that
        # the string is absent -- recording the deferral is required.
        era5_used = (
            contract.get("era5_land_cross_check_in_initial_run") is not False
            or "ERA5" in str(contract.get("collection", "")).upper()
            or any(
                "ERA5" in str(v).upper()
                for v in (climate_inv.get("variable_recipes") or [])
            )
        )
        report.ok("15b.era5_not_used_in_initial_run", not era5_used,
                  "era5_land_cross_check_in_initial_run is false and no ERA5 band is read",
                  {"flag": contract.get("era5_land_cross_check_in_initial_run"),
                   "collection": contract.get("collection")})

    forbidden_roots = ("step5", "step5c", "data/modis", "data/ndvi_timeseries",
                       "data/landsat_timeseries", "data/current_period")
    blob = json.dumps(climate_inv or {})
    bad_paths = [p for p in forbidden_roots if p in blob]
    report.ok("16.no_event_period_predictor_as_climate", not bad_paths,
              "no event-period path feeds any climate_* field", bad_paths or "none")

    if climate_pairs is not None:
        problems = []
        for _, row in climate_pairs.iterrows():
            contributions = json.loads(row["climate_component_contributions"])
            if len(contributions) != 4:
                problems.append(f"{row['experiment_a']}/{row['experiment_b']}: not 4 components")
            if abs(sum(contributions.values()) - float(row["climate_distance_squared"])) > 1e-12:
                problems.append(f"{row['experiment_a']}/{row['experiment_b']}: contributions sum")
        report.ok("15c.climate_component_contributions", not problems,
                  "4 contributions summing to squared distance", problems or "ok")

    # --- 17/18. geometry -----------------------------------------------------
    try:
        checked = mac.assert_geometry_matches_registry()
        report.ok("17.geometry_source_canonical", True,
                  "pinned bboxes match core/regions.py", sorted(checked))
    except SystemExit as exc:
        report.ok("17.geometry_source_canonical", False,
                  "pinned bboxes match core/regions.py", str(exc))
    report.ok("17b.extended_evia_not_narrow",
              mac.CANONICAL_AOI_BBOX["evia_2021_extended"] == (23.05, 38.55, 23.85, 39.15),
              (23.05, 38.55, 23.85, 39.15),
              mac.CANONICAL_AOI_BBOX["evia_2021_extended"])

    can_recompute = geodesic_inverse is not None or mac.geographiclib_available()
    if geo_pairs is not None and can_recompute:
        problems = []
        for _, row in geo_pairs.iterrows():
            bbox_a = json.loads(row["bbox_a"])
            bbox_b = json.loads(row["bbox_b"])
            lon_a, lat_a = mac.bbox_centre(bbox_a)
            lon_b, lat_b = mac.bbox_centre(bbox_b)
            recomputed = mac.geodesic_distance_km(
                lon_a, lat_a, lon_b, lat_b, geodesic_inverse=geodesic_inverse
            )
            if abs(recomputed - float(row["centroid_geodesic_distance_km"])) > 1e-6:
                problems.append(f"{row['experiment_a']}/{row['experiment_b']}")
        report.ok("18.km_recomputable", not problems,
                  "centroid distance recomputes from the stored bboxes to <= 1e-6 km",
                  problems or "ok")
    elif geo_pairs is None:
        report.ok("18.km_recomputable", False,
                  "centroid distance recomputes from the stored bboxes",
                  "geographic distance is ABSENT")
    else:
        report.ok("18.km_recomputable", False,
                  "centroid distance recomputes from the stored bboxes",
                  "geographiclib is not installed, so the shipped kilometre "
                  "values cannot be independently recomputed")

    # --- 19/20/21. isolation -------------------------------------------------
    if metadata:
        outside = [
            p for p in metadata.get("output_sha256", {})
            if p.startswith("..") or Path(p).is_absolute()
        ]
        report.ok("19.output_inside_namespace", not outside,
                  "every output path is namespace-relative", outside or "none")
        report.ok("20.marginal_aoa_v1_unchanged",
                  metadata.get("existing_marginal_aoa_outputs_modified") is False,
                  False, metadata.get("existing_marginal_aoa_outputs_modified"))
        report.ok("21.canonical_outputs_unchanged",
                  metadata.get("canonical_outputs_modified") is False,
                  False, metadata.get("canonical_outputs_modified"))
    else:
        for check_id in ("19.output_inside_namespace", "20.marginal_aoa_v1_unchanged",
                         "21.canonical_outputs_unchanged"):
            report.ok(check_id, False, "completion_metadata.json", "missing")

    # --- 22. transfer metrics only under comparison/ -------------------------
    transfer_columns = {
        "within_target_auc", "raw_auc", "adapted_auc", "raw_gap",
        "adaptation_effect", "remaining_gap", "recovered_fraction",
        "recovery_status", "chance_level",
    }
    leaked = []
    for path in sorted(root.rglob("*.csv")):
        if path.parent.name == "comparison":
            continue
        frame = _load_csv(path)
        if frame is None:
            continue
        hits = transfer_columns & set(frame.columns)
        if hits:
            leaked.append(f"{path.relative_to(root)}: {sorted(hits)}")
    report.ok("22.transfer_metrics_only_in_comparison", not leaked,
              "no transfer column outside comparison/", leaked or "none")

    if ranking is None:
        report.ok("22b.primary_is_raw_thermal_roc_auc", False,
                  "raw_thermal_roc_auc", "ranking_summary.csv is ABSENT")
    else:
        primary = ranking[ranking["is_primary_comparison"].astype(bool)]
        names = sorted(set(primary["transfer"])) if len(primary) else []
        report.ok("22b.primary_is_raw_thermal_roc_auc", names == ["raw_thermal_roc_auc"],
                  ["raw_thermal_roc_auc"], names)
        present = sorted(set(ranking["transfer"]))
        expected = sorted({q[0] for q in mac.TRANSFER_QUANTITIES})
        report.ok("22c.secondary_block_complete", present == expected,
                  expected, present)
        pvalue_columns = [c for c in ranking.columns
                          if "p_value" in c or "pvalue" in c or "significance" in c]
        report.ok("22d.no_p_values", not pvalue_columns,
                  "no p-value column", pvalue_columns or "none")

    # --- 23/24. truthful "did not do" flags ----------------------------------
    source_text = Path(mac.__file__).read_text(encoding="utf-8")
    estimator_hits = [
        token for token in ("sklearn.ensemble", "sklearn.linear_model", "xgboost", ".fit(")
        if token in source_text
    ]
    report.ok("23.model_fitted_false", not estimator_hits,
              "no estimator import and no .fit( call", estimator_hits or "none")
    if metadata:
        report.ok("23b.metadata_model_fit_false",
                  metadata.get("model_fit") is False and metadata.get("bootstrap_run") is False,
                  "model_fit=false, bootstrap_run=false",
                  {"model_fit": metadata.get("model_fit"),
                   "bootstrap_run": metadata.get("bootstrap_run")})
    gee_hits = [t for t in ("import ee", "ee.Image", "ee.ImageCollection") if t in source_text]
    report.ok("24.gee_query_false", not gee_hits,
              "the module imports no Earth Engine API", gee_hits or "none")

    # --- 25. metadata binds every output hash --------------------------------
    if metadata:
        on_disk = {
            str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).parts[0] != mac.STAGE_MARKER_DIR
        }
        recorded = set(metadata.get("output_sha256", {}))
        # completion_metadata.json is written after the hash map is built;
        # `stages/` is run bookkeeping with its own per-stage hashes.
        missing = sorted(on_disk - recorded - {"completion_metadata.json"})
        extra = sorted(recorded - on_disk)
        report.ok("25.metadata_binds_outputs", not missing and not extra,
                  "output_sha256 covers every file in the namespace",
                  {"missing": missing, "extra": extra})

    # --- 26. every required stage completed and is hash-valid ---------------
    required_stages = ("plan", "weighted-predictor-space", "climate-distance",
                       "geographic-distance", "compare")
    incomplete: list[str] = []
    for stage in required_stages:
        try:
            state = mac.verify_stage_complete(analysis_id, stage, output_root)
        except SystemExit as exc:
            incomplete.append(f"{stage}: {exc}")
            continue
        if not state["complete"]:
            incomplete.append(f"{stage}: {state['reason']}")
    report.ok("26.required_stages_complete", not incomplete,
              "plan, weighted-predictor-space, climate-distance, "
              "geographic-distance and compare all complete and hash-valid",
              incomplete or "all complete")

    climate_export = mac.verify_stage_complete(
        analysis_id, mac.STAGE_CLIMATE_EXPORT, output_root
    )
    report.ok("26b.climate_export_complete", climate_export["complete"],
              "the authorised climate export is complete",
              climate_export.get("reason", "complete"))

    # --- 27. no required component value is null ----------------------------
    if summary is None:
        report.ok("27.no_null_required_components", False,
                  "no null climate/geographic value", "summary missing")
    else:
        null_columns: list[str] = []
        for column in ("climate_distance", "centroid_geodesic_distance_km"):
            if column not in summary.columns:
                null_columns.append(f"{column}: column absent")
            elif summary[column].isna().any():
                n = int(summary[column].isna().sum())
                null_columns.append(f"{column}: {n} null value(s)")
        report.ok("27.no_null_required_components", not null_columns,
                  "every directed row carries a real climate and geographic value",
                  null_columns or "ok")

    return report


# =============================================================================
# Rendering
# =============================================================================
TECHNICAL = {"1", "2", "3", "7", "14", "14b", "17", "17b", "18", "25", "00",
             "26", "26b"}
SCIENTIFIC = {"4", "4b", "5", "5b", "6", "6b", "6c", "9", "10", "11", "12", "12b",
              "13", "13b", "13c", "15", "15b", "15c", "16", "22b", "22c",
              "22d", "27"}
# 15b is emitted as "15b.era5_not_used_in_initial_run"; its head is "15b".
SAFETY = {"6d", "8", "8b", "19", "20", "21", "22", "23", "23b", "24"}


def _bucket(check_id: str) -> str:
    head = check_id.split(".")[0]
    if head in TECHNICAL:
        return "TECHNICAL STATUS"
    if head in SCIENTIFIC:
        return "SCIENTIFIC-CONTRACT STATUS"
    if head in SAFETY:
        return "NAMESPACE / PROVENANCE SAFETY"
    return "TECHNICAL STATUS"


def render(report: Report, mode: str) -> str:
    buckets: dict[str, list[dict[str, Any]]] = {
        "TECHNICAL STATUS": [], "SCIENTIFIC-CONTRACT STATUS": [],
        "NAMESPACE / PROVENANCE SAFETY": [],
    }
    for check in report.checks:
        buckets[_bucket(check["check_id"])].append(check)

    lines = [f"marginal_aoa_completion validator -- mode: {mode}", ""]
    for name, checks in buckets.items():
        failed = [c for c in checks if c["status"] == FAIL]
        skipped = [c for c in checks if c["status"] == SKIP]
        status = FAIL if failed else (f"{PASS} ({len(skipped)} skipped)" if skipped else PASS)
        lines.append(f"{name}: {status}")
        for check in checks:
            marker = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[check["status"]]
            lines.append(f"{marker}{check['check_id']}")
            if check["status"] != PASS:
                lines.append(f"        expected: {check['expected']}")
                lines.append(f"        observed: {check['observed']}")
        lines.append("")
    overall = FAIL if report.failed else PASS
    # An actual validation may never PASS with a skipped required check.
    if mode == "actual" and report.skipped:
        overall = FAIL
        lines.append(
            "OVERALL: an actual validation cannot PASS with a skipped required "
            f"check ({len(report.skipped)} skipped)."
        )
    lines.append(f"OVERALL STATUS: {overall}")
    lines.append(f"checks: {len(report.checks)}  failed: {len(report.failed)}  skipped: {len(report.skipped)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a marginal_aoa_completion artifact (27 checks)."
    )
    parser.add_argument("--mode", choices=("dry-run", "actual"), default="dry-run")
    parser.add_argument("--analysis-id", default=None,
                        help="Required for --mode actual.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--json", action="store_true", help="Emit the raw check records.")
    parser.add_argument(
        "--no-strict-hashes", action="store_true",
        help=(
            "Skip the frozen canonical Step8A hash comparison. Intended only "
            "for an injected synthetic tree; never for a production artifact."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    experiments_root = Path(args.experiments_root) if args.experiments_root else None

    strict = not args.no_strict_hashes
    if args.mode == "dry-run":
        report = validate_dry_run(output_root, experiments_root, strict_hashes=strict)
    else:
        if not args.analysis_id:
            print("--analysis-id is required for --mode actual", file=sys.stderr)
            return 2
        report = validate_actual(
            args.analysis_id, output_root, experiments_root, strict_hashes=strict,
        )

    if args.json:
        print(json.dumps(report.checks, indent=2, default=str))
    else:
        print(render(report, args.mode))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
