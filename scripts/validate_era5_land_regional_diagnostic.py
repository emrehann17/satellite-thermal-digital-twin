#!/usr/bin/env python3
"""Validator for the ERA5-Land AOI-level regional meteorology diagnostic.

Two modes:

    dry-run  -- contract-only checks (cohort, order, window conversion,
                climatology mapping, schema, analysis_id determinism). Reads no
                produced artifact, writes nothing, contacts no Earth Engine.
    actual   -- every check, against a produced namespace.

Every check emits {check_id, status, expected, observed, evidence_path}. Any
FAIL makes the overall status FAIL and the exit code 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.era5_land_regional_diagnostic as era5  # noqa: E402
from core.regions import get_experiment  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIPPED"

EXPECTED_ROW_COUNT = 5


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

    def skip(self, check_id: str, expected: Any, note: str) -> None:
        self.add(check_id, SKIP, expected, None, None, note)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == FAIL]

    @property
    def skipped(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == SKIP]


def _load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_csv(path: Path) -> Optional[list[dict[str, str]]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError):
        return None


def _csv_header(path: Path) -> Optional[list[str]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle), None)
    except (OSError, ValueError):
        return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _walk_numbers(node: Any):
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_numbers(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_numbers(value)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        yield float(node)


# =============================================================================
# Dry-run: contract only
# =============================================================================
def validate_dry_run(output_root: Optional[Path] = None,
                     experiments: Optional[list[str]] = None) -> Report:
    report = Report()

    def _namespace_listing() -> set[str]:
        root_dir = era5.diagnostics_root(output_root)
        return {p.name for p in root_dir.iterdir()} if root_dir.is_dir() else set()

    listing_before = _namespace_listing()
    plan = era5.build_plan(experiments, output_root)
    listing_after = _namespace_listing()
    is_default = experiments is None

    # C01 default cohort: exactly the frozen five, in order
    if is_default:
        report.ok(
            "C01_default_cohort_exact_set_and_order",
            tuple(plan["experiment_ids"]) == era5.DEFAULT_EXPERIMENTS,
            list(era5.DEFAULT_EXPERIMENTS), plan["experiment_ids"],
        )
        report.ok(
            "C02_default_cohort_row_count",
            len(plan["experiment_ids"]) == EXPECTED_ROW_COUNT,
            EXPECTED_ROW_COUNT, len(plan["experiment_ids"]),
        )
        report.ok(
            "C03_mugla_2022_absent_by_default",
            "mugla_2022" not in plan["experiment_ids"],
            "mugla_2022 not in the default cohort", plan["experiment_ids"],
            note=era5.NON_DEFAULT_EXPERIMENTS.get("mugla_2022"),
        )
    else:
        for check_id in ("C01_default_cohort_exact_set_and_order",
                         "C02_default_cohort_row_count",
                         "C03_mugla_2022_absent_by_default"):
            report.skip(check_id, "default cohort", "explicit --experiments given")

    # C04 source contract
    report.ok(
        "C04_collection_and_bands",
        plan["collection"] == era5.COLLECTION_ID
        and tuple(plan["bands"]) == era5.SOURCE_BANDS,
        {"collection": era5.COLLECTION_ID, "bands": list(era5.SOURCE_BANDS)},
        {"collection": plan["collection"], "bands": plan["bands"]},
    )

    # C05 climatology years
    report.ok(
        "C05_reference_years_exact",
        tuple(plan["climatology_years"]) == (2017, 2018, 2019, 2020),
        [2017, 2018, 2019, 2020], plan["climatology_years"],
    )

    # C06 observed windows equal the registry, C07 inclusive -> exclusive
    window_ok, exclusive_ok, mapping_ok = True, True, True
    for entry in plan["experiments"]:
        record = get_experiment(entry["experiment_id"])
        if record["region_key"] != entry["region_key"]:
            window_ok = False
        for kind in era5.WINDOW_KINDS:
            observed = entry["observed_windows"][kind]
            if (observed["start_date"] != record[f"{kind}_start_date"]
                    or observed["end_date_inclusive"] != record[f"{kind}_end_date"]):
                window_ok = False
            if observed["end_date_exclusive"] != era5.exclusive_end(
                    record[f"{kind}_end_date"]):
                exclusive_ok = False
            mapped = entry["climatology_windows"][kind]
            if [m["reference_year"] for m in mapped] != list(era5.CLIMATOLOGY_YEARS):
                mapping_ok = False
            for m in mapped:
                start = m["start_date"]
                if (start[5:] != observed["start_date"][5:]
                        or m["end_date_inclusive"][5:]
                        != observed["end_date_inclusive"][5:]
                        or start[:4] != str(m["reference_year"])):
                    mapping_ok = False
    report.ok("C06_observed_windows_match_registry", window_ok,
              "registry predictor/label dates and region_key", "see plan")
    report.ok("C07_inclusive_end_converted_to_exclusive", exclusive_ok,
              "end_date_exclusive == registry end + 1 day", "see plan")
    report.ok("C08_climatology_window_mapping", mapping_ok,
              "same month/day in each of 2017-2020", "see plan")

    # C09 schema
    columns = plan["summary_columns"]
    expected_columns = era5.summary_columns()
    report.ok("C09_summary_schema", columns == expected_columns,
              f"{len(expected_columns)} ordered columns", f"{len(columns)} columns")

    # C10 analysis_id determinism
    recomputed = era5.compute_analysis_id(plan["scientific_config"])
    report.ok("C10_analysis_id_matches_scientific_config",
              recomputed == plan["analysis_id"], recomputed, plan["analysis_id"])
    report.ok("C11_analysis_id_is_stable",
              era5.build_plan(experiments, output_root)["analysis_id"]
              == plan["analysis_id"],
              plan["analysis_id"], "second build_plan()")

    # C12 the hashed config carries every required clause
    config = plan["scientific_config"]
    required = {
        "experiment_ids": config.get("experiment_ids"),
        "collection": config.get("source", {}).get("collection"),
        "bands": config.get("source", {}).get("bands"),
        "observed_window_source": config.get("observed_window_source"),
        "climatology_years": config.get("climatology", {}).get("reference_years"),
        "sd_ddof": config.get("climatology", {}).get("sd_ddof"),
        "rh_constants": config.get("derived_variables", {})
                              .get("relative_humidity_percent", {}).get("constants"),
        "wind_recipe": config.get("derived_variables", {})
                             .get("wind_speed_m_s", {}).get("recipe"),
        "precipitation_recipe": config.get("derived_variables", {})
                                      .get("precipitation_mm", {}).get("recipe"),
        "spatial_weighting": config.get("spatial_aggregation", {}).get("semantics"),
        "temporal_aggregation": config.get("temporal_aggregation", {}).get("semantics"),
        "output_schema_version": config.get("output_schema_version"),
    }
    report.ok("C12_scientific_config_completeness",
              all(v not in (None, {}, []) for v in required.values()),
              "every hashed clause present",
              {k: (v is not None) for k, v in required.items()})

    # C15 the hourly-completeness rule is hashed into analysis_id
    temporal = config.get("temporal_aggregation") or {}
    report.ok("C15_hourly_completeness_rule_is_hashed",
              temporal.get("hourly_completeness_rule") == era5.HOURLY_COMPLETENESS_RULE
              and temporal.get("expected_hours_formula")
              == f"n_days_inclusive * {era5.HOURS_PER_DAY}"
              and temporal.get("timestamp_convention") == era5.TIMESTAMP_CONVENTION
              and temporal.get("incomplete_window_policy") == "fail_closed",
              "completeness rule, expected-hours formula, timestamp convention "
              "and fail-closed incomplete-window policy present in the hashed config",
              {k: temporal.get(k) for k in ("hourly_completeness_rule",
                                            "expected_hours_formula",
                                            "timestamp_convention",
                                            "incomplete_window_policy")})

    # C13 planned namespace
    root = Path(plan["planned_output_root"])
    report.ok("C13_planned_namespace_layout",
              root.name == plan["analysis_id"]
              and root.parent.name == era5.DIAGNOSTIC_NAMESPACE
              and sorted(plan["planned_output_paths"]) == sorted(era5.EXPECTED_FILENAMES),
              f"outputs/diagnostics/{era5.DIAGNOSTIC_NAMESPACE}/<analysis_id>/ with "
              f"{sorted(era5.EXPECTED_FILENAMES)}",
              str(root))
    report.ok("C14_planning_created_no_namespace", listing_before == listing_after,
              "planning creates no directory",
              sorted(listing_after - listing_before) or "nothing created",
              str(era5.diagnostics_root(output_root)))
    return report


# =============================================================================
# Actual: produced artifact
# =============================================================================
def validate_actual(analysis_id: str, output_root: Optional[Path] = None) -> Report:
    report = Report()
    root = era5.analysis_root(analysis_id, output_root)
    csv_path = root / era5.SUMMARY_CSV_NAME
    json_path = root / era5.SUMMARY_JSON_NAME
    contract_path = root / era5.CONTRACT_JSON_NAME
    manifest_path = root / era5.MANIFEST_JSON_NAME

    present = [name for name in era5.EXPECTED_FILENAMES if (root / name).is_file()]
    if not report.ok("A01_all_expected_files_present",
                     len(present) == len(era5.EXPECTED_FILENAMES),
                     list(era5.EXPECTED_FILENAMES), present, str(root)):
        return report

    contract = _load_json(contract_path) or {}
    summary = _load_json(json_path) or {}
    manifest = _load_json(manifest_path) or {}
    rows = _load_csv(csv_path) or []
    header = _csv_header(csv_path) or []
    config = contract.get("scientific_config") or {}

    # A02 analysis_id is the hash of the scientific config
    recomputed = era5.compute_analysis_id(config) if config else None
    report.ok("A02_analysis_id_matches_scientific_config",
              recomputed == analysis_id, analysis_id, recomputed, str(contract_path))
    report.ok("A03_namespace_directory_matches_analysis_id",
              root.name == analysis_id, analysis_id, root.name, str(root))

    # A04 expected experiment set/order from the contract
    contract_ids = list(config.get("experiment_ids") or [])
    csv_ids = [row.get("experiment_id") for row in rows]
    json_ids = list(summary.get("experiment_ids") or [])
    report.ok("A04_contract_csv_json_experiment_order_agree",
              bool(contract_ids) and contract_ids == csv_ids == json_ids,
              contract_ids, {"csv": csv_ids, "json": json_ids}, str(csv_path))

    # A05 exactly five rows, one unique row per experiment
    report.ok("A05_row_count_is_five", len(rows) == EXPECTED_ROW_COUNT,
              EXPECTED_ROW_COUNT, len(rows), str(csv_path))
    report.ok("A06_one_unique_row_per_experiment",
              len(csv_ids) == len(set(csv_ids)) and set(csv_ids) == set(contract_ids),
              sorted(set(contract_ids)), csv_ids, str(csv_path))

    # A07 mugla_2022 absent from the default five-AOI analysis
    is_default_cohort = tuple(contract_ids) == era5.DEFAULT_EXPERIMENTS
    report.ok("A07_mugla_2022_absent_from_default_analysis",
              "mugla_2022" not in contract_ids and "mugla_2022" not in csv_ids,
              "mugla_2022 absent",
              {"contract": contract_ids, "csv": csv_ids},
              note=None if is_default_cohort else
              "cohort is not the frozen default five")
    report.ok("A08_cohort_is_the_frozen_five", is_default_cohort,
              list(era5.DEFAULT_EXPERIMENTS), contract_ids, str(contract_path))

    # A09 reference years exactly 2017-2020
    years = list((config.get("climatology") or {}).get("reference_years") or [])
    report.ok("A09_reference_years_exact", years == [2017, 2018, 2019, 2020],
              [2017, 2018, 2019, 2020], years, str(contract_path))
    report.ok("A10_sd_ddof_is_one",
              (config.get("climatology") or {}).get("sd_ddof") == 1,
              1, (config.get("climatology") or {}).get("sd_ddof"), str(contract_path))

    # A11 collection/bands
    source = config.get("source") or {}
    report.ok("A11_collection_and_bands",
              source.get("collection") == era5.COLLECTION_ID
              and list(source.get("bands") or []) == list(era5.SOURCE_BANDS),
              {"collection": era5.COLLECTION_ID, "bands": list(era5.SOURCE_BANDS)},
              source, str(contract_path))

    # A12 schema: every expected metric column present, in order
    expected_columns = era5.summary_columns()
    report.ok("A12_csv_schema_exact", header == expected_columns,
              f"{len(expected_columns)} ordered columns",
              f"{len(header)} columns", str(csv_path))
    units = {
        variable: (config.get("derived_variables") or {}).get(variable, {}).get("units")
        for variable in era5.VARIABLES
    }
    report.ok("A13_units_declared_for_every_variable",
              units == dict(era5.VARIABLE_UNITS), dict(era5.VARIABLE_UNITS), units,
              str(contract_path))

    # A14 registry agreement: region keys and observed dates
    registry_ok, registry_detail = True, {}
    for row in rows:
        experiment_id = row.get("experiment_id")
        try:
            record = get_experiment(str(experiment_id))
        except ValueError:
            registry_ok = False
            registry_detail[str(experiment_id)] = "not registered"
            continue
        problems = []
        if row.get("region_key") != record["region_key"]:
            problems.append("region_key")
        for kind in era5.WINDOW_KINDS:
            if row.get(f"{kind}_start_date") != record[f"{kind}_start_date"]:
                problems.append(f"{kind}_start_date")
            if row.get(f"{kind}_end_date_inclusive") != record[f"{kind}_end_date"]:
                problems.append(f"{kind}_end_date_inclusive")
            if row.get(f"{kind}_end_date_exclusive") != era5.exclusive_end(
                    record[f"{kind}_end_date"]):
                problems.append(f"{kind}_end_date_exclusive")
        if problems:
            registry_ok = False
            registry_detail[str(experiment_id)] = problems
    report.ok("A14_region_keys_and_dates_match_registry", registry_ok,
              "CSV dates/region_key == core.regions", registry_detail or "match",
              str(csv_path))

    # A15/A16/A17 numeric contract, per metric
    finite_ok, z_rule_ok, agreement_ok = True, True, True
    finite_detail: list[str] = []
    z_detail: list[str] = []
    agreement_detail: list[str] = []
    json_rows = {str(r.get("experiment_id")): r for r in (summary.get("rows") or [])}

    for row in rows:
        experiment_id = str(row.get("experiment_id"))
        json_row = json_rows.get(experiment_id, {})
        for window in era5.WINDOW_KINDS:
            for variable, statistics in era5.VARIABLES.items():
                for statistic in statistics:
                    keys = {
                        field: era5.metric_key(window, variable, statistic, field)
                        for field in era5.METRIC_FIELDS
                    }
                    for field in ("observed", "climatology_mean", "climatology_sd",
                                  "anomaly"):
                        raw = row.get(keys[field])
                        if not _finite(raw):
                            finite_ok = False
                            finite_detail.append(f"{experiment_id}.{keys[field]}={raw!r}")

                    sd_raw = row.get(keys["climatology_sd"])
                    z_raw = row.get(keys["standardized_anomaly"])
                    zero_flag = str(row.get(keys["zero_climatology_sd"])).lower()
                    try:
                        sd_zero = float(sd_raw) == 0.0
                    except (TypeError, ValueError):
                        sd_zero = False
                    z_is_null = z_raw is None or str(z_raw).strip() == ""
                    if z_is_null != sd_zero or (zero_flag == "true") != sd_zero:
                        z_rule_ok = False
                        z_detail.append(
                            f"{experiment_id}.{window}.{variable}.{statistic}: "
                            f"sd={sd_raw!r} z={z_raw!r} flag={zero_flag!r}"
                        )
                    if not z_is_null and not _finite(z_raw):
                        finite_ok = False
                        finite_detail.append(
                            f"{experiment_id}.{keys['standardized_anomaly']}={z_raw!r}"
                        )

                    for field, key in keys.items():
                        csv_raw = row.get(key)
                        json_value = json_row.get(key)
                        if field == "zero_climatology_sd":
                            same = (str(csv_raw).lower()
                                    == ("true" if json_value else "false"))
                        elif json_value is None:
                            same = csv_raw is None or str(csv_raw).strip() == ""
                        else:
                            try:
                                same = float(csv_raw) == float(json_value)
                            except (TypeError, ValueError):
                                same = False
                        if not same:
                            agreement_ok = False
                            agreement_detail.append(
                                f"{experiment_id}.{key}: csv={csv_raw!r} json={json_value!r}"
                            )

    report.ok("A15_observed_climatology_anomaly_finite", finite_ok,
              "finite observed/climatology_mean/climatology_sd/anomaly",
              finite_detail[:10] or "all finite", str(csv_path))
    report.ok("A16_z_null_iff_zero_climatology_sd", z_rule_ok,
              "standardized_anomaly null exactly when climatology_sd == 0",
              z_detail[:10] or "rule holds", str(csv_path))
    report.ok("A17_csv_and_json_values_agree", agreement_ok,
              "identical values in both summaries",
              agreement_detail[:10] or "identical", str(json_path))

    # A18 no inf anywhere in the JSON summary
    non_finite = [v for v in _walk_numbers(summary) if not math.isfinite(v)]
    report.ok("A18_no_inf_or_nan_in_summary_json", not non_finite,
              "no inf/NaN", len(non_finite), str(json_path))
    raw_json_text = json_path.read_text(encoding="utf-8") if json_path.is_file() else ""
    report.ok("A19_no_inf_literal_in_serialized_json",
              "Infinity" not in raw_json_text and "NaN" not in raw_json_text,
              "no Infinity/NaN literal", "clean" if raw_json_text else "unreadable",
              str(json_path))

    # A20 climatology realizations retained and consistent
    realization_ok, realization_detail = True, []
    for detail in summary.get("experiments") or []:
        experiment_id = str(detail.get("experiment_id"))
        for window in era5.WINDOW_KINDS:
            metrics = ((detail.get("windows") or {}).get(window) or {}).get("metrics") or {}
            for variable, statistics in era5.VARIABLES.items():
                for statistic in statistics:
                    metric = metrics.get(f"{variable}_{statistic}") or {}
                    realizations = metric.get("climatology_realizations") or {}
                    if sorted(realizations) != [str(y) for y in era5.CLIMATOLOGY_YEARS]:
                        realization_ok = False
                        realization_detail.append(
                            f"{experiment_id}.{window}.{variable}.{statistic}: "
                            f"{sorted(realizations)}"
                        )
                        continue
                    values = [float(realizations[str(y)]) for y in era5.CLIMATOLOGY_YEARS]
                    mean = math.fsum(values) / len(values)
                    sd = era5.sample_standard_deviation(values)
                    if not (math.isclose(mean, float(metric["climatology_mean"]),
                                         rel_tol=1e-12, abs_tol=1e-12)
                            and math.isclose(sd, float(metric["climatology_sd"]),
                                             rel_tol=1e-12, abs_tol=1e-12)):
                        realization_ok = False
                        realization_detail.append(
                            f"{experiment_id}.{window}.{variable}.{statistic}: "
                            "mean/sd disagree with the four realizations"
                        )
    report.ok("A20_climatology_realizations_reproduce_mean_and_sd", realization_ok,
              "mean/sd recomputable from the four retained realizations",
              realization_detail[:10] or "consistent", str(json_path))

    # A21 manifest hashes/byte sizes
    manifest_ok, manifest_detail = True, []
    recorded = manifest.get("files")
    if not isinstance(recorded, list) or not recorded:
        manifest_ok = False
        manifest_detail.append("manifest records no files")
    else:
        names = sorted(str(entry.get("name")) for entry in recorded)
        if names != sorted(era5.PRODUCT_FILENAMES):
            manifest_ok = False
            manifest_detail.append(f"file set {names}")
        for entry in recorded:
            path = root / str(entry.get("name"))
            if not path.is_file():
                manifest_ok = False
                manifest_detail.append(f"missing {entry.get('name')}")
                continue
            if int(path.stat().st_size) != int(entry.get("bytes", -1)):
                manifest_ok = False
                manifest_detail.append(f"{path.name}: byte size mismatch")
            if era5.sha256_file(path) != entry.get("sha256"):
                manifest_ok = False
                manifest_detail.append(f"{path.name}: sha256 mismatch")
    report.ok("A21_manifest_hashes_and_sizes_match", manifest_ok,
              "every recorded size/sha256 matches the file on disk",
              manifest_detail[:10] or "match", str(manifest_path))
    report.ok("A22_manifest_analysis_id_matches",
              manifest.get("analysis_id") == analysis_id, analysis_id,
              manifest.get("analysis_id"), str(manifest_path))

    # A23 the namespace is classified complete by the module itself
    state = era5.namespace_state(analysis_id, output_root)
    report.ok("A23_namespace_state_is_complete", state.get("state") == "complete",
              "complete", state.get("state"), str(root), state.get("reason"))

    # A25 hourly completeness: observed windows
    observed_hours_ok, observed_hours_detail = True, []
    for row in rows:
        for window in era5.WINDOW_KINDS:
            try:
                n_days = int(row[f"{window}_n_days_inclusive"])
                n_hours = int(row[f"{window}_n_hours_observed"])
            except (KeyError, TypeError, ValueError):
                observed_hours_ok = False
                observed_hours_detail.append(
                    f"{row.get('experiment_id')}.{window}: unreadable hour/day counts"
                )
                continue
            if n_hours != n_days * era5.HOURS_PER_DAY:
                observed_hours_ok = False
                observed_hours_detail.append(
                    f"{row.get('experiment_id')}.{window}: {n_hours} hours for "
                    f"{n_days} day(s) (expected {n_days * era5.HOURS_PER_DAY})"
                )
    report.ok("A25_observed_windows_are_hourly_complete", observed_hours_ok,
              f"n_hours_observed == n_days_inclusive * {era5.HOURS_PER_DAY}",
              observed_hours_detail[:10] or "complete", str(csv_path))

    # A26 hourly completeness: every retained climatology realization
    clim_hours_ok, clim_hours_detail = True, []
    for detail in summary.get("experiments") or []:
        experiment_id = str(detail.get("experiment_id"))
        for window in era5.WINDOW_KINDS:
            block = ((detail.get("windows") or {}).get(window)) or {}
            mapped = {int(m["reference_year"]): int(m["n_days_inclusive"])
                      for m in (block.get("climatology_windows") or [])}
            recorded = block.get("climatology_n_hours") or {}
            if sorted(int(y) for y in recorded) != list(era5.CLIMATOLOGY_YEARS):
                clim_hours_ok = False
                clim_hours_detail.append(
                    f"{experiment_id}.{window}: hour counts recorded for "
                    f"{sorted(recorded)}"
                )
                continue
            for year in era5.CLIMATOLOGY_YEARS:
                expected = mapped.get(int(year), -1) * era5.HOURS_PER_DAY
                observed_hours = int(recorded[str(year)])
                if expected <= 0 or observed_hours != expected:
                    clim_hours_ok = False
                    clim_hours_detail.append(
                        f"{experiment_id}.{window}.{year}: {observed_hours} hours "
                        f"(expected {expected})"
                    )
    report.ok("A26_climatology_realizations_are_hourly_complete", clim_hours_ok,
              f"each realization has n_days_inclusive * {era5.HOURS_PER_DAY} hours",
              clim_hours_detail[:10] or "complete", str(json_path))

    # A27 the completeness rule is part of the hashed contract
    temporal = config.get("temporal_aggregation") or {}
    report.ok("A27_contract_records_the_hourly_completeness_rule",
              temporal.get("hourly_completeness_rule") == era5.HOURLY_COMPLETENESS_RULE
              and temporal.get("expected_hours_formula")
              == f"n_days_inclusive * {era5.HOURS_PER_DAY}"
              and temporal.get("incomplete_window_policy") == "fail_closed",
              {"hourly_completeness_rule": era5.HOURLY_COMPLETENESS_RULE,
               "expected_hours_formula": f"n_days_inclusive * {era5.HOURS_PER_DAY}",
               "incomplete_window_policy": "fail_closed"},
              {k: temporal.get(k) for k in ("hourly_completeness_rule",
                                            "expected_hours_formula",
                                            "incomplete_window_policy")},
              str(contract_path))

    # A24 no raster/export leaked into the namespace
    unexpected = sorted(
        p.name for p in root.iterdir() if p.name not in era5.EXPECTED_FILENAMES
    ) if root.is_dir() else []
    report.ok("A24_namespace_contains_only_expected_files", not unexpected,
              list(era5.EXPECTED_FILENAMES), unexpected, str(root))
    return report


# =============================================================================
# CLI
# =============================================================================
def render(report: Report, mode: str) -> str:
    lines = [f"ERA5-Land regional diagnostic validator -- mode: {mode}", ""]
    for check in report.checks:
        lines.append(f"[{check['status']:<7}] {check['check_id']}")
        if check["status"] == FAIL:
            lines.append(f"           expected: {check['expected']!r}")
            lines.append(f"           observed: {check['observed']!r}")
        if check["note"]:
            lines.append(f"           note: {check['note']}")
    passed = sum(1 for c in report.checks if c["status"] == PASS)
    lines += [
        "",
        f"{passed}/{len(report.checks)} checks passed; "
        f"{len(report.failed)} failed, {len(report.skipped)} skipped.",
        "OVERALL: " + ("FAIL" if report.failed else "PASS"),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the ERA5-Land AOI-level regional meteorology diagnostic.",
    )
    parser.add_argument("--mode", choices=("dry-run", "actual"), default="dry-run")
    parser.add_argument("--analysis-id", default=None,
                        help="Required for --mode actual.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="dry-run only: validate a non-default cohort.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the raw check records.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None

    if args.mode == "dry-run":
        report = validate_dry_run(output_root, args.experiments)
    else:
        if not args.analysis_id:
            print("--analysis-id is required for --mode actual", file=sys.stderr)
            return 2
        report = validate_actual(args.analysis_id, output_root)

    if args.json:
        print(json.dumps(report.checks, indent=2, default=str))
    else:
        print(render(report, args.mode))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
