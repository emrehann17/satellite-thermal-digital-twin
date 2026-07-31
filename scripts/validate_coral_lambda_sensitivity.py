#!/usr/bin/env python3
"""Independent structural validator for coral_lambda_sensitivity.v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.coral_lambda_sensitivity import (
    CANONICAL_LAMBDA_INDEX, CANONICAL_STEP8A_SHA256, DIAGNOSTIC_CLASS, DIRECTIONS,
    EXPECTED_SCIENTIFIC_FITS, PRIMARY_EXPERIMENTS,
    LAMBDA_GRID, LAMBDA_TOKEN_SEQUENCE, METRICS, MODEL_FAMILIES, SCHEMA_VERSION,
    compute_analysis_id, scientific_config, sha256_file,
)


FORBIDDEN_RESULT_WORDING = ("statistically significant", "p-value", "p_value", "pvalue",
                            "proven", "optimal", "best lambda")
EXCLUDED_RESULT_EXPERIMENTS = ("evia_2021", "evia_2021_extended")
SCIENTIFIC_TABLE_COLUMNS = {
    "metrics.csv": ("direction", "source_experiment", "target_experiment"),
    "canonical_reproduction.csv": ("direction", "source_experiment", "target_experiment"),
    "bootstrap_summary.csv": ("direction", "source_experiment", "target_experiment"),
    "sensitivity_summary.csv": ("direction", "source_experiment", "target_experiment"),
    "numerical_diagnostics.csv": ("direction", "source_experiment", "target_experiment"),
    "adaptation_statistics.parquet": ("direction", "source_experiment", "target_experiment"),
    "predictions.parquet": ("direction", "source_experiment", "target_experiment"),
    "bootstrap_replicates.parquet": ("direction", "source_experiment", "target_experiment"),
}


def _check(check_id: str, passed: bool, expected: Any, observed: Any, note: str = "") -> dict[str, Any]:
    return {"check_id": check_id, "status": "PASS" if passed else "FAIL", "expected": expected,
            "observed": observed, "note": note}


def _contains_excluded_result_token(value: Any) -> bool:
    text = str(value).lower()
    return any(token in text for token in EXCLUDED_RESULT_EXPERIMENTS)


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.dataset as ds
    return list(ds.dataset(path, format="parquet", partitioning="hive").schema.names)


def check_no_evia_result(root: Path, scientific_config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check scientific participation structurally, not by global text scan."""
    violations: list[dict[str, Any]] = []
    for field in ("experiments", "included_experiments"):
        for value in scientific_config.get(field, ()) or ():
            if _contains_excluded_result_token(value):
                violations.append({"location": f"config.{field}", "value": value})
    directions = scientific_config.get("directions", ()) or ()
    if isinstance(directions, dict):
        directions = [value for values in directions.values() for value in (values if isinstance(values, list) else [values])]
    for value in directions:
        if _contains_excluded_result_token(value):
            violations.append({"location": "config.directions", "value": value})

    for relative, candidate_columns in SCIENTIFIC_TABLE_COLUMNS.items():
        path = root / relative
        if not path.exists():
            continue
        try:
            if path.suffix == ".csv":
                header = pd.read_csv(path, nrows=0).columns
                columns = [column for column in candidate_columns if column in header]
                frame = pd.read_csv(path, usecols=columns) if columns else pd.DataFrame()
            else:
                available = _parquet_columns(path)
                columns = [column for column in candidate_columns if column in available]
                frame = pd.read_parquet(path, columns=columns) if columns else pd.DataFrame()
        except Exception as exc:
            violations.append({"location": relative, "error": f"unreadable:{type(exc).__name__}"})
            continue
        for column in columns:
            bad = frame[column].dropna().astype(str).map(_contains_excluded_result_token)
            if bad.any():
                violations.append({"location": relative, "column": column,
                                   "values": sorted(frame.loc[bad, column].astype(str).unique().tolist())})
        # Inspect Hive partition keys without scanning arbitrary prose/content.
        if path.is_dir():
            for part in path.rglob("part.parquet"):
                for component in part.relative_to(path).parts[:-1]:
                    if "=" in component:
                        key, value = component.split("=", 1)
                        if key in candidate_columns and _contains_excluded_result_token(value):
                            violations.append({"location": relative, "partition_key": key, "value": value})
    return not violations, {"excluded_tokens": list(EXCLUDED_RESULT_EXPERIMENTS), "violations": violations}


def validate(root: Path, deep: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        config = json.loads((root / "config.json").read_text()); sci = config["scientific_config"]
    except Exception as exc:
        checks.append(_check("config_readable", False, "readable", type(exc).__name__))
        return {"status": "FAIL", "checks": checks}
    checks += [
        _check("schema_version", config.get("schema_version") == SCHEMA_VERSION, SCHEMA_VERSION, config.get("schema_version")),
        _check("diagnostic_class", config.get("diagnostic_class") == DIAGNOSTIC_CLASS, DIAGNOSTIC_CLASS, config.get("diagnostic_class")),
        _check("analysis_id_deterministic", compute_analysis_id(sci) == root.name, root.name, compute_analysis_id(sci)),
        _check("four_directions_exact", tuple(sci.get("directions", ())) == DIRECTIONS, DIRECTIONS, sci.get("directions")),
        _check("two_families_exact", tuple(sci.get("model_families", ())) == MODEL_FAMILIES, MODEL_FAMILIES, sci.get("model_families")),
        _check("lambda_values_exact", tuple(sci.get("lambda_grid", ())) == LAMBDA_GRID, LAMBDA_GRID, sci.get("lambda_grid")),
        _check("lambda_tokens_exact", tuple(sci.get("lambda_tokens", ())) == LAMBDA_TOKEN_SEQUENCE, LAMBDA_TOKEN_SEQUENCE, sci.get("lambda_tokens")),
        _check("canonical_lambda_index", sci.get("canonical_lambda_index") == CANONICAL_LAMBDA_INDEX, 4, sci.get("canonical_lambda_index")),
        _check("fit_ceiling", sci.get("expected_scientific_fits") == EXPECTED_SCIENTIFIC_FITS, 72, sci.get("expected_scientific_fits")),
        _check("no_gee", sci.get("earth_engine_used", False) is False, False, sci.get("earth_engine_used", False)),
    ]
    grid_path = root / "lambda_grid.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        checks.append(_check("grid_rows", len(grid) == 9, 9, len(grid)))
        checks.append(_check("lambda_zero_exact", float(grid.iloc[0]["lambda_value"]) == 0.0, 0.0, grid.iloc[0]["lambda_value"]))
    result_files = [p for p in root.rglob("*") if p.is_file() and p.name != "validation_report.json"]
    combined = "\n".join(p.read_text(errors="ignore") for p in result_files if p.suffix in {".json", ".csv", ".md"}).lower()
    checks.append(_check("no_forbidden_wording", not any(w in combined for w in FORBIDDEN_RESULT_WORDING), "no hits", "hit" if any(w in combined for w in FORBIDDEN_RESULT_WORDING) else "no hits"))
    no_evia, evia_evidence = check_no_evia_result(root, sci)
    checks.append(_check("no_evia_result", no_evia,
                         "excluded AOIs absent from included scope and scientific result rows",
                         evia_evidence,
                         "Exclusion provenance and human-readable exclusion statements are allowed."))
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()); drift = []
        for item in manifest.get("files", []):
            if sha256_file(root / item["path"]) != item["sha256"]: drift.append(item["path"])
        checks.append(_check("no_hash_drift", not drift, [], drift))
    return {"schema_version": SCHEMA_VERSION, "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "deep": deep, "checks": checks}


def validate_dry_run(root: Path) -> dict[str, Any]:
    """Validate the frozen contract without reading or creating run artifacts."""
    sci = scientific_config()
    expected_id = compute_analysis_id(sci)
    experiments = tuple(sci.get("experiments", ()))
    directions = tuple(sci.get("directions", ()))
    checks = [
        _check("schema_version", sci.get("schema_version") == SCHEMA_VERSION,
               SCHEMA_VERSION, sci.get("schema_version")),
        _check("diagnostic_class", sci.get("diagnostic_class") == DIAGNOSTIC_CLASS,
               DIAGNOSTIC_CLASS, sci.get("diagnostic_class")),
        _check("analysis_id_deterministic", root.name == expected_id, expected_id, root.name),
        _check("four_directions_exact", directions == DIRECTIONS, DIRECTIONS, directions),
        _check("two_families_exact", tuple(sci.get("model_families", ())) == MODEL_FAMILIES,
               MODEL_FAMILIES, sci.get("model_families")),
        _check("nine_lambda_values_exact", tuple(sci.get("lambda_grid", ())) == LAMBDA_GRID,
               LAMBDA_GRID, sci.get("lambda_grid")),
        _check("lambda_tokens_exact", tuple(sci.get("lambda_tokens", ())) == LAMBDA_TOKEN_SEQUENCE,
               LAMBDA_TOKEN_SEQUENCE, sci.get("lambda_tokens")),
        _check("canonical_lambda_index", sci.get("canonical_lambda_index") == CANONICAL_LAMBDA_INDEX,
               4, sci.get("canonical_lambda_index")),
        _check("expected_scientific_fits", sci.get("expected_scientific_fits") == EXPECTED_SCIENTIFIC_FITS,
               72, sci.get("expected_scientific_fits")),
        _check("excluded_aoi_absent", all("evia" not in value.lower() for value in experiments + directions),
               "absent from experiments and directions", {"experiments": experiments, "directions": directions}),
        _check("earth_engine_disabled", sci.get("earth_engine_used", False) is False,
               False, sci.get("earth_engine_used", False)),
        _check("canonical_step8a_expectations_registered",
               set(CANONICAL_STEP8A_SHA256) == set(PRIMARY_EXPERIMENTS)
               and all(len(value) == 64 for value in CANONICAL_STEP8A_SHA256.values()),
               {name: "64-character sha256" for name in PRIMARY_EXPERIMENTS},
               CANONICAL_STEP8A_SHA256),
        _check("dry_run_root_absent", not root.exists(), False, root.exists(),
               "Dry-run does not create the analysis namespace."),
    ]
    return {"schema_version": SCHEMA_VERSION,
            "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
            "mode": "dry_run", "analysis_root": str(root), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("analysis_root", type=Path)
    parser.add_argument("--deep", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = validate_dry_run(args.analysis_root) if args.dry_run else validate(args.analysis_root, deep=args.deep)
    if not args.dry_run and args.analysis_root.is_dir():
        (args.analysis_root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"OVERALL STATUS: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
