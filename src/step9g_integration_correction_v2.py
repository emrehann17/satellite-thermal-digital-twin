"""
step9g_integration_correction_v2.py

Step9G REPORT-INTEGRATION CORRECTION (v2).

WHAT THIS FIXES
---------------
The original Step9G integration (analysis_id
87d4ec94021fed17446ce5e7870d4222154a131ce830535339a4f0d01c0c6ba7) assumed
that each transfer direction (A->B and B->A) lives in its OWN directory
(e.g. `cross_region/bejis_2022__manavgat_2021/step9f/`). It does not. The
repository stores BOTH logical directions inside ONE shared pair-level
directory `cross_region/manavgat_2021__bejis_2022/<stage>/`:

  - Step9E: pair-global `relationship_direction_flips.csv` carrying
    `source_experiment_id` / `target_experiment_id` columns (a single
    symmetric artifact, NOT one file per direction).
  - Step9F: `exploratory_candidate_screening.csv` with per-direction columns
    prefixed `{source}_to_{target}__...`, and `by_direction`-style grouped
    bootstrap JSON. Model/representation level only.
  - Step10: a single combined `step10_final_report.json` keyed `by_direction`
    with both `manavgat_2021_to_bejis_2022` and `bejis_2022_to_manavgat_2021`.

Because v1 looked for a nonexistent reverse-direction directory, it wrongly
reported the reverse direction as `unavailable` and recorded zero frozen
reference files for it.

WHAT THIS DOES NOT DO
---------------------
It recomputes NO Step9G quantity: not a univariate AUC, bootstrap replicate,
confidence interval, landcover row, or reversal classification. The frozen
Step9G numeric outputs are read verbatim, hash-protected, and reused. Only the
INTEGRATION section (which frozen 9E/9F/10 artifacts are surfaced, and how) is
corrected, plus two reporting defects:
  (6) `thermal_features_consistent_with_step9e` wrongly included
      `elevation_mean` (a baseline, not a thermal, feature). Replaced with a
      correct split (`features_consistent_with_step9e` +
      `baseline_features_consistent_with_step9e` /
      `thermal_features_consistent_with_step9e`).
  (7) relationship-direction flags are emitted as real JSON booleans, never
      the strings "True"/"False".

The corrected report is written under a SEPARATE correction namespace; the
original Step9G outputs remain immutable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from src.step8_large_block_robustness import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    _git_commit,
    _package_versions,
)
import src.step9g_univariate_feature_auc_direction_reversal as step9g

log, log_file = setup_logger("step9g_integration_correction_v2")

SCHEMA_VERSION = "step9g.integration_correction.v2"

SOURCE_ID = step9g.SOURCE_ID  # manavgat_2021
TARGET_ID = step9g.TARGET_ID  # bejis_2022
PAIR_TOKEN = step9g.PAIR_TOKEN  # manavgat_2021__bejis_2022

# Both LOGICAL directions live inside the single pair directory.
FORWARD_DIRECTION = f"{SOURCE_ID}_to_{TARGET_ID}"
REVERSE_DIRECTION = f"{TARGET_ID}_to_{SOURCE_ID}"
LOGICAL_DIRECTIONS = (FORWARD_DIRECTION, REVERSE_DIRECTION)

NUMERIC_FEATURES = step9g.NUMERIC_FEATURES

# Baseline vs thermal split (matches Step9A shared feature sets:
# baseline = ndvi/elevation/slope [+landcover, excluded here], thermal = LST/TVDI).
BASELINE_NUMERIC_FEATURES = ("ndvi_mean", "elevation_mean", "slope_mean")
THERMAL_NUMERIC_FEATURES = (
    "lst_anomaly_mean", "current_lst_mean", "current_tvdi_mean",
    "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean",
)

EXPECTED_FROZEN_STEP9G_ANALYSIS_ID = (
    "87d4ec94021fed17446ce5e7870d4222154a131ce830535339a4f0d01c0c6ba7"
)

# The pair directory holding all shared cross-region stage artifacts.
def _pair_dir(source_id: str = SOURCE_ID, target_id: str = TARGET_ID) -> Path:
    return PROJECT_ROOT / "outputs" / "cross_region" / f"{source_id}__{target_id}"


def _step9g_frozen_root(source_id: str = SOURCE_ID, target_id: str = TARGET_ID) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "diagnostics"
        / "step9g_univariate_feature_auc_direction_reversal" / f"{source_id}__{target_id}"
    )


def _output_root(source_id: str = SOURCE_ID, target_id: str = TARGET_ID) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "diagnostics"
        / "step9g_univariate_feature_auc_direction_reversal_integration_v2" / f"{source_id}__{target_id}"
    )


# Module-level constants recomputed lazily so tests can monkeypatch PROJECT_ROOT.
STEP9G_FROZEN_ROOT = _step9g_frozen_root()
OUTPUT_ROOT = _output_root()


class Step9GIntegrationError(SystemExit):
    """Fail-fast error for the Step9G integration correction."""


# =============================================================================
# Frozen Step9G numeric outputs -- read-only, hash-protected, never recomputed
# =============================================================================
FROZEN_STEP9G_NUMERIC_FILES = (
    "step9g_univariate_auc_by_region.csv",
    "step9g_direction_reversal_table.csv",
    "step9g_bootstrap_replicates.parquet",
    "step9g_landcover_descriptive.csv",
    "step9g_final_report.json",
    "step9g_preregistration.json",
)


def frozen_step9g_root(source_id: str = SOURCE_ID, target_id: str = TARGET_ID) -> Path:
    return _step9g_frozen_root(source_id, target_id)


def hash_frozen_step9g(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, str]:
    root = frozen_step9g_root(source_id, target_id)
    if not root.exists():
        raise Step9GIntegrationError(f"Frozen Step9G outputs not found: {root}")
    hashes = {}
    for name in FROZEN_STEP9G_NUMERIC_FILES:
        path = root / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    if "step9g_direction_reversal_table.csv" not in hashes:
        raise Step9GIntegrationError(
            "Frozen Step9G direction-reversal table is missing; cannot correct integration."
        )
    return hashes


def assert_frozen_step9g_analysis_id(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> str:
    root = frozen_step9g_root(source_id, target_id)
    report = json.loads((root / "step9g_final_report.json").read_text(encoding="utf-8"))
    analysis_id = report.get("analysis_id")
    prereg = _read_json(root / "step9g_preregistration.json") or {}
    expected = prereg.get("analysis_id")
    if (source_id, target_id) == (SOURCE_ID, TARGET_ID):
        expected = EXPECTED_FROZEN_STEP9G_ANALYSIS_ID
    if not expected or analysis_id != expected:
        raise Step9GIntegrationError(
            "Frozen Step9G analysis_id does not match the protected value. "
            f"Expected {expected}, found {analysis_id}. "
            "Refusing to proceed."
        )
    return analysis_id


def load_frozen_reversal_table(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> pd.DataFrame:
    """Reads the frozen Step9G direction-reversal table VERBATIM. No AUC, CI,
    reversal status, or feature ordering is altered."""
    return pd.read_csv(frozen_step9g_root(source_id, target_id) / "step9g_direction_reversal_table.csv")


# =============================================================================
# Correctly parse the SHARED pair-level frozen 9E / 9F / 10 artifacts
# =============================================================================
def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _as_bool_or_none(value: Any) -> bool | None:
    """Coerce a value to a real Python bool, or None. Handles native bool,
    numpy bool_ (what pandas returns from a boolean CSV column), numeric 0/1,
    and the string 'True'/'False'/'nan' forms that CSV round-trips produce.
    Never returns a string boolean and never returns a numpy scalar."""
    if value is None:
        return None
    # numpy bool_ is NOT a Python bool and NOT an int/float, so it must be
    # handled explicitly before the generic checks below.
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "1.0"):
            return True
        if s in ("false", "0", "0.0"):
            return False
        if s in ("", "nan", "none", "null"):
            return None
    return None


def parse_step9e(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """
    Step9E stores relationship-direction diagnostics PAIR-GLOBALLY in one
    shared `relationship_direction_flips.csv`. Each row carries
    source_experiment_id / target_experiment_id and symmetric per-feature
    flip flags for the pair. There is NO per-direction directory; this schema
    is represented as `pair_global`, not duplicated into two direction records.
    """
    path = _pair_dir(source_id, target_id) / "step9e" / "relationship_direction_flips.csv"
    df = _read_csv(path)
    result: dict[str, Any] = {
        "artifact": str(path),
        "available": df is not None,
        "schema": "pair_global",
        "per_feature": {},
    }
    if df is None:
        return result
    prim = df
    if "population" in df.columns:
        prim = df[df["population"] == step9g.PRIMARY_POPULATION]
    for feature in NUMERIC_FEATURES:
        sub = prim[prim["feature"] == feature] if "feature" in prim.columns else prim.iloc[0:0]
        if sub.empty:
            result["per_feature"][feature] = {"available": False}
            continue
        row = sub.iloc[0]
        result["per_feature"][feature] = {
            "available": True,
            "mean_direction_flip": _as_bool_or_none(row.get("mean_direction_flip")),
            "median_direction_flip": _as_bool_or_none(row.get("median_direction_flip")),
            "rank_effect_direction_flip": _as_bool_or_none(row.get("rank_effect_direction_flip")),
            "raw_auc_below_0_5_in_one_region_only": _as_bool_or_none(row.get("raw_auc_below_0_5_in_one_region_only")),
            "relationship_flip_score": (
                None if pd.isna(row.get("relationship_flip_score")) else int(row.get("relationship_flip_score"))
            ) if "relationship_flip_score" in sub.columns else None,
        }
    return result


def parse_step9f(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """
    Step9F stores BOTH directions inside one shared pair-level directory.
    `exploratory_candidate_screening.csv` carries per-direction columns
    prefixed `{direction}__...`; the bootstrap JSON groups rows by
    `transfer_direction`. This is MODEL/representation-level only; it is never
    joined per univariate feature.
    """
    pair_dir = _pair_dir(source_id, target_id)
    screening = _read_csv(pair_dir / "step9f" / "exploratory_candidate_screening.csv")
    manifest = _read_json(pair_dir / "step9f" / "step9f_experiment_manifest.json")
    boot = _read_json(pair_dir / "step9f" / "spatial_block_bootstrap_deltas.json")

    result: dict[str, Any] = {
        "level": "model_representation_level_only",
        "note": (
            "Step9F reports model/representation-level ranking reversal across "
            "feature variants for both transfer directions in one shared "
            "artifact. It is never fabricated into per-feature univariate joins."
        ),
        "screening_available": screening is not None,
        "manifest_analysis_id": (manifest or {}).get("analysis_id") if manifest else None,
        "directions": {},
    }
    boot_groups = (boot or {}).get("groups", []) if isinstance(boot, dict) else []
    logical_directions = (
        f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}",
    )
    for direction in logical_directions:
        entry: dict[str, Any] = {"available": False}
        if screening is not None:
            dir_cols = [c for c in screening.columns if c.startswith(f"{direction}__")]
            if dir_cols:
                reversal_col = f"{direction}__ranking_reversal_suspected"
                entry = {
                    "available": True,
                    "screening_column_count": len(dir_cols),
                    "any_ranking_reversal_suspected": (
                        bool(screening[reversal_col].astype("boolean").fillna(False).any())
                        if reversal_col in screening.columns else None
                    ),
                }
        if boot_groups:
            dir_groups = [g for g in boot_groups if g.get("transfer_direction") == direction]
            if dir_groups:
                entry["available"] = True
                entry["bootstrap_group_count"] = len(dir_groups)
        result["directions"][direction] = entry
    return result


def parse_step10(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    """
    Step10 stores a SINGLE combined final report keyed `by_direction`, holding
    both directions. Confirm both from that combined report + associated
    tables (target_performance / adaptation_effect rows also carry a
    `direction` field).
    """
    pair_dir = _pair_dir(source_id, target_id)
    report = _read_json(pair_dir / "step10" / "step10_final_report.json")
    result: dict[str, Any] = {
        "artifact": str(pair_dir / "step10" / "step10_final_report.json"),
        "available": report is not None,
        "analysis_id": (report or {}).get("analysis_id") if report else None,
        "directions": {},
    }
    if report is None:
        return result

    logical_directions = (
        f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}",
    )

    # Directions can be confirmed from any of: target_performance rows,
    # within_transfer_decomposition, or the integrated interpretation block.
    def _directions_in(section: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(section, list):
            for row in section:
                if isinstance(row, dict) and "direction" in row:
                    found.add(row["direction"])
        elif isinstance(section, dict):
            found |= {k for k in section.keys() if k in logical_directions}
        return found

    confirmed: set[str] = set()
    for key in ("target_performance", "adaptation_effect", "within_transfer_decomposition", "integrated_interpretation"):
        confirmed |= _directions_in(report.get(key))

    for direction in logical_directions:
        rows = [
            r for r in (report.get("target_performance") or [])
            if isinstance(r, dict) and r.get("direction") == direction
        ]
        result["directions"][direction] = {
            "available": direction in confirmed,
            "target_performance_row_count": len(rows),
        }
    return result


# =============================================================================
# Corrected feature-level integration (Step9E per-feature; 9F/10 model-level)
# =============================================================================
def build_corrected_integration(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    reversal_df = load_frozen_reversal_table(source_id, target_id)
    step9e = parse_step9e(source_id, target_id)
    step9f = parse_step9f(source_id, target_id)
    step10 = parse_step10(source_id, target_id)
    source_key = step9g._region_output_key(source_id, source_id, target_id)
    target_key = step9g._region_output_key(target_id, source_id, target_id)

    # Which features have a Step9E relationship-direction flag AND a Step9G
    # point reversal -> "consistent with Step9E". Real booleans throughout.
    consistent_features: list[str] = []
    per_feature_rows: list[dict[str, Any]] = []
    for _, r in reversal_df.iterrows():
        feature = r["feature"]
        e = step9e["per_feature"].get(feature, {"available": False})
        e_flag = e.get("rank_effect_direction_flip") if e.get("available") else None
        point_reversal = _as_bool_or_none(r.get("point_direction_reversal"))
        consistent = bool(e_flag) and bool(point_reversal)
        if consistent:
            consistent_features.append(feature)
        per_feature_rows.append({
            "feature": feature,
            "source_experiment_id": source_id,
            "target_experiment_id": target_id,
            "source_auc": r.get(f"{source_key}_auc"),
            "source_ci_low": r.get(f"{source_key}_ci_low"),
            "source_ci_high": r.get(f"{source_key}_ci_high"),
            "source_direction": r.get(f"{source_key}_direction"),
            "target_auc": r.get(f"{target_key}_auc"),
            "target_ci_low": r.get(f"{target_key}_ci_low"),
            "target_ci_high": r.get(f"{target_key}_ci_high"),
            "target_direction": r.get(f"{target_key}_direction"),
            "reversal_status": r.get("reversal_status"),
            "point_direction_reversal": point_reversal,  # real bool
            "step9e_rank_effect_direction_flip": e_flag,  # real bool or None
            "step9e_consistent": consistent,  # real bool
        })

    baseline_consistent = [f for f in consistent_features if f in BASELINE_NUMERIC_FEATURES]
    thermal_consistent = [f for f in consistent_features if f in THERMAL_NUMERIC_FEATURES]

    supported = [
        r["feature"] for r in per_feature_rows
        if r["reversal_status"] == "bootstrap_supported_direction_reversal"
    ]
    uncertain = [
        r["feature"] for r in per_feature_rows
        if r["reversal_status"] == "point_direction_reversal_interval_uncertain"
    ]

    return {
        "per_feature": per_feature_rows,
        "step9e": step9e,
        "step9f": step9f,
        "step10": step10,
        "features_consistent_with_step9e": consistent_features,
        "baseline_features_consistent_with_step9e": baseline_consistent,
        "thermal_features_consistent_with_step9e": thermal_consistent,  # NO elevation
        "bootstrap_supported_direction_reversals": supported,
        "point_reversals_interval_uncertain": uncertain,
    }


# =============================================================================
# Availability before/after table
# =============================================================================
def availability_table(
    corrected: dict[str, Any], source_id: str = SOURCE_ID,
    target_id: str = TARGET_ID,
) -> list[dict[str, Any]]:
    """Before = v1 (reverse direction wrongly unavailable / zero files).
    After = v2 corrected parsing of the shared pair-level artifacts."""
    rows = []
    step9f_dirs = corrected["step9f"]["directions"]
    step10_dirs = corrected["step10"]["directions"]
    rows.append({
        "stage": "step9e", "direction": "pair_global",
        "before_v1": "forward_only (reverse dir assumed missing)",
        "after_v2": "available" if corrected["step9e"]["available"] else "unavailable",
        "schema": "pair_global",
    })
    logical_directions = (
        f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}",
    )
    reverse_direction = logical_directions[1]
    for direction in logical_directions:
        is_reverse = direction == reverse_direction
        rows.append({
            "stage": "step9f", "direction": direction,
            "before_v1": "unavailable" if is_reverse else "available",
            "after_v2": "available" if step9f_dirs.get(direction, {}).get("available") else "unavailable",
            "schema": "shared_pair_directory",
        })
    for direction in logical_directions:
        is_reverse = direction == reverse_direction
        rows.append({
            "stage": "step10", "direction": direction,
            "before_v1": "unavailable" if is_reverse else "available",
            "after_v2": "available" if step10_dirs.get(direction, {}).get("available") else "unavailable",
            "schema": "combined_report_by_direction",
        })
    return rows


# =============================================================================
# Analysis id / manifest for the correction
# =============================================================================
def used_reference_hashes(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, str]:
    """Hashes of the ACTUALLY-USED shared 9E/9F/10 artifacts."""
    pair_dir = _pair_dir(source_id, target_id)
    candidates = {
        "step9e_relationship_direction_flips.csv": pair_dir / "step9e" / "relationship_direction_flips.csv",
        "step9f_exploratory_candidate_screening.csv": pair_dir / "step9f" / "exploratory_candidate_screening.csv",
        "step9f_spatial_block_bootstrap_deltas.json": pair_dir / "step9f" / "spatial_block_bootstrap_deltas.json",
        "step9f_experiment_manifest.json": pair_dir / "step9f" / "step9f_experiment_manifest.json",
        "step10_final_report.json": pair_dir / "step10" / "step10_final_report.json",
    }
    return {name: sha256_file(path) for name, path in candidates.items() if path.is_file()}


def correction_configuration(
    frozen_step9g_hashes: dict[str, str], reference_hashes: dict[str, str],
    corrected_analysis_id: str = EXPECTED_FROZEN_STEP9G_ANALYSIS_ID,
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    logical_directions = (
        f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "corrects_analysis_id": corrected_analysis_id,
        "pair_token": f"{source_id}__{target_id}",
        "logical_directions": list(logical_directions),
        "recomputes_step9g_numeric": False,
        "frozen_step9g_numeric_hashes": frozen_step9g_hashes,
        "used_reference_hashes": reference_hashes,
        "schema_findings": {
            "step9e": "pair_global relationship_direction_flips.csv (source/target columns)",
            "step9f": "shared pair directory; per-direction columns '{direction}__...'; model-level only",
            "step10": "single combined final report keyed by_direction",
        },
        "corrections_applied": [
            "reverse direction parsed from shared pair-level artifacts (no per-direction directory required)",
            "thermal_features_consistent_with_step9e no longer includes baseline elevation_mean",
            "relationship-direction flags emitted as real JSON booleans, not string booleans",
        ],
        "baseline_numeric_features": list(BASELINE_NUMERIC_FEATURES),
        "thermal_numeric_features": list(THERMAL_NUMERIC_FEATURES),
        "output_namespace": str(_output_root(source_id, target_id)),
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
    }


def build_manifest(
    frozen_step9g_hashes: dict[str, str], reference_hashes: dict[str, str],
    corrected_analysis_id: str = EXPECTED_FROZEN_STEP9G_ANALYSIS_ID,
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
) -> dict[str, Any]:
    config = correction_configuration(
        frozen_step9g_hashes, reference_hashes, corrected_analysis_id,
        source_id, target_id,
    )
    analysis_id = sha256_bytes(canonical_json(config).encode("utf-8"))
    return {"analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "correction_configuration": config}


# =============================================================================
# Claim boundaries (preserved verbatim)
# =============================================================================
CLAIM_BOUNDARIES = [
    "Feature-level univariate AUC direction reversals are consistent with residual concept/relationship shift.",
    "This is not causal proof.",
    "This does not prove that concept shift is the only source of transfer failure.",
]

REQUIRED_UNCERTAIN_FEATURES = (
    "current_lst_mean", "tvdi_difference_mean", "downscaled_lst_mean", "fused_lst_mean",
)
REQUIRED_SUPPORTED_FEATURE = "elevation_mean"


def _assert_required_statements(corrected: dict[str, Any]) -> None:
    """Enforce requirement (10)/(11): the frozen numeric result must show
    exactly one supported reversal (elevation_mean) and four uncertain LST/TVDI
    point reversals -- and the correction must NOT relabel the uncertain ones
    as supported. If the frozen table disagrees, fail loudly rather than
    silently emitting a wrong report."""
    supported = set(corrected["bootstrap_supported_direction_reversals"])
    uncertain = set(corrected["point_reversals_interval_uncertain"])
    if supported != {REQUIRED_SUPPORTED_FEATURE}:
        raise Step9GIntegrationError(
            f"Frozen Step9G supported reversals {sorted(supported)} != "
            f"expected {{{REQUIRED_SUPPORTED_FEATURE}}}. Integration correction "
            "will not fabricate or drop supported reversals."
        )
    if not set(REQUIRED_UNCERTAIN_FEATURES).issubset(uncertain):
        raise Step9GIntegrationError(
            f"Frozen Step9G uncertain reversals {sorted(uncertain)} do not "
            f"include the expected {sorted(REQUIRED_UNCERTAIN_FEATURES)}."
        )
    # requirement (11): none of the uncertain features may be in supported.
    if uncertain & supported:
        raise Step9GIntegrationError("An uncertain LST/TVDI reversal is also marked supported; refusing.")


# =============================================================================
# Report writing (new namespace only; original Step9G untouched)
# =============================================================================
def _assert_namespace(path: Path, output_root: Path = OUTPUT_ROOT) -> None:
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise Step9GIntegrationError(f"Namespace isolation FAILED: '{path}' is outside '{output_root}'.")


def write_reports(
    output_root: Path, manifest: dict[str, Any], corrected: dict[str, Any],
    avail: list[dict[str, Any]], source_id: str = SOURCE_ID,
    target_id: str = TARGET_ID,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_id = manifest["analysis_id"]

    def _w(name: str, writer) -> Path:
        path = output_root / name
        _assert_namespace(path, output_root)
        writer(path)
        return path

    paths: dict[str, Path] = {}
    paths["manifest"] = _w(
        "step9g_integration_correction_manifest.json",
        lambda p: p.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"),
    )
    paths["availability"] = _w(
        "step9g_integration_availability_before_after.csv",
        lambda p: pd.DataFrame(avail).to_csv(p, index=False),
    )

    report = {
        "analysis_id": analysis_id,
        "schema_version": SCHEMA_VERSION,
        "corrects_analysis_id": manifest["correction_configuration"]["corrects_analysis_id"],
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "frozen_step9g_numeric_reused_verbatim": True,
        "primary_population": step9g.PRIMARY_POPULATION,
        "bootstrap_supported_direction_reversal": corrected["bootstrap_supported_direction_reversals"],
        "point_reversals_interval_uncertain": corrected["point_reversals_interval_uncertain"],
        "features_consistent_with_step9e": corrected["features_consistent_with_step9e"],
        "baseline_features_consistent_with_step9e": corrected["baseline_features_consistent_with_step9e"],
        "thermal_features_consistent_with_step9e": corrected["thermal_features_consistent_with_step9e"],
        "step9e_integration": corrected["step9e"],
        "step9f_model_level_integration": corrected["step9f"],
        "step10_transfer_integration": corrected["step10"],
        "per_feature_integration": corrected["per_feature"],
        "availability_before_after": avail,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    paths["final_report_json"] = _w(
        "step9g_integration_correction_final_report.json",
        lambda p: p.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"),
    )
    paths["per_feature_csv"] = _w(
        "step9g_integration_correction_per_feature.csv",
        lambda p: pd.DataFrame(corrected["per_feature"]).to_csv(p, index=False),
    )
    paths["final_report_md"] = _w(
        "step9g_integration_correction_final_report.md",
        lambda p: p.write_text(_report_md(report), encoding="utf-8"),
    )
    return paths


def _report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Step9G Integration Correction (v2) -- Final Report",
        "",
        f"- correction analysis_id: `{report['analysis_id']}`",
        f"- corrects Step9G analysis_id: `{report['corrects_analysis_id']}`",
        f"- source: `{report['source_experiment_id']}`",
        f"- target: `{report['target_experiment_id']}`",
        "- frozen Step9G numeric outputs reused verbatim (nothing recomputed)",
        f"- primary population: {report['primary_population']}",
        "",
        "## Direction-reversal findings (from frozen Step9G numeric outputs)",
        "",
        f"- Bootstrap-supported reversals: **{', '.join(report['bootstrap_supported_direction_reversal']) or 'none'}**",
        f"- Point reversals with intervals including chance: "
        f"**{', '.join(report['point_reversals_interval_uncertain'])}**",
        "- The uncertain LST/TVDI reversals are NOT described as bootstrap-supported.",
        "",
        "## Step9E agreement",
        "",
        f"- Features consistent with Step9E relationship-direction diagnostics: "
        f"{report['features_consistent_with_step9e']}",
        f"  - baseline: {report['baseline_features_consistent_with_step9e']}",
        f"  - thermal: {report['thermal_features_consistent_with_step9e']} (elevation is baseline, excluded here)",
        "",
        "## Step9F (model/representation level, both directions)",
        "",
    ]
    for direction, entry in report["step9f_model_level_integration"]["directions"].items():
        lines.append(f"- {direction}: available={entry.get('available')}")
    lines += ["", "## Step10 (raw/adapted transfer, both directions)", ""]
    for direction, entry in report["step10_transfer_integration"]["directions"].items():
        lines.append(f"- {direction}: available={entry.get('available')}")
    lines += ["", "## Availability before/after", "",
              "| stage | direction | before (v1) | after (v2) | schema |",
              "| --- | --- | --- | --- | --- |"]
    for r in report["availability_before_after"]:
        lines.append(f"| {r['stage']} | {r['direction']} | {r['before_v1']} | {r['after_v2']} | {r['schema']} |")
    lines += ["", "## Claim boundaries", ""] + [f"- {c}" for c in report["claim_boundaries"]]
    return "\n".join(lines) + "\n"


# =============================================================================
# Orchestration
# =============================================================================
def run_correction(
    source_id: str = SOURCE_ID, target_id: str = TARGET_ID,
    dry: bool = False, force: bool = False, output_root: Path | None = None,
) -> dict[str, Any]:
    if source_id == target_id:
        raise Step9GIntegrationError("--source and --target must be different experiment IDs.")
    output_root = _output_root(source_id, target_id) if output_root is None else output_root

    frozen_step9g_hashes_before = hash_frozen_step9g(source_id, target_id)
    corrected_analysis_id = assert_frozen_step9g_analysis_id(source_id, target_id)
    reference_hashes = used_reference_hashes(source_id, target_id)
    logical_directions = (
        f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}",
    )

    if dry:
        return {
            "mode": "dry_run",
            "recomputes_step9g_numeric": False,
            "writes_files": False,
            "frozen_step9g_files": list(frozen_step9g_hashes_before.keys()),
            "used_reference_files": list(reference_hashes.keys()),
            "source_experiment_id": source_id,
            "target_experiment_id": target_id,
            "logical_directions": list(logical_directions),
            "output_namespace": str(output_root),
        }

    corrected = build_corrected_integration(source_id, target_id)
    if (source_id, target_id) == (SOURCE_ID, TARGET_ID):
        _assert_required_statements(corrected)
    avail = availability_table(corrected, source_id, target_id)
    manifest = build_manifest(
        frozen_step9g_hashes_before, reference_hashes, corrected_analysis_id,
        source_id, target_id,
    )

    if output_root.exists() and not force:
        existing = list(p for p in output_root.rglob("*") if p.is_file())
        if existing:
            raise Step9GIntegrationError(
                f"Correction outputs already exist under {output_root}; use --force. "
                f"({[p.name for p in existing]})"
            )
    paths = write_reports(
        output_root, manifest, corrected, avail, source_id, target_id,
    )

    # frozen Step9G numeric outputs must be byte-identical after the run
    frozen_step9g_hashes_after = hash_frozen_step9g(source_id, target_id)
    if frozen_step9g_hashes_before != frozen_step9g_hashes_after:
        raise Step9GIntegrationError("Frozen Step9G numeric outputs changed during the correction run.")

    return {
        "ran": True,
        "analysis_id": manifest["analysis_id"],
        "corrects_analysis_id": corrected_analysis_id,
        "bootstrap_supported_direction_reversal": corrected["bootstrap_supported_direction_reversals"],
        "point_reversals_interval_uncertain": corrected["point_reversals_interval_uncertain"],
        "thermal_features_consistent_with_step9e": corrected["thermal_features_consistent_with_step9e"],
        "features_consistent_with_step9e": corrected["features_consistent_with_step9e"],
        "availability_before_after": avail,
        "frozen_step9g_hash_check": "passed",
        "report_paths": {k: str(v) for k, v in paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step9G report-integration correction (v2).")
    parser.add_argument("--source", required=True, help="Source experiment ID.")
    parser.add_argument("--target", required=True, help="Target experiment ID.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_correction(
        source_id=args.source, target_id=args.target,
        dry=args.dry_run, force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
