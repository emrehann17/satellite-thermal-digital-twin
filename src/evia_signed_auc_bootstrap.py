"""
Evia signed-AUC spatial-block bootstrap (advisor follow-up item 5).

Converts each predictor's RAW univariate ROC-AUC into a signed effect scale and
reports it with spatial-block bootstrap uncertainty, for the canonical Evia
experiment `evia_2021_extended`.

    signed_auc = 2 * raw_auc - 1

The AUC is NEVER inverted: a feature whose high values rank UNBURNED cells
higher keeps raw_auc < 0.5 and therefore receives a NEGATIVE signed_auc. The
sign carries the direction; the magnitude carries the effect size.

Bootstrap contract
------------------
* population        : burnable_tree_shrub_grass (canonical primary population)
* resampling unit   : whole ~5 km spatial blocks (10 x 10 cells of 500 m),
                      sampled WITH replacement, multiplicity preserved
* replicates        : 1000
* seed              : 42
* CI                : 2.5 / 97.5 percentile of the REPLICATE-LEVEL signed
                      distribution
* min successful    : 900
* row bootstrap     : forbidden
* target label      : used ONLY to compute the diagnostic AUC

Why Step9G is reused and Step10C is NOT
---------------------------------------
`src/step9g_univariate_feature_auc_direction_reversal` already implements
precisely this estimand -- univariate raw AUC of a single predictor against the
burned label, resampled over 5 km spatial blocks -- and its constants match the
required contract exactly (BLOCK_SIZE_CELLS=10 -> "approximately_5_km",
BOOTSTRAP_SEED=42, BOOTSTRAP_REPLICATES=1000, CI 2.5/97.5,
MIN_VALID_REPLICATES=900) and it already defines
`signed_rank_effect = 2*raw_univariate_auc - 1`. This module therefore reuses
Step9G's helpers verbatim so the two cannot drift apart.

`src/step10c_paired_evaluation_bootstrap` is deliberately NOT reused. Its
estimand is a MODEL-level paired transfer AUC (baseline/thermal x raw/adapted),
its resampling frame is the aligned source->target prediction join, and its unit
is the Step10 target block id. Borrowing it would silently change the estimand,
so it is rejected despite being available.

Read-only with respect to every frozen artefact. Writes exclusively under
outputs/diagnostics/evia_signed_auc_bootstrap/<experiment_id>/.
"""
from __future__ import annotations

import hashlib
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

from core.paths import PROJECT_ROOT
from src.step9a_audit_cross_region_inputs import TARGET_COLUMN
from src.step9g_univariate_feature_auc_direction_reversal import (
    BLOCK_SIZE_CELLS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CI_LOWER_PCT,
    CI_UPPER_PCT,
    MIN_VALID_REPLICATES,
    NOMINAL_BLOCK_SCALE,
    NUMERIC_FEATURES,
    PRIMARY_POPULATION,
    _block_bootstrap_auc,
    assign_blocks_then_filter,
    load_step8a,
    univariate_feature_stats,
)

SCHEMA_VERSION = "evia_signed_auc_bootstrap.v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "evia_signed_auc_bootstrap"

CANONICAL_EXPERIMENT = "evia_2021_extended"
SENSITIVITY_EXPERIMENT = "evia_2021"

STEP9G_ROOT = (
    PROJECT_ROOT / "outputs" / "diagnostics"
    / "step9g_univariate_feature_auc_direction_reversal"
)

SUPPORT_POSITIVE = "bootstrap_supported_positive_direction"
SUPPORT_NEGATIVE = "bootstrap_supported_negative_direction"
SUPPORT_ZERO = "interval_includes_zero"
SUPPORT_UNSTABLE = "unstable_bootstrap"
SUPPORT_UNAVAILABLE = "unavailable"

# Step9G expresses the identical decision on the RAW scale (thresholds at 0.5);
# this module expresses it on the SIGNED scale (thresholds at 0). Because
# signed = 2*raw - 1 is strictly increasing, raw_ci_low > 0.5 <=> signed_ci_low > 0
# and raw_ci_high < 0.5 <=> signed_ci_high < 0, so the two vocabularies are
# semantically equivalent. This map exists ONLY so the AOI-sensitivity table
# compares like with like; without it every row would spuriously report
# support_changed = True purely because of naming.
STEP9G_SUPPORT_EQUIVALENCE = {
    "bootstrap_supported_higher_values_rank_burned": SUPPORT_POSITIVE,
    "bootstrap_supported_lower_values_rank_burned": SUPPORT_NEGATIVE,
    "interval_includes_chance": SUPPORT_ZERO,
    "unstable_bootstrap": SUPPORT_UNSTABLE,
    "unavailable": SUPPORT_UNAVAILABLE,
}


def normalize_step9g_support(status: Any) -> Any:
    """Map a frozen Step9G support label onto this module's signed vocabulary."""
    if not isinstance(status, str):
        return status
    return STEP9G_SUPPORT_EQUIVALENCE.get(status, status)


class EviaSignedAucError(SystemExit):
    """Fatal, contract-violating condition."""


# =============================================================================
# Helpers
# =============================================================================
def output_dir_for(experiment_id: str) -> Path:
    return OUTPUT_ROOT / experiment_id


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_auc(raw_auc: float) -> float:
    """The one and only signed-AUC definition used in this module."""
    return 2.0 * float(raw_auc) - 1.0


def _signed_support_status(
    ci_low: float | None, ci_high: float | None, stable: bool, available: bool
) -> str:
    if not available:
        return SUPPORT_UNAVAILABLE
    if not stable or ci_low is None or ci_high is None:
        return SUPPORT_UNSTABLE
    if ci_low > 0.0:
        return SUPPORT_POSITIVE
    if ci_high < 0.0:
        return SUPPORT_NEGATIVE
    return SUPPORT_ZERO


def _direction_from_signed(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0:
        return "higher_values_rank_burned"
    if value < 0:
        return "lower_values_rank_burned"
    return "exactly_at_chance"


# =============================================================================
# Core computation
# =============================================================================
def compute_experiment(experiment_id: str) -> dict[str, Any]:
    """Recompute raw + signed univariate AUC with 5 km block bootstrap."""
    df = load_step8a(experiment_id)
    pop = assign_blocks_then_filter(df, experiment_id)

    n_rows = int(len(pop))
    y = pd.to_numeric(pop[TARGET_COLUMN], errors="coerce")
    n_burned = int((y == 1).sum())
    n_unburned = int((y == 0).sum())
    n_blocks = int(pop["large_block_id"].nunique())

    if n_burned == 0 or n_unburned == 0:
        raise EviaSignedAucError(
            f"{experiment_id}: primary population '{PRIMARY_POPULATION}' lacks both "
            f"classes (burned={n_burned}, unburned={n_unburned}); AUC is undefined."
        )

    summary_rows: list[dict[str, Any]] = []
    replicate_frames: list[pd.DataFrame] = []

    for feature in NUMERIC_FEATURES:
        point = univariate_feature_stats(pop, feature)
        boot = _block_bootstrap_auc(pop, feature, seed=BOOTSTRAP_SEED)

        available = bool(point.get("available")) and point.get("raw_univariate_auc") is not None
        raw_auc = point.get("raw_univariate_auc")
        arr = boot["replicate_aucs"]

        if available and arr.size:
            raw_lo = float(np.percentile(arr, CI_LOWER_PCT))
            raw_hi = float(np.percentile(arr, CI_UPPER_PCT))
            # Signed CI is taken DIRECTLY from the replicate-level signed
            # distribution, as required -- not by transforming the raw CI.
            signed_arr = 2.0 * arr - 1.0
            sgn_lo = float(np.percentile(signed_arr, CI_LOWER_PCT))
            sgn_hi = float(np.percentile(signed_arr, CI_UPPER_PCT))
            sgn = signed_auc(raw_auc)
            replicate_frames.append(
                pd.DataFrame(
                    {
                        "experiment_id": experiment_id,
                        "feature": feature,
                        "replicate": np.arange(arr.size, dtype=int),
                        "raw_auc": arr,
                        "signed_auc": signed_arr,
                    }
                )
            )
        else:
            raw_lo = raw_hi = sgn_lo = sgn_hi = sgn = None

        status = _signed_support_status(sgn_lo, sgn_hi, bool(boot["stable"]), available)

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "feature": feature,
                "n_rows": n_rows,
                "n_burned": n_burned,
                "n_unburned": n_unburned,
                "n_spatial_blocks": n_blocks,
                "raw_auc": raw_auc,
                "raw_auc_ci_low": raw_lo,
                "raw_auc_ci_high": raw_hi,
                "signed_auc": sgn,
                "signed_auc_ci_low": sgn_lo,
                "signed_auc_ci_high": sgn_hi,
                "direction": _direction_from_signed(sgn),
                "support_status": status,
                "successful_replicates": int(boot["valid"]),
                "requested_replicates": int(BOOTSTRAP_REPLICATES),
                "n_complete_case": point.get("n_complete_case"),
                "n_missing": point.get("n_missing"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    replicates = (
        pd.concat(replicate_frames, ignore_index=True)
        if replicate_frames
        else pd.DataFrame(columns=["experiment_id", "feature", "replicate", "raw_auc", "signed_auc"])
    )
    return {
        "summary": summary,
        "replicates": replicates,
        "n_rows": n_rows,
        "n_burned": n_burned,
        "n_unburned": n_unburned,
        "n_spatial_blocks": n_blocks,
    }


# =============================================================================
# Step9G cross-validation + old/extended sensitivity
# =============================================================================
def _load_frozen_step9g(experiment_id: str) -> pd.DataFrame:
    """Collect the frozen Step9G univariate rows for one experiment.

    Step9G writes one table per cross-region pair directory; a given experiment
    appears in several. The values are deterministic, so every occurrence must
    agree -- this function verifies that and returns the unique rows.
    """
    frames = []
    for path in sorted(STEP9G_ROOT.glob("*/step9g_univariate_auc_by_region.csv")):
        table = pd.read_csv(path)
        sub = table[table["experiment_id"] == experiment_id]
        if not sub.empty:
            frames.append(sub.assign(_source=path.parent.name))
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    drift = merged.groupby("feature")["raw_univariate_auc"].nunique()
    if (drift > 1).any():
        bad = sorted(drift[drift > 1].index)
        raise EviaSignedAucError(
            f"{experiment_id}: frozen Step9G raw AUC disagrees across pair "
            f"directories for {bad}; the frozen inputs are inconsistent."
        )
    return merged.drop_duplicates(subset=["feature"]).reset_index(drop=True)


def crossvalidate_against_step9g(
    computed: pd.DataFrame, experiment_id: str, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Prove this module reproduces the frozen Step9G point estimates."""
    frozen = _load_frozen_step9g(experiment_id)
    if frozen.empty:
        return {"available": False, "reason": "no frozen Step9G rows for this experiment"}

    merged = computed.merge(
        frozen[["feature", "raw_univariate_auc", "signed_rank_effect"]],
        on="feature",
        how="inner",
    )
    merged["raw_abs_diff"] = (merged["raw_auc"] - merged["raw_univariate_auc"]).abs()
    merged["signed_abs_diff"] = (merged["signed_auc"] - merged["signed_rank_effect"]).abs()
    max_raw = float(merged["raw_abs_diff"].max())
    max_signed = float(merged["signed_abs_diff"].max())
    return {
        "available": True,
        "features_compared": int(len(merged)),
        "max_abs_diff_raw_auc": max_raw,
        "max_abs_diff_signed_auc": max_signed,
        "tolerance": tolerance,
        "reproduces_frozen_step9g": bool(max_raw <= tolerance and max_signed <= tolerance),
        "per_feature": merged[
            ["feature", "raw_auc", "raw_univariate_auc", "raw_abs_diff", "signed_abs_diff"]
        ].to_dict(orient="records"),
    }


def build_sensitivity_table(extended_summary: pd.DataFrame) -> pd.DataFrame:
    """Secondary AOI-sensitivity comparison: old evia_2021 vs extended.

    The OLD Evia numbers are READ from frozen Step9G. No model, no Step10 and no
    bootstrap is re-run for evia_2021.
    """
    old = _load_frozen_step9g(SENSITIVITY_EXPERIMENT)
    if old.empty:
        return pd.DataFrame()

    old = old.rename(
        columns={
            "signed_rank_effect": "old_signed_auc",
            "raw_univariate_auc": "old_raw_auc",
            "auc_ci_low": "old_raw_ci_low",
            "auc_ci_high": "old_raw_ci_high",
            "support_status": "old_support_status",
        }
    )
    # Frozen Step9G stores the RAW-AUC CI; map it onto the signed scale with the
    # same monotone transform used for the point estimate.
    old["old_signed_ci_low"] = 2.0 * old["old_raw_ci_low"] - 1.0
    old["old_signed_ci_high"] = 2.0 * old["old_raw_ci_high"] - 1.0
    # Normalise the frozen raw-scale label onto the signed vocabulary so that
    # support_changed reports a genuine change, not a renaming.
    old["old_support_status_raw_label"] = old["old_support_status"]
    old["old_support_status"] = old["old_support_status"].map(normalize_step9g_support)

    ext = extended_summary.rename(
        columns={
            "signed_auc": "extended_signed_auc",
            "signed_auc_ci_low": "extended_signed_ci_low",
            "signed_auc_ci_high": "extended_signed_ci_high",
            "support_status": "extended_support_status",
        }
    )

    merged = ext[
        [
            "feature",
            "extended_signed_auc",
            "extended_signed_ci_low",
            "extended_signed_ci_high",
            "extended_support_status",
        ]
    ].merge(
        old[
            [
                "feature",
                "old_signed_auc",
                "old_signed_ci_low",
                "old_signed_ci_high",
                "old_support_status",
                "old_support_status_raw_label",
            ]
        ],
        on="feature",
        how="outer",
    )

    def _sign(v: Any) -> int | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(np.sign(v))

    merged["direction_preserved"] = [
        None
        if _sign(a) is None or _sign(b) is None
        else bool(_sign(a) == _sign(b))
        for a, b in zip(merged["old_signed_auc"], merged["extended_signed_auc"])
    ]
    merged["support_changed"] = [
        None
        if (not isinstance(a, str)) or (not isinstance(b, str))
        else bool(a != b)
        for a, b in zip(merged["old_support_status"], merged["extended_support_status"])
    ]
    merged["old_signed_ci"] = [
        None if pd.isna(lo) or pd.isna(hi) else f"[{lo:.4f}, {hi:.4f}]"
        for lo, hi in zip(merged["old_signed_ci_low"], merged["old_signed_ci_high"])
    ]
    merged["extended_signed_ci"] = [
        None if pd.isna(lo) or pd.isna(hi) else f"[{lo:.4f}, {hi:.4f}]"
        for lo, hi in zip(merged["extended_signed_ci_low"], merged["extended_signed_ci_high"])
    ]
    return merged


# =============================================================================
# Preregistration / audit / manifest
# =============================================================================
def build_preregistration(experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "evia_signed_auc_spatial_block_bootstrap",
        "canonical_experiment_id": experiment_id,
        "secondary_sensitivity_experiment_id": SENSITIVITY_EXPERIMENT,
        "estimand": (
            "Per-feature RAW univariate ROC-AUC of a single predictor against the "
            "burned label, expressed on the signed scale 2*AUC-1."
        ),
        "signed_auc_definition": "signed_auc = 2 * raw_auc - 1",
        "auc_inversion": "forbidden",
        "population": PRIMARY_POPULATION,
        "bootstrap": {
            "unit": "spatial_block",
            "block_size_cells": BLOCK_SIZE_CELLS,
            "nominal_block_scale": NOMINAL_BLOCK_SCALE,
            "cell_size_m": 500,
            "resampling": "whole blocks, with replacement, multiplicity preserved",
            "row_bootstrap": "forbidden",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": f"{CI_LOWER_PCT}/{CI_UPPER_PCT} percentile",
            "ci_taken_from": "replicate-level SIGNED distribution",
            "min_successful_replicates": MIN_VALID_REPLICATES,
        },
        "features": list(NUMERIC_FEATURES),
        "target_label_use": "diagnostic AUC computation only",
        "support_convention": {
            "signed_ci_entirely_above_zero": SUPPORT_POSITIVE,
            "signed_ci_entirely_below_zero": SUPPORT_NEGATIVE,
            "otherwise": SUPPORT_ZERO,
        },
        "reused_module": "src.step9g_univariate_feature_auc_direction_reversal",
        "rejected_module": {
            "module": "src.step10c_paired_evaluation_bootstrap",
            "reason": (
                "Different estimand (model-level paired transfer AUC) and a "
                "different resampling frame (aligned source->target prediction "
                "join). Reusing it would silently change the estimand."
            ),
        },
        "writes_only_under": str(OUTPUT_ROOT),
        "reruns_nothing": [
            "step8", "step9", "step9e", "step9g", "step10", "multi_aoi_transfer_synthesis",
        ],
    }


def build_input_audit(experiment_id: str) -> dict[str, Any]:
    step8a = PROJECT_ROOT / "outputs" / "experiments" / experiment_id / "step8a" / "step8a_500m_modeling_dataset.parquet"
    entries = [
        {
            "role": "primary_input",
            "experiment_id": experiment_id,
            "path": str(step8a.relative_to(PROJECT_ROOT)),
            "exists": step8a.is_file(),
            "sha256": _sha256(step8a),
        }
    ]
    for path in sorted(STEP9G_ROOT.glob("*/step9g_univariate_auc_by_region.csv")):
        entries.append(
            {
                "role": "frozen_step9g_reference",
                "pair": path.parent.name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": entries,
        "input_count": len(entries),
    }


# =============================================================================
# Rendering
# =============================================================================
def render_markdown(payload: dict[str, Any], summary: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    eid = payload["experiment_id"]
    L: list[str] = []
    A = L.append
    A(f"# Evia signed-AUC spatial-block bootstrap -- `{eid}`")
    A("")
    A(f"- Generated: `{payload['created_at']}`")
    A(f"- Population: `{PRIMARY_POPULATION}` -- {payload['n_rows']:,} cells "
      f"({payload['n_burned']:,} burned / {payload['n_unburned']:,} unburned)")
    A(f"- Spatial blocks: {payload['n_spatial_blocks']:,} "
      f"({BLOCK_SIZE_CELLS}x{BLOCK_SIZE_CELLS} cells of 500 m = {NOMINAL_BLOCK_SCALE})")
    A(f"- Bootstrap: {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}, "
      f"{CI_LOWER_PCT}/{CI_UPPER_PCT} percentile CI, min successful {MIN_VALID_REPLICATES}")
    A("")
    A("`signed_auc = 2 * raw_auc - 1`. The AUC is never inverted, so a feature whose "
      "high values rank UNBURNED cells higher keeps a negative signed value.")
    A("")
    A("## Primary result")
    A("")
    A("| Feature | raw AUC | raw 95% CI | signed AUC | signed 95% CI | Support | Reps |")
    A("|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        if r["raw_auc"] is None or pd.isna(r["raw_auc"]):
            A(f"| `{r['feature']}` | - | - | - | - | {r['support_status']} | "
              f"{int(r['successful_replicates'])}/{int(r['requested_replicates'])} |")
            continue
        A(
            f"| `{r['feature']}` | {r['raw_auc']:.4f} | "
            f"[{r['raw_auc_ci_low']:.4f}, {r['raw_auc_ci_high']:.4f}] | "
            f"{r['signed_auc']:+.4f} | "
            f"[{r['signed_auc_ci_low']:+.4f}, {r['signed_auc_ci_high']:+.4f}] | "
            f"{r['support_status']} | "
            f"{int(r['successful_replicates'])}/{int(r['requested_replicates'])} |"
        )
    A("")

    counts = summary["support_status"].value_counts().to_dict()
    A("### Support summary")
    A("")
    for k, v in sorted(counts.items()):
        A(f"- `{k}`: {v}")
    A("")

    xv = payload.get("step9g_crossvalidation", {})
    A("## Cross-validation against frozen Step9G")
    A("")
    if xv.get("available"):
        A(f"- Features compared: {xv['features_compared']}")
        A(f"- Max |Δ| raw AUC: `{xv['max_abs_diff_raw_auc']:.3e}`")
        A(f"- Max |Δ| signed AUC: `{xv['max_abs_diff_signed_auc']:.3e}`")
        A(f"- Reproduces frozen Step9G: **{xv['reproduces_frozen_step9g']}** "
          f"(tolerance `{xv['tolerance']:.0e}`)")
        A("")
        A("This module recomputes the estimand independently and lands on the frozen "
          "Step9G point estimates, which confirms the estimand and the resampling "
          "contract are identical.")
    else:
        A(f"- Not available: {xv.get('reason')}")
    A("")

    A("## Secondary: AOI sensitivity (old `evia_2021` vs extended)")
    A("")
    if sensitivity.empty:
        A("_No frozen Step9G rows found for `evia_2021`._")
    else:
        A("Old-Evia values are READ from frozen Step9G; nothing was re-run for it.")
        A("")
        A("| Feature | old signed | old CI | extended signed | extended CI | Direction preserved | Support changed |")
        A("|---|---|---|---|---|---|---|")
        for _, r in sensitivity.iterrows():
            osa = "-" if pd.isna(r["old_signed_auc"]) else f"{r['old_signed_auc']:+.4f}"
            esa = "-" if pd.isna(r["extended_signed_auc"]) else f"{r['extended_signed_auc']:+.4f}"
            A(
                f"| `{r['feature']}` | {osa} | {r['old_signed_ci'] or '-'} | "
                f"{esa} | {r['extended_signed_ci'] or '-'} | "
                f"{r['direction_preserved']} | {r['support_changed']} |"
            )
    A("")
    A("> The old 0.40x0.40 deg Evia AOI carries ~67% burned prevalence inside the "
      "burnable mask versus ~28.6% for the extended AOI. The comparison above is an "
      "AOI-sensitivity diagnostic, NOT a second primary result.")
    A("")
    return "\n".join(L)


# =============================================================================
# Entry point
# =============================================================================
def run(
    experiment_id: str = CANONICAL_EXPERIMENT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    dry_run: bool = False,
    force: bool = False,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise EviaSignedAucError(
            f"bootstrap_replicates={bootstrap_replicates} does not match the "
            f"preregistered {BOOTSTRAP_REPLICATES}; refusing to run off-contract."
        )
    if seed != BOOTSTRAP_SEED:
        raise EviaSignedAucError(
            f"seed={seed} does not match the preregistered {BOOTSTRAP_SEED}; "
            "refusing to run off-contract."
        )

    out_dir = (output_root or OUTPUT_ROOT) / experiment_id
    prereg = build_preregistration(experiment_id)
    audit = build_input_audit(experiment_id)

    filenames = [
        "evia_signed_auc_preregistration.json",
        "evia_signed_auc_input_audit.json",
        "evia_signed_auc_summary.csv",
        "evia_signed_auc_bootstrap_replicates.parquet",
        "evia_signed_auc_final_report.json",
        "evia_signed_auc_final_report.md",
        "evia_signed_auc_manifest.json",
    ]

    if dry_run:
        missing = [e["path"] for e in audit["inputs"] if not e["exists"]]
        return {
            "ran": False,
            "dry_run": True,
            "experiment_id": experiment_id,
            "preregistration": prereg,
            "input_audit": audit,
            "missing_inputs": missing,
            "planned_output_paths": {n: str(out_dir / n) for n in filenames},
            "output_dir_exists": out_dir.exists(),
        }

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise EviaSignedAucError(
            f"{out_dir} already contains outputs; pass force=True to overwrite."
        )

    computed = compute_experiment(experiment_id)
    summary: pd.DataFrame = computed["summary"]
    replicates: pd.DataFrame = computed["replicates"]

    xval = crossvalidate_against_step9g(summary, experiment_id)
    sensitivity = build_sensitivity_table(summary)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "population": PRIMARY_POPULATION,
        "n_rows": computed["n_rows"],
        "n_burned": computed["n_burned"],
        "n_unburned": computed["n_unburned"],
        "n_spatial_blocks": computed["n_spatial_blocks"],
        "preregistration": prereg,
        "input_audit": audit,
        "summary": summary.to_dict(orient="records"),
        "step9g_crossvalidation": xval,
        "sensitivity_old_vs_extended": sensitivity.to_dict(orient="records"),
        "support_counts": summary["support_status"].value_counts().to_dict(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evia_signed_auc_preregistration.json").write_text(
        json.dumps(prereg, indent=2, ensure_ascii=False)
    )
    (out_dir / "evia_signed_auc_input_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    summary.to_csv(out_dir / "evia_signed_auc_summary.csv", index=False)
    replicates.to_parquet(out_dir / "evia_signed_auc_bootstrap_replicates.parquet", index=False)
    (out_dir / "evia_signed_auc_final_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    (out_dir / "evia_signed_auc_final_report.md").write_text(
        render_markdown(payload, summary, sensitivity)
    )
    if not sensitivity.empty:
        sensitivity.to_csv(out_dir / "evia_signed_auc_aoi_sensitivity.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": payload["created_at"],
        "experiment_id": experiment_id,
        "outputs": [
            {
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(out_dir.glob("*"))
            if p.is_file() and p.name != "evia_signed_auc_manifest.json"
        ],
        "inputs": audit["inputs"],
    }
    (out_dir / "evia_signed_auc_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    return {
        "ran": True,
        "dry_run": False,
        "experiment_id": experiment_id,
        "output_dir": str(out_dir),
        "support_counts": payload["support_counts"],
        "step9g_crossvalidation": xval,
        "n_features": int(len(summary)),
    }
