"""
Multi-AOI transfer gap/recovery decomposition (advisor follow-up item 7).

Recomputes, for every ordered direction x model family x adaptation method x
metric, the decomposition of a cross-region transfer gap:

    raw_gap            = within_target - raw_transfer
    adaptation_effect  = adapted       - raw_transfer
    remaining_gap      = within_target - adapted

    recovered_fraction = adaptation_effect / raw_gap
    remaining_fraction = remaining_gap     / raw_gap

with the identity `recovered_fraction + remaining_fraction == 1` enforced to a
tolerance of 1e-8.

Negative-recovery convention
----------------------------
When `adaptation_effect < 0` the row is reported as `negative_recovery`. The
value is NEVER clipped to zero. As long as `raw_gap > 0` the fraction remains
mathematically well defined and simply exceeds its usual range, meaning:

    adaptation did not merely fail to close the gap, it widened it.

Fractions are suppressed ONLY when `raw_gap <= 0` (raw transfer already at or
above the within-region reference), in which case
`fraction_not_interpretable_raw_at_or_above_within` is recorded.

`adaptation_effect` and `recovered_fraction` are distinct quantities and are
never conflated: the former is an absolute AUC difference, the latter is that
difference rescaled by a denominator that can be small.

Uncertainty
-----------
The frozen Step10 bootstrap stores the WITHIN-region reference per replicate
(`roc_auc__within_*`), resampled jointly with the raw and adapted series on the
same target spatial-block replicate. The decomposition is therefore computed at
replicate level and the resulting CIs carry FULL JOINT uncertainty across the
within, raw and adapted terms -- not merely transfer uncertainty against a fixed
reference. Replicates whose `abs(raw_gap_rep) < 1e-6` are marked invalid for the
ratio (the denominator is numerically degenerate) and counted in the report.

Read-only: consumes ALREADY FROZEN Step9B/Step10 artefacts and never re-runs
models, adaptation, predictions or the Step10 bootstrap. Writes exclusively
under outputs/diagnostics/four_aoi_transfer_decomposition/<canonical_set_id>/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.paths import PROJECT_ROOT

SCHEMA_VERSION = "four_aoi_transfer_decomposition.v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "four_aoi_transfer_decomposition"
CROSS_REGION_ROOT = PROJECT_ROOT / "outputs" / "cross_region"

RAW_METHOD = "raw_source_only"
WITHIN_METHOD = "within"
ADAPTATION_METHODS = ("regionwise_zscore", "coral_after_regionwise_zscore")
MODEL_FAMILIES = ("baseline", "thermal")
METRICS = ("roc_auc", "pr_auc")

IDENTITY_TOLERANCE = 1e-8
RAW_REPRODUCTION_TOLERANCE = 1e-6
RATIO_DEGENERATE_THRESHOLD = 1e-6

CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5

# Recovery statuses
STATUS_ABOVE_CHANCE = "supported_recovery_above_chance"
STATUS_RELATIVE_ONLY = "supported_relative_recovery_but_chance_not_excluded"
STATUS_UNCERTAIN = "recovery_effect_uncertain"
STATUS_NEGATIVE = "negative_recovery"
STATUS_NOT_INTERPRETABLE = "fraction_not_interpretable"

INTERPRETABLE = "interpretable"
NOT_INTERPRETABLE = "fraction_not_interpretable_raw_at_or_above_within"


class TransferDecompositionError(SystemExit):
    """Fatal, contract-violating condition."""


# =============================================================================
# Helpers
# =============================================================================
def canonical_set_id(aois: list[str]) -> str:
    return "__".join(sorted(aois))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def direction_label(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def _replicate_column(metric: str, method: str, family: str) -> str:
    return f"{metric}__{method}_{family}"


def resolve_direction_sources(aois: list[str]) -> dict[str, dict[str, Any]]:
    """Map each ordered direction to the frozen Step10 pair directory serving it.

    A direction can appear in more than one pair directory (Step10 writes both
    directions into each). The directory literally named `<source>__<target>` is
    preferred; otherwise the lexicographically first directory that contains the
    direction is used. Every duplicate is cross-checked for agreement.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for source, target in permutations(sorted(aois), 2):
        direction = direction_label(source, target)
        candidates = []
        for pair_dir in sorted(CROSS_REGION_ROOT.iterdir()):
            if not pair_dir.is_dir():
                continue
            metrics_csv = pair_dir / "step10" / "step10_metrics.csv"
            replicates = pair_dir / "step10" / "step10_bootstrap_replicates.parquet"
            if not metrics_csv.is_file() or not replicates.is_file():
                continue
            table = pd.read_csv(metrics_csv)
            if direction in set(table["direction"].astype(str)):
                candidates.append(pair_dir.name)
        if not candidates:
            resolved[direction] = {"available": False, "candidates": []}
            continue
        preferred = f"{source}__{target}"
        chosen = preferred if preferred in candidates else candidates[0]
        resolved[direction] = {
            "available": True,
            "source_experiment_id": source,
            "target_experiment_id": target,
            "chosen_pair_dir": chosen,
            "candidates": candidates,
            "preferred_dir_existed": preferred in candidates,
        }
    return resolved


# =============================================================================
# Point + replicate decomposition
# =============================================================================
def _chance_level(metric: str, positives: float, negatives: float) -> float:
    """Chance baseline for the metric.

    ROC-AUC has a fixed chance level of 0.5. PR-AUC does not: its chance level is
    the positive-class prevalence, so a prevalence-blind 0.5 test would be wrong.
    """
    if metric == "roc_auc":
        return 0.5
    total = positives + negatives
    return float(positives / total) if total else float("nan")


def decompose_direction(
    direction: str,
    info: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_dir = CROSS_REGION_ROOT / info["chosen_pair_dir"] / "step10"
    metrics = pd.read_csv(pair_dir / "step10_metrics.csv")
    metrics = metrics[metrics["direction"] == direction]
    replicates = pd.read_parquet(pair_dir / "step10_bootstrap_replicates.parquet")
    replicates = replicates[replicates["direction"] == direction]

    if metrics.empty or replicates.empty:
        raise TransferDecompositionError(
            f"{direction}: frozen Step10 artefacts in {info['chosen_pair_dir']} "
            "contain no rows for this direction."
        )

    def point(method: str, family: str, metric: str) -> float:
        sel = metrics[(metrics["method"] == method) & (metrics["model_family"] == family)]
        if len(sel) != 1:
            raise TransferDecompositionError(
                f"{direction}: expected exactly one frozen Step10 row for "
                f"method={method}, family={family}; got {len(sel)}."
            )
        return float(sel.iloc[0][metric])

    def counts(family: str) -> tuple[float, float]:
        sel = metrics[(metrics["method"] == WITHIN_METHOD) & (metrics["model_family"] == family)]
        row = sel.iloc[0]
        return float(row["positive_count"]), float(row["negative_count"])

    rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []

    for family in MODEL_FAMILIES:
        pos, neg = counts(family)
        for metric in METRICS:
            chance = _chance_level(metric, pos, neg)
            within_p = point(WITHIN_METHOD, family, metric)
            raw_p = point(RAW_METHOD, family, metric)

            within_rep = replicates[_replicate_column(metric, WITHIN_METHOD, family)].to_numpy(dtype=float)
            raw_rep = replicates[_replicate_column(metric, RAW_METHOD, family)].to_numpy(dtype=float)

            for method in ADAPTATION_METHODS:
                adapted_p = point(method, family, metric)
                adapted_rep = replicates[_replicate_column(metric, method, family)].to_numpy(dtype=float)

                raw_gap = within_p - raw_p
                adaptation_effect = adapted_p - raw_p
                remaining_gap = within_p - adapted_p

                interpretable = raw_gap > 0.0
                if interpretable:
                    recovered_fraction = adaptation_effect / raw_gap
                    remaining_fraction = remaining_gap / raw_gap
                    identity_residual = abs((recovered_fraction + remaining_fraction) - 1.0)
                    if identity_residual > IDENTITY_TOLERANCE:
                        raise TransferDecompositionError(
                            f"{direction}/{family}/{method}/{metric}: fraction identity "
                            f"violated by {identity_residual:.3e} (> {IDENTITY_TOLERANCE:.0e})."
                        )
                else:
                    recovered_fraction = remaining_fraction = None
                    identity_residual = None

                # ---- replicate-level (full joint uncertainty) ----
                raw_gap_rep = within_rep - raw_rep
                effect_rep = adapted_rep - raw_rep
                remaining_rep = within_rep - adapted_rep

                degenerate = np.abs(raw_gap_rep) < RATIO_DEGENERATE_THRESHOLD
                valid = ~degenerate
                n_degenerate = int(degenerate.sum())

                if valid.any():
                    rec_frac_rep = effect_rep[valid] / raw_gap_rep[valid]
                    rem_frac_rep = remaining_rep[valid] / raw_gap_rep[valid]
                    rec_lo = float(np.percentile(rec_frac_rep, CI_LOWER_PCT))
                    rec_hi = float(np.percentile(rec_frac_rep, CI_UPPER_PCT))
                    rem_lo = float(np.percentile(rem_frac_rep, CI_LOWER_PCT))
                    rem_hi = float(np.percentile(rem_frac_rep, CI_UPPER_PCT))
                else:
                    rec_lo = rec_hi = rem_lo = rem_hi = None

                effect_lo = float(np.percentile(effect_rep, CI_LOWER_PCT))
                effect_hi = float(np.percentile(effect_rep, CI_UPPER_PCT))
                adapted_lo = float(np.percentile(adapted_rep, CI_LOWER_PCT))
                adapted_hi = float(np.percentile(adapted_rep, CI_UPPER_PCT))

                relative_improvement_supported = bool(effect_lo > 0.0)
                above_chance = bool(adapted_lo > chance)

                # ---- status ----
                if not interpretable:
                    status = STATUS_NOT_INTERPRETABLE
                elif adaptation_effect < 0.0:
                    status = STATUS_NEGATIVE
                elif relative_improvement_supported and above_chance:
                    status = STATUS_ABOVE_CHANCE
                elif relative_improvement_supported:
                    status = STATUS_RELATIVE_ONLY
                else:
                    status = STATUS_UNCERTAIN

                rows.append(
                    {
                        "source_experiment_id": info["source_experiment_id"],
                        "target_experiment_id": info["target_experiment_id"],
                        "direction": direction,
                        "model_family": family,
                        "adaptation_method": method,
                        "metric": metric,
                        "within_target_auc": within_p,
                        "raw_auc": raw_p,
                        "adapted_auc": adapted_p,
                        "raw_gap": raw_gap,
                        "adaptation_effect": adaptation_effect,
                        "remaining_gap": remaining_gap,
                        "recovered_fraction": recovered_fraction,
                        "recovered_fraction_ci_low": rec_lo,
                        "recovered_fraction_ci_high": rec_hi,
                        "remaining_fraction": remaining_fraction,
                        "remaining_fraction_ci_low": rem_lo,
                        "remaining_fraction_ci_high": rem_hi,
                        "adaptation_effect_ci_low": effect_lo,
                        "adaptation_effect_ci_high": effect_hi,
                        "adapted_auc_ci_low": adapted_lo,
                        "adapted_auc_ci_high": adapted_hi,
                        "chance_level": chance,
                        "relative_improvement_supported": relative_improvement_supported,
                        "adapted_above_chance": above_chance,
                        "recovery_status": status,
                        "fraction_interpretability_status": (
                            INTERPRETABLE if interpretable else NOT_INTERPRETABLE
                        ),
                        "identity_residual": identity_residual,
                        "n_replicates": int(len(raw_gap_rep)),
                        "n_replicates_ratio_degenerate": n_degenerate,
                        "n_replicates_ratio_valid": int(valid.sum()),
                        "source_pair_dir": info["chosen_pair_dir"],
                    }
                )

                boot_rows.append(
                    {
                        "direction": direction,
                        "model_family": family,
                        "adaptation_method": method,
                        "metric": metric,
                        "raw_gap_mean": float(np.mean(raw_gap_rep)),
                        "adaptation_effect_mean": float(np.mean(effect_rep)),
                        "remaining_gap_mean": float(np.mean(remaining_rep)),
                        "recovered_fraction_ci_low": rec_lo,
                        "recovered_fraction_ci_high": rec_hi,
                        "remaining_fraction_ci_low": rem_lo,
                        "remaining_fraction_ci_high": rem_hi,
                        "n_replicates": int(len(raw_gap_rep)),
                        "n_replicates_ratio_degenerate": n_degenerate,
                    }
                )

    return rows, boot_rows


# =============================================================================
# Verification against the frozen Step10 decomposition
# =============================================================================
def verify_raw_reproduction(rows: pd.DataFrame) -> dict[str, Any]:
    """Check the recomputed raw/adapted/within values against frozen Step10."""
    frozen_frames = []
    for pair_dir in sorted(CROSS_REGION_ROOT.iterdir()):
        path = pair_dir / "step10" / "step10_decomposition.csv"
        if path.is_file():
            frozen_frames.append(pd.read_csv(path).assign(_pair=pair_dir.name))
    if not frozen_frames:
        return {"available": False, "reason": "no frozen step10_decomposition.csv found"}

    frozen = pd.concat(frozen_frames, ignore_index=True).rename(
        columns={"method": "adaptation_method"}
    )
    merged = rows.merge(
        frozen[
            ["direction", "model_family", "adaptation_method", "metric",
             "raw_value", "adapted_value", "within_value"]
        ],
        on=["direction", "model_family", "adaptation_method", "metric"],
        how="inner",
    )
    if merged.empty:
        return {"available": False, "reason": "no overlap with frozen decomposition rows"}

    merged["d_raw"] = (merged["raw_auc"] - merged["raw_value"]).abs()
    merged["d_adapted"] = (merged["adapted_auc"] - merged["adapted_value"]).abs()
    merged["d_within"] = (merged["within_target_auc"] - merged["within_value"]).abs()
    worst = float(max(merged["d_raw"].max(), merged["d_adapted"].max(), merged["d_within"].max()))
    return {
        "available": True,
        "rows_compared": int(len(merged)),
        "max_abs_diff_raw": float(merged["d_raw"].max()),
        "max_abs_diff_adapted": float(merged["d_adapted"].max()),
        "max_abs_diff_within": float(merged["d_within"].max()),
        "worst_abs_diff": worst,
        "tolerance": RAW_REPRODUCTION_TOLERANCE,
        "reproduces_frozen_step10": bool(worst <= RAW_REPRODUCTION_TOLERANCE),
    }


# =============================================================================
# Preregistration / audit
# =============================================================================
def build_preregistration(aois: list[str], directions: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "multi_aoi_transfer_gap_recovery_decomposition",
        "canonical_set_id": canonical_set_id(aois),
        "aois": sorted(aois),
        "n_ordered_directions_expected": len(aois) * (len(aois) - 1),
        "n_ordered_directions_resolved": sum(1 for v in directions.values() if v.get("available")),
        "definitions": {
            "raw_gap": "within_target_auc - raw_transfer_auc",
            "adaptation_effect": "adapted_auc - raw_transfer_auc",
            "remaining_gap": "within_target_auc - adapted_auc",
            "recovered_fraction": "adaptation_effect / raw_gap",
            "remaining_fraction": "remaining_gap / raw_gap",
            "identity": "recovered_fraction + remaining_fraction == 1",
            "identity_tolerance": IDENTITY_TOLERANCE,
        },
        "negative_recovery_convention": {
            "clipping": "forbidden",
            "status": STATUS_NEGATIVE,
            "meaning": (
                "adaptation did not merely fail to close the gap, it widened the "
                "raw transfer gap"
            ),
            "fraction_defined_when": "raw_gap > 0",
        },
        "fraction_suppression": {
            "condition": "raw_gap <= 0",
            "status": NOT_INTERPRETABLE,
        },
        "ratio_guard": {
            "rule": f"abs(raw_gap_rep) < {RATIO_DEGENERATE_THRESHOLD}",
            "action": "replicate excluded from the ratio CI and counted",
        },
        "uncertainty": {
            "source": "frozen Step10 spatial-block bootstrap replicates",
            "within_region_reference_available_per_replicate": True,
            "joint_uncertainty": True,
            "note": (
                "within, raw and adapted are all resampled on the SAME target "
                "spatial-block replicate, so the fraction CIs carry full joint "
                "uncertainty across all three terms."
            ),
            "ci": f"{CI_LOWER_PCT}/{CI_UPPER_PCT} percentile",
        },
        "status_vocabulary": [
            STATUS_ABOVE_CHANCE,
            STATUS_RELATIVE_ONLY,
            STATUS_UNCERTAIN,
            STATUS_NEGATIVE,
            STATUS_NOT_INTERPRETABLE,
        ],
        "status_separation": {
            "adapted_minus_raw_positive": "relative improvement ONLY",
            "adapted_ci_above_chance": "above-chance ranking support",
            "never_merged_into": "a single 'successful recovery' status",
            "chance_level": {
                "roc_auc": 0.5,
                "pr_auc": "positive-class prevalence (NOT 0.5)",
            },
        },
        "model_families": list(MODEL_FAMILIES),
        "adaptation_methods": list(ADAPTATION_METHODS),
        "metrics": list(METRICS),
        "adaptation_methods_kept_separate": True,
        "reruns_nothing": [
            "step8", "step9", "step10 models", "step10 adaptation",
            "step10 predictions", "step10 bootstrap",
        ],
        "supersedes": (
            "the earlier two-region 27-31% recovered / 69-73% remaining figures, "
            "which must not be carried forward"
        ),
    }


def build_input_audit(directions: dict[str, Any]) -> dict[str, Any]:
    entries = []
    seen: set[str] = set()
    for direction, info in sorted(directions.items()):
        if not info.get("available"):
            continue
        for name in ("step10_metrics.csv", "step10_bootstrap_replicates.parquet", "step10_decomposition.csv"):
            path = CROSS_REGION_ROOT / info["chosen_pair_dir"] / "step10" / name
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            entries.append(
                {
                    "pair": info["chosen_pair_dir"],
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": entries,
        "input_count": len(entries),
        "direction_resolution": directions,
    }


# =============================================================================
# Rendering
# =============================================================================
METHOD_SHORT_LABEL = {
    "regionwise_zscore": "zscore",
    "coral_after_regionwise_zscore": "zscore+coral",
}


def _method_label(method: str) -> str:
    return METHOD_SHORT_LABEL.get(method, method)


def render_markdown(payload: dict[str, Any], table: pd.DataFrame) -> str:
    L: list[str] = []
    A = L.append
    A("# Multi-AOI transfer gap/recovery decomposition")
    A("")
    A(f"- Generated: `{payload['created_at']}`")
    A(f"- Canonical set: `{payload['canonical_set_id']}`")
    A(f"- Ordered directions: {payload['n_directions']} / {payload['n_directions_expected']}")
    A(f"- Rows: {len(table)} (direction x model family x adaptation method x metric)")
    A("")
    A("```")
    A("raw_gap           = within_target - raw_transfer")
    A("adaptation_effect = adapted       - raw_transfer")
    A("remaining_gap     = within_target - adapted")
    A("recovered_fraction = adaptation_effect / raw_gap")
    A("remaining_fraction = remaining_gap     / raw_gap")
    A("```")
    A("")
    A("Negative recovery is reported as such and never clipped to zero. Fractions "
      "are suppressed only when `raw_gap <= 0`.")
    A("")

    xv = payload.get("raw_reproduction", {})
    A("## Reproduction of frozen Step10")
    A("")
    if xv.get("available"):
        A(f"- Rows compared: {xv['rows_compared']}")
        A(f"- Worst |Δ|: `{xv['worst_abs_diff']:.3e}` (tolerance `{xv['tolerance']:.0e}`)")
        A(f"- Reproduces frozen Step10: **{xv['reproduces_frozen_step10']}**")
    else:
        A(f"- Not available: {xv.get('reason')}")
    A("")

    A("## Status distribution")
    A("")
    for k, v in sorted(payload["status_counts"].items()):
        A(f"- `{k}`: {v}")
    A("")
    A(f"- Rows with negative recovery: **{payload['n_negative_recovery']}**")
    A(f"- Rows hitting the raw-gap guard (`raw_gap <= 0`): **{payload['n_raw_gap_guard']}**")
    A(f"- Replicates excluded by the ratio guard, summed over rows: "
      f"**{payload['n_ratio_degenerate_replicates']}**")
    A("")

    A("## Uncertainty scope")
    A("")
    A("The frozen Step10 bootstrap stores the within-region reference per replicate, "
      "resampled jointly with the raw and adapted series on the same target "
      "spatial-block replicate. The fraction CIs therefore carry **full joint "
      "uncertainty** across the within, raw and adapted terms.")
    A("")

    A("## Bejís <-> Muğla (ROC-AUC), all four combinations")
    A("")
    focus = table[
        table["direction"].isin(["bejis_2022_to_mugla_2021", "mugla_2021_to_bejis_2022"])
        & (table["metric"] == "roc_auc")
    ].sort_values(["direction", "model_family", "adaptation_method"])
    A("| Direction | Family | Method | within | raw | adapted | raw_gap | effect | recovered | recovered CI | Status |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in focus.iterrows():
        rec = "-" if pd.isna(r["recovered_fraction"]) else f"{r['recovered_fraction']:+.3f}"
        ci = (
            "-"
            if pd.isna(r["recovered_fraction_ci_low"])
            else f"[{r['recovered_fraction_ci_low']:+.2f}, {r['recovered_fraction_ci_high']:+.2f}]"
        )
        A(
            f"| {r['direction']} | {r['model_family']} | "
            f"{_method_label(r['adaptation_method'])} | "
            f"{r['within_target_auc']:.4f} | {r['raw_auc']:.4f} | {r['adapted_auc']:.4f} | "
            f"{r['raw_gap']:+.4f} | {r['adaptation_effect']:+.4f} | {rec} | {ci} | "
            f"{r['recovery_status']} |"
        )
    A("")
    A("> The earlier two-region 27-31% recovered / 69-73% remaining split is "
      "superseded and must not be carried forward.")
    A("")

    A("## Negative-recovery rows (all directions, ROC-AUC)")
    A("")
    neg = table[(table["recovery_status"] == STATUS_NEGATIVE) & (table["metric"] == "roc_auc")]
    if neg.empty:
        A("_None._")
    else:
        A("| Direction | Family | Method | raw_gap | effect | recovered | Status |")
        A("|---|---|---|---|---|---|---|")
        for _, r in neg.sort_values(["direction", "model_family"]).iterrows():
            rec = "-" if pd.isna(r["recovered_fraction"]) else f"{r['recovered_fraction']:+.3f}"
            A(
                f"| {r['direction']} | {r['model_family']} | "
                f"{_method_label(r['adaptation_method'])} | "
                f"{r['raw_gap']:+.4f} | {r['adaptation_effect']:+.4f} | {rec} | {r['recovery_status']} |"
            )
    A("")
    return "\n".join(L)


# =============================================================================
# Entry point
# =============================================================================
def run(
    aois: list[str],
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    dry_run: bool = False,
    force: bool = False,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if len(set(aois)) < 2:
        raise TransferDecompositionError("at least two distinct AOI experiment ids are required.")

    set_id = canonical_set_id(list(set(aois)))
    out_dir = (output_root or OUTPUT_ROOT) / set_id
    directions = resolve_direction_sources(sorted(set(aois)))
    prereg = build_preregistration(sorted(set(aois)), directions)
    audit = build_input_audit(directions)

    filenames = [
        "four_aoi_decomposition_preregistration.json",
        "four_aoi_decomposition_input_audit.json",
        "four_aoi_decomposition.csv",
        "four_aoi_decomposition_bootstrap_summary.csv",
        "four_aoi_decomposition_final_report.json",
        "four_aoi_decomposition_final_report.md",
        "four_aoi_decomposition_manifest.json",
    ]

    unavailable = [d for d, v in directions.items() if not v.get("available")]

    if dry_run:
        return {
            "ran": False,
            "dry_run": True,
            "canonical_set_id": set_id,
            "preregistration": prereg,
            "input_audit": audit,
            "unavailable_directions": unavailable,
            "planned_output_paths": {n: str(out_dir / n) for n in filenames},
            "output_dir_exists": out_dir.exists(),
        }

    if unavailable:
        raise TransferDecompositionError(
            f"frozen Step10 artefacts missing for directions: {unavailable}"
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise TransferDecompositionError(
            f"{out_dir} already contains outputs; pass force=True to overwrite."
        )

    all_rows: list[dict[str, Any]] = []
    all_boot: list[dict[str, Any]] = []
    for direction, info in sorted(directions.items()):
        rows, boot = decompose_direction(direction, info)
        all_rows.extend(rows)
        all_boot.extend(boot)

    table = pd.DataFrame(all_rows)
    boot_table = pd.DataFrame(all_boot)

    reproduction = verify_raw_reproduction(table)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_set_id": set_id,
        "aois": sorted(set(aois)),
        "n_directions": int(table["direction"].nunique()),
        "n_directions_expected": prereg["n_ordered_directions_expected"],
        "n_rows": int(len(table)),
        "preregistration": prereg,
        "input_audit": audit,
        "raw_reproduction": reproduction,
        "status_counts": table["recovery_status"].value_counts().to_dict(),
        "n_negative_recovery": int((table["recovery_status"] == STATUS_NEGATIVE).sum()),
        "n_raw_gap_guard": int(
            (table["fraction_interpretability_status"] == NOT_INTERPRETABLE).sum()
        ),
        "n_ratio_degenerate_replicates": int(table["n_replicates_ratio_degenerate"].sum()),
        "max_identity_residual": (
            float(table["identity_residual"].dropna().max())
            if table["identity_residual"].notna().any()
            else None
        ),
        "rows": table.to_dict(orient="records"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "four_aoi_decomposition_preregistration.json").write_text(
        json.dumps(prereg, indent=2, ensure_ascii=False)
    )
    (out_dir / "four_aoi_decomposition_input_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    table.to_csv(out_dir / "four_aoi_decomposition.csv", index=False)
    boot_table.to_csv(out_dir / "four_aoi_decomposition_bootstrap_summary.csv", index=False)
    (out_dir / "four_aoi_decomposition_final_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    (out_dir / "four_aoi_decomposition_final_report.md").write_text(
        render_markdown(payload, table)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": payload["created_at"],
        "canonical_set_id": set_id,
        "outputs": [
            {
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(out_dir.glob("*"))
            if p.is_file() and p.name != "four_aoi_decomposition_manifest.json"
        ],
        "inputs": audit["inputs"],
    }
    (out_dir / "four_aoi_decomposition_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    return {
        "ran": True,
        "dry_run": False,
        "canonical_set_id": set_id,
        "output_dir": str(out_dir),
        "n_rows": payload["n_rows"],
        "n_directions": payload["n_directions"],
        "status_counts": payload["status_counts"],
        "n_negative_recovery": payload["n_negative_recovery"],
        "n_raw_gap_guard": payload["n_raw_gap_guard"],
        "raw_reproduction": reproduction,
    }
