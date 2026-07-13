"""
step10d_final_report.py

Step10D: Step10A (preregistration), Step10B (label-blind tahminler) ve
Step10C'nin (esli degerlendirme + bootstrap) ciktilarini SALT-OKUNUR olarak
okur; ONCEDEN KAYITLI (preregistered) yorumlama kurallarini uygulayarak
step10_final_report.json/md uretir.

Bu asama HICBIR HESAPLAMA/MODEL FIT YAPMAZ -- yalnizca Step10A-C'nin zaten
hesapladigi nokta-tahminleri ve bootstrap guven araliklarini, Step10A'da
ONCEDEN DONDURULMUS yorumlama kurallarina gore siniflandirir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.step10_shared import MODEL_FAMILIES, Step10Error, step10_output_dir

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step10d_final_report")

SAFE_WORDING = [
    "Step9 is the raw source-only transfer result; Step10 is unsupervised "
    "target-covariate adaptation informed by region-wise standardization and "
    "CORAL alignment.",
    "Target labels are not used for adaptation (fitting, imputation, "
    "covariance estimation, feature selection, method selection, threshold "
    "selection, or probability calibration).",
    "Step10 is not probability calibration.",
    "Step10 does not prove operational transfer.",
    "Residual performance gaps are consistent with remaining concept shift "
    "but are not by themselves a causal proof.",
]

NEVER_CLAIMS = [
    "statistical significance", "p-value", "causal fire-risk relationships",
    "operational wildfire prediction", "successful cross-region transfer proven",
    "Step9 was corrected", "CORAL definitively outperforms region-wise standardization",
]


# =============================================================================
# Yorumlama siniflandiricilari (Step10A'da ONCEDEN DONDURULMUS kurallar)
# =============================================================================
def _ci(bootstrap_ci: dict, key: str) -> dict:
    return bootstrap_ci.get(key, {"ci_2_5": None, "ci_97_5": None, "mean": None, "median": None})


def classify_raw_anti_predictive(raw_point_roc_auc: float | None, raw_ci: dict) -> str:
    lo, hi = raw_ci.get("ci_2_5"), raw_ci.get("ci_97_5")
    if hi is not None and hi < 0.5:
        return "bootstrap-supported below-chance (anti-predictive)"
    if raw_point_roc_auc is not None and raw_point_roc_auc < 0.5:
        return "anti-predictive at the point-estimate level"
    return "not below chance at the point-estimate level"


def classify_covariate_recovery(delta_zscore_minus_raw_ci: dict, zscore_roc_auc_ci: dict) -> str:
    lo = delta_zscore_minus_raw_ci.get("ci_2_5")
    if lo is not None and lo > 0:
        return "strong support (the percentile interval for z-score - raw excludes zero, entirely above 0)"
    z_lo = zscore_roc_auc_ci.get("ci_2_5")
    if z_lo is not None and z_lo > 0.5:
        return "above-chance support (full z-score ROC-AUC percentile interval above 0.5)"
    return "point-estimate improvement only (not bootstrap-supported)"


def classify_residual_gap(within_minus_adapted_ci: dict) -> str:
    lo = within_minus_adapted_ci.get("ci_2_5")
    if lo is not None and lo > 0:
        return (
            "supported residual performance gap after covariate adaptation, "
            "consistent with remaining concept shift or other non-covariate "
            "differences (not definitive proof by itself)"
        )
    return "residual gap not bootstrap-supported (the percentile interval crosses zero or is unavailable)"


def classify_coral_vs_zscore(coral_minus_zscore_ci: dict) -> str:
    lo, hi = coral_minus_zscore_ci.get("ci_2_5"), coral_minus_zscore_ci.get("ci_97_5")
    if lo is None or hi is None:
        return "insufficient bootstrap data"
    if lo <= 0 <= hi:
        return "CORAL did not show supported improvement over simple region-wise standardization (the percentile interval includes zero)"
    if lo > 0:
        return "bootstrap-supported improvement of CORAL over region-wise standardization"
    return "bootstrap-supported degradation of CORAL relative to region-wise standardization"


def classify_bidirectional(label_a: str, label_b: str) -> str:
    return "bidirectional" if label_a == label_b else "direction-dependent"


# =============================================================================
# Rapor uretimi
# =============================================================================
def build_final_report(source_id: str, target_id: str, analysis_id: str) -> dict:
    output_dir = step10_output_dir(source_id, target_id)
    metrics_path, bootstrap_summary_path, decomposition_path = (
        output_dir / "step10_metrics.json", output_dir / "step10_bootstrap_summary.json", output_dir / "step10_decomposition.csv",
    )
    for p in (metrics_path, bootstrap_summary_path, decomposition_path):
        if not p.exists():
            raise Step10Error(f"Step10D icin gerekli girdi bulunamadi: {p}. Once Step10A-C calistirilmali.")

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    bootstrap_payload = json.loads(bootstrap_summary_path.read_text(encoding="utf-8"))

    directions = list(metrics_payload["point_metrics"].keys())
    per_direction: dict = {}

    for direction in directions:
        point = metrics_payload["point_metrics"][direction]
        boot_ci = bootstrap_payload["by_direction"].get(direction, {}).get("ci", {})
        unstable = bootstrap_payload["by_direction"].get(direction, {}).get("bootstrap_unstable")

        per_model: dict = {}
        for model_family in MODEL_FAMILIES:
            raw_roc = point["raw_source_only"][model_family].get("roc_auc")
            raw_ci = _ci(boot_ci, f"roc_auc__raw_source_only_{model_family}")
            zscore_roc_ci = _ci(boot_ci, f"roc_auc__regionwise_zscore_{model_family}")
            delta_zscore_raw = _ci(boot_ci, f"delta_roc_auc__zscore_minus_raw__{model_family}")
            delta_coral_raw = _ci(boot_ci, f"delta_roc_auc__coral_minus_raw__{model_family}")
            delta_coral_zscore = _ci(boot_ci, f"delta_roc_auc__coral_minus_zscore__{model_family}")
            delta_within_zscore = _ci(boot_ci, f"delta_roc_auc__within_minus_zscore__{model_family}")
            delta_within_coral = _ci(boot_ci, f"delta_roc_auc__within_minus_coral__{model_family}")

            per_model[model_family] = {
                "raw_anti_predictive": classify_raw_anti_predictive(raw_roc, raw_ci),
                "covariate_recovery_zscore": classify_covariate_recovery(delta_zscore_raw, zscore_roc_ci),
                "residual_gap_zscore": classify_residual_gap(delta_within_zscore),
                "residual_gap_coral": classify_residual_gap(delta_within_coral),
                "coral_vs_zscore": classify_coral_vs_zscore(delta_coral_zscore),
                "delta_roc_auc_zscore_minus_raw_ci": delta_zscore_raw,
                "delta_roc_auc_coral_minus_raw_ci": delta_coral_raw,
                "delta_roc_auc_coral_minus_zscore_ci": delta_coral_zscore,
            }
        per_direction[direction] = {"bootstrap_unstable": unstable, "by_model_family": per_model}

    # --- Bidirectional / direction-dependent siniflandirmasi (primary estimand: thermal) ---
    bidirectional_summary = {}
    if len(directions) == 2:
        d0, d1 = directions
        for model_family in MODEL_FAMILIES:
            for key in ("covariate_recovery_zscore", "residual_gap_zscore", "residual_gap_coral", "coral_vs_zscore"):
                label_a = per_direction[d0]["by_model_family"][model_family][key]
                label_b = per_direction[d1]["by_model_family"][model_family][key]
                bidirectional_summary[f"{model_family}__{key}"] = classify_bidirectional(label_a, label_b)

    any_unstable = any(per_direction[d]["bootstrap_unstable"] for d in directions)

    report = {
        "analysis_id": analysis_id, "source_experiment_id": source_id, "target_experiment_id": target_id,
        "directions": directions, "safe_wording": SAFE_WORDING, "never_claims": NEVER_CLAIMS,
        "any_direction_bootstrap_unstable": any_unstable,
        "per_direction_interpretation": per_direction,
        "bidirectional_vs_direction_dependent": bidirectional_summary,
        "raw_reproduction": metrics_payload.get("raw_reproduction"),
        "within_region_reproduction": metrics_payload.get("within_region_reproduction"),
    }
    return report


def write_final_report_md(report: dict, output_dir: Path) -> Path:
    lines = [
        "# Step10 Final Report: Unsupervised Self-Calibrated Cross-Region Transfer", "",
        f"- analysis_id: `{report['analysis_id']}`",
        f"- source: `{report['source_experiment_id']}` / target: `{report['target_experiment_id']}`",
        f"- directions: {report['directions']}",
        f"- any_direction_bootstrap_unstable: **{report['any_direction_bootstrap_unstable']}**", "",
    ]
    for w in report["safe_wording"]:
        lines.append(f"> {w}")
        lines.append("")

    lines.extend(["## Per-direction interpretation", ""])
    for direction, d in report["per_direction_interpretation"].items():
        lines.append(f"### {direction}")
        lines.append(f"- bootstrap_unstable: {d['bootstrap_unstable']}")
        for model_family, cls in d["by_model_family"].items():
            lines.append(f"- **{model_family}**:")
            lines.append(f"  - raw_anti_predictive: {cls['raw_anti_predictive']}")
            lines.append(f"  - covariate_recovery (z-score - raw): {cls['covariate_recovery_zscore']}")
            lines.append(f"  - residual_gap (within - z-score): {cls['residual_gap_zscore']}")
            lines.append(f"  - residual_gap (within - CORAL): {cls['residual_gap_coral']}")
            lines.append(f"  - CORAL vs z-score: {cls['coral_vs_zscore']}")
        lines.append("")

    lines.extend(["## Bidirectional vs. direction-dependent", ""])
    for key, val in report["bidirectional_vs_direction_dependent"].items():
        lines.append(f"- {key}: **{val}**")

    lines.extend(["", "## Never claimed", ""])
    for c in report["never_claims"]:
        lines.append(f"- {c}")

    path = output_dir / "step10_final_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_step10d(source_id: str, target_id: str, analysis_id: str, force: bool = False) -> dict:
    output_dir = step10_output_dir(source_id, target_id)
    report_path = output_dir / "step10_final_report.json"
    if report_path.exists() and not force:
        log.info("Step10D ciktisi zaten var; --force verilmedigi icin atlaniyor.")
        return json.loads(report_path.read_text(encoding="utf-8"))

    report = build_final_report(source_id, target_id, analysis_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_final_report_md(report, output_dir)
    log.info("Step10D tamamlandi: %s", report_path)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step10D: final report (yorumlama kurallari, hesaplama YAPMAZ).")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--analysis-id", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_step10d(source_id=args.source, target_id=args.target, analysis_id=args.analysis_id, force=args.force)