"""
step9d_build_cross_region_report.py

Step9D: Step9A (audit) + Step9B (transfer metrics) + Step9C (bootstrap)
ciktilarini birlestirip, iki yonlu (bidirectional) bir cross-region transfer
degerlendirme raporu uretir.

Bu bir 30 m yangin tahmin modeli DEGILDIR, operasyonel bir yangin tespit
sistemi DEGILDIR, ve klasik istatistiksel anlamlilik iddiasi ICERMEZ.

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9d/final_cross_region_report.json
    outputs/cross_region/<source>__<target>/step9d/final_cross_region_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from src.step9a_audit_cross_region_inputs import PRIMARY_POPULATIONS, cross_region_output_root

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9d_build_cross_region_report")

PRIMARY_POPULATION = PRIMARY_POPULATIONS[0]

CAUTIOUS_STATEMENT = (
    "The thermal predictor set showed measurable cross-region transfer "
    "improvement supported by target-region spatial-block bootstrap."
)
CAUTION_NOTES = [
    "This is NOT a claim of operational wildfire prediction.",
    "This is NOT a claim of causal fire prediction.",
    "Labels are MCD64A1 ~500 m reconstructed cells, NOT 30 m fire labels.",
    "Bootstrap intervals are target-region spatial-block percentile "
    "intervals, NOT classical statistical significance.",
]


class Step9DError(SystemExit):
    """Fail-fast error for Step9D (diğer step'lerle aynı konvansiyon)."""


def _load_json(path: Path, required: bool = True) -> dict | None:
    if not path.exists():
        if required:
            raise Step9DError(f"Gerekli girdi bulunamadi: {path}. Onceki Step9 asamalarini calistirin.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_step9b_result(step9b_payload: dict, direction: str, population: str) -> dict | None:
    for r in step9b_payload.get("results", []):
        if r["transfer_direction"] == direction and r["population"] == population:
            return r
    return None


def _find_step9c_group(step9c_payload: dict, direction: str, population: str) -> dict | None:
    for g in step9c_payload.get("groups", []):
        if g["transfer_direction"] == direction and g["population"] == population:
            return g
    return None


def build_direction_summary(
    direction: str, step9b_payload: dict, step9c_payload: dict,
) -> dict:
    src_id, tgt_id = direction.split("_to_")
    primary_b = _find_step9b_result(step9b_payload, direction, PRIMARY_POPULATION)
    primary_c = _find_step9c_group(step9c_payload, direction, PRIMARY_POPULATION)

    population_results = {}
    for r in step9b_payload.get("results", []):
        if r["transfer_direction"] != direction:
            continue
        c = _find_step9c_group(step9c_payload, direction, r["population"])
        population_results[r["population"]] = {"transfer": r, "bootstrap": c}

    summary = {
        "transfer_direction": direction,
        "source_experiment_id": src_id,
        "target_experiment_id": tgt_id,
        "primary_population": PRIMARY_POPULATION,
        "population_results": population_results,
    }

    if primary_b is None or primary_b.get("skipped"):
        summary["primary_population_skipped"] = True
        summary["skip_reason"] = primary_b.get("reason") if primary_b else "no_step9b_result"
        summary["bootstrap_interpretation"] = {
            "delta_auc": "uncertain", "delta_pr_auc": "uncertain", "delta_brier": "uncertain",
        }
        return summary

    summary["primary_population_skipped"] = False
    summary["source_cell_count"] = primary_b["source_cell_count"]
    summary["target_cell_count"] = primary_b["target_cell_count"]
    summary["source_positive_count"] = primary_b["source_positive_count"]
    summary["target_positive_count"] = primary_b["target_positive_count"]
    summary["target_burned_prevalence"] = primary_b["target_burned_prevalence"]
    summary["baseline_target_metrics"] = primary_b["baseline_metrics"]
    summary["thermal_target_metrics"] = primary_b["thermal_metrics"]
    delta_metrics = dict(primary_b["delta_metrics"])
    if "brier_improvement" not in delta_metrics and delta_metrics.get("delta_brier") is not None:
        delta_metrics["brier_improvement"] = -delta_metrics["delta_brier"]
    summary["delta_metrics"] = delta_metrics

    if primary_c and primary_c.get("n_successful_replicates", 0) > 0:
        ci = primary_c["confidence_intervals"]
        if "brier_improvement" not in ci and "delta_brier" in ci:
            legacy = ci["delta_brier"]
            ci = dict(ci)
            ci["brier_improvement"] = {
                "ci_2_5": -legacy["ci_97_5"],
                "ci_97_5": -legacy["ci_2_5"],
                "mean": -legacy["mean"],
                "interpretation": legacy["interpretation"],
            }
        summary["bootstrap_confidence_intervals"] = ci
        summary["bootstrap_interpretation"] = {
            "delta_auc": ci["delta_roc_auc"]["interpretation"],
            "delta_pr_auc": ci["delta_pr_auc"]["interpretation"],
            "delta_brier": ci["delta_brier"]["interpretation"],
        }
    else:
        summary["bootstrap_confidence_intervals"] = None
        summary["bootstrap_interpretation"] = {
            "delta_auc": "uncertain", "delta_pr_auc": "uncertain", "delta_brier": "uncertain",
        }

    return summary


def classify_overall_conclusion(direction_summaries: list[dict]) -> tuple[str, str]:
    def fully_supported(d: dict) -> bool:
        interp = d.get("bootstrap_interpretation", {})
        return (
            interp.get("delta_auc") == "positive_bootstrap_support"
            and interp.get("delta_pr_auc") == "positive_bootstrap_support"
            and interp.get("delta_brier") == "positive_bootstrap_support"
        )

    def any_support(d: dict) -> bool:
        interp = d.get("bootstrap_interpretation", {})
        return any(v == "positive_bootstrap_support" for v in interp.values())

    fully = [fully_supported(d) for d in direction_summaries]
    any_sup = [any_support(d) for d in direction_summaries]

    if len(direction_summaries) >= 2 and all(fully):
        return "bidirectional_transfer_supported", (
            "Both transfer directions have delta_auc, delta_pr_auc CIs entirely "
            "above zero and delta_brier CI entirely below zero (target-region "
            "spatial-block bootstrap). " + CAUTIOUS_STATEMENT
        )
    if any(fully) or any(any_sup):
        return "partial_transfer_supported", (
            "Only one direction (or only some metrics) show positive "
            "target-region spatial-block bootstrap support for the thermal "
            "predictor set; the cross-region transfer result is mixed."
        )
    return "transfer_not_supported", (
        "Neither transfer direction shows positive target-region "
        "spatial-block bootstrap support for the thermal predictor set over "
        "the baseline."
    )


def build_report(source_id: str, target_id: str) -> dict:
    root = cross_region_output_root(source_id, target_id)
    step9a_payload = _load_json(root / "step9a" / "cross_region_input_audit.json")
    step9b_payload = _load_json(root / "step9b" / "cross_region_transfer_metrics.json")
    step9c_payload = _load_json(root / "step9c" / "cross_region_bootstrap_metrics.json")

    if not step9a_payload.get("passed"):
        raise Step9DError(
            f"Step9A audit passed=False; final rapor uretilemez. Detaylar: "
            f"{root / 'step9a' / 'cross_region_input_audit.json'}"
        )

    directions = sorted({r["transfer_direction"] for r in step9b_payload.get("results", [])})
    direction_summaries = [
        build_direction_summary(d, step9b_payload, step9c_payload) for d in directions
    ]

    conclusion, conclusion_text = classify_overall_conclusion(direction_summaries)

    report = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "directions_evaluated": directions,
        "primary_population": PRIMARY_POPULATION,
        "direction_summaries": direction_summaries,
        "overall_conclusion": conclusion,
        "overall_conclusion_text": conclusion_text,
        "cautious_statement_template": CAUTIOUS_STATEMENT,
        "caution_notes": CAUTION_NOTES,
        "reproducibility": {
            "git_commit": step9b_payload.get("git_commit"),
            "resolved_inputs": step9b_payload.get("resolved_inputs"),
            "model_name": step9b_payload.get("model_name"),
            "model_parameters": step9b_payload.get("model_parameters"),
            "preprocessing_parameters": step9b_payload.get("preprocessing_parameters"),
            "random_seed": step9b_payload.get("random_seed"),
            "spatial_cv_n_splits_requested": step9b_payload.get("spatial_cv_n_splits_requested"),
            "minimum_positives_and_negatives_per_population": step9b_payload.get(
                "minimum_positives_and_negatives_per_population"
            ),
            "baseline_features": step9b_payload.get("baseline_features"),
            "thermal_model_features": step9b_payload.get("thermal_model_features"),
            "population_definition": step9b_payload.get("population_definition"),
            "spatial_block_size_cells": step9b_payload.get("spatial_block_size_cells"),
            "spatial_block_definition": step9b_payload.get("spatial_block_definition"),
            "bootstrap_settings": {
                key: step9c_payload.get(key) for key in (
                    "n_bootstrap_replicates_requested", "random_seed",
                    "spatial_block_column", "resampling_unit", "resampling_scheme",
                    "percentile_interval", "max_attempts_multiplier",
                )
            },
        },
        "scope": {
            "evaluates": "Step8 ~500m MCD64A1-cell burned-area association model, cross-region transfer",
            "not_a_30m_fire_prediction_model": True,
            "not_an_operational_fire_detection_system": True,
            "does_not_transfer_step7_downscaling_model": True,
            "no_classical_statistical_significance_claimed": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_cross_region_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# Final Cross-Region Transfer Report",
        "",
        f"- source: `{report['source_experiment_id']}`",
        f"- target: `{report['target_experiment_id']}`",
        f"- primary population: `{report['primary_population']}`",
        "",
        f"**Overall conclusion: `{report['overall_conclusion']}`**",
        "",
        report["overall_conclusion_text"],
        "",
        "## Per-direction results (primary population)",
        "",
    ]
    for d in report["direction_summaries"]:
        lines.append(f"### {d['transfer_direction']}")
        lines.append("")
        if d.get("primary_population_skipped"):
            lines.append(f"- SKIPPED: {d.get('skip_reason')}")
            lines.append("")
            continue
        lines.extend([
            f"- source cells: {d['source_cell_count']} (positive: {d['source_positive_count']})",
            f"- target cells: {d['target_cell_count']} (positive: {d['target_positive_count']})",
            f"- target burned prevalence: {d['target_burned_prevalence']:.4f}",
            f"- baseline target ROC-AUC: {d['baseline_target_metrics']['roc_auc']}",
            f"- thermal target ROC-AUC: {d['thermal_target_metrics']['roc_auc']}",
            f"- baseline target PR-AUC: {d['baseline_target_metrics']['pr_auc']}",
            f"- thermal target PR-AUC: {d['thermal_target_metrics']['pr_auc']}",
            f"- baseline target Brier: {d['baseline_target_metrics']['brier_score']}",
            f"- thermal target Brier: {d['thermal_target_metrics']['brier_score']}",
            f"- delta_auc: {d['delta_metrics']['delta_auc']}",
            f"- delta_pr_auc: {d['delta_metrics']['delta_pr_auc']}",
            f"- delta_brier: {d['delta_metrics']['delta_brier']}",
            f"- brier_improvement (baseline - thermal): {d['delta_metrics']['brier_improvement']}",
            f"- bootstrap interpretation: {d['bootstrap_interpretation']}",
            "",
        ])

    lines.extend(["## Caution notes", ""])
    for note in report["caution_notes"]:
        lines.append(f"- {note}")

    md_path = output_dir / "final_cross_region_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main(source_id: str, target_id: str, force: bool = False) -> dict:
    output_dir = cross_region_output_root(source_id, target_id) / "step9d"
    json_path = output_dir / "final_cross_region_report.json"
    if json_path.exists() and not force:
        log.info("Step9D ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", json_path)
        return json.loads(json_path.read_text(encoding="utf-8"))

    report = build_report(source_id, target_id)
    json_path, md_path = write_report(report, output_dir)
    log.info(
        "Step9D tamamlandi: overall_conclusion=%s (%s, %s)",
        report["overall_conclusion"], json_path, md_path,
    )
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9D: Step9A-C ciktilarini birlestirip final cross-region "
        "transfer raporunu uretir."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(source_id=args.source, target_id=args.target, force=args.force)
