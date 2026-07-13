"""
step10a_preregistration_and_audit.py

Step10A: Step10 ("unsupervised self-calibrated cross-region transfer")
denemesinin DEGISMEZ (immutable) on-kayit (preregistration) manifestini
olusturur/dogrular ve girdi/sema denetimini calistirir.

BU ASAMA:
    - hicbir model fit ETMEZ, hicbir tahmin URETMEZ
    - Step8/Step9 dosyalarini DEGISTIRMEZ (yalnizca salt-okunur okur +
      SHA256 hash'lerini kaydeder)
    - bir kez olusturulduktan sonra, `scientific_config` bloku BIR DAHA ASLA
      DEGISTIRILMEZ -- calisma-zamani (runtime) bilimsel ayarlari mevcut
      manifest ile UYUSMUYORSA fail-fast durur. Bilimsel bir degisiklik,
      YENI bir analiz versiyonu gerektirir (uzerine yazma DEGIL).
    - `--force`, YALNIZCA step10_input_audit.json'u (ve downstream Step10B-D
      ciktilarini) yeniler; on-kayit dosyalarina ASLA DOKUNMAZ.

analysis_id = SHA256(canonical_json(scientific_config)) -- her Step10
ciktisina GOMULUR (embed edilir).
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

from core.config import (
    STEP10_BOOTSTRAP_CI_LOWER_PERCENTILE,
    STEP10_BOOTSTRAP_CI_UPPER_PERCENTILE,
    STEP10_BOOTSTRAP_REPLICATES,
    STEP10_CORAL_LAMBDA,
    STEP10_MIN_VALID_BOOTSTRAP_REPLICATES,
    STEP10_RANDOM_STATE,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.step10_shared import (
    ADAPTATION_METHODS,
    CATEGORICAL_FEATURES,
    FEATURE_LISTS,
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_NAME,
    NUMERIC_FEATURE_POOL,
    PRIMARY_POPULATION,
    REGIONWISE_ZSCORE_METADATA_CLASS,
    Step10Error,
    assert_paths_are_safely_namespaced,
    canonical_json,
    compute_analysis_id,
    git_commit_if_available,
    package_versions,
    resolve_step8b_metrics_path,
    resolve_step8b_predictions_path,
    resolve_step9b_metrics_path,
    resolve_step9b_predictions_path,
    sha256_file,
    step10_output_dir,
)
from src.step8b_train_baseline_vs_thermal_model import build_classifier
from src.step9a_audit_cross_region_inputs import resolve_step8a_dataset_path

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step10a_preregistration_and_audit")


INTERPRETATION_RULES = {
    "raw_anti_predictive": (
        "If raw ROC-AUC point estimate < 0.5, say 'anti-predictive at the "
        "point-estimate level'. Only say bootstrap-supported below-chance if "
        "the full 95% CI is below 0.5."
    ),
    "covariate_recovery": (
        "Strong support if CI(z-score - raw) is entirely above 0. "
        "Above-chance support if the full z-score ROC-AUC CI is above 0.5. "
        "Otherwise report only point-estimate improvement."
    ),
    "residual_gap": (
        "If CI(within - adapted) is entirely above 0, report a supported "
        "residual performance gap after covariate adaptation. Call it "
        "consistent with remaining concept shift, not definitive proof by itself."
    ),
    "coral_vs_zscore": (
        "If CI(CORAL - z-score) includes 0, state that CORAL did not show "
        "supported improvement over simple region-wise standardization."
    ),
    "bidirectional_claim": (
        "Only call the pattern bidirectional if the same qualitative result "
        "occurs in both directions. Otherwise call it direction-dependent."
    ),
    "significance_wording": (
        "Never use p-values or 'statistically significant'. Use "
        "'bootstrap-supported' or 'the percentile interval excludes zero'."
    ),
}

PROHIBITED_ACTIONS = [
    "new_regions", "new_feature_subsets", "feature_selection", "hyperparameter_tuning",
    "new_model_types", "deep_adaptation", "target_label_calibration",
    "target_label_threshold_fitting", "prediction_inversion", "post_hoc_acceptance_criteria",
]


def build_scientific_config(source_id: str, target_id: str) -> dict:
    """Step10'un TUM bilimsel tasarimini (feature setleri, model
    hiperparametreleri, adaptasyon tanimlari, bootstrap konfigurasyonu,
    yorumlama kurallari) iceren, VERIYE BAGIMLI OLMAYAN (deterministic)
    canonical bir sozluk dondurur. analysis_id BUNDAN hesaplanir."""
    clf = build_classifier(MODEL_NAME, STEP10_RANDOM_STATE)
    model_hyperparameters = {k: v for k, v in clf.get_params().items()}

    return {
        "schema_version": "step10.v1",
        "source_experiment_id": source_id, "target_experiment_id": target_id,
        "directions": [f"{source_id}_to_{target_id}", f"{target_id}_to_{source_id}"],
        "primary_population": PRIMARY_POPULATION,
        "primary_estimand": (
            "delta_roc_auc_thermal = regionwise_zscore(thermal).roc_auc - "
            "raw_source_only(thermal).roc_auc"
        ),
        "secondary_estimands": [
            "delta_pr_auc_thermal = regionwise_zscore(thermal).pr_auc - raw_source_only(thermal).pr_auc",
            "delta_roc_auc_coral_minus_raw_thermal",
            "delta_roc_auc_coral_minus_zscore_thermal",
            "baseline_model_results (raw/zscore/coral, roc_auc + pr_auc)",
            "target_within_region_step8b_oof_metric_minus_adapted_transfer_metric",
        ],
        "baseline_numeric_features": [f for f in FEATURE_LISTS["baseline"] if f not in CATEGORICAL_FEATURES],
        "baseline_categorical_features": list(CATEGORICAL_FEATURES),
        "thermal_numeric_features": [f for f in FEATURE_LISTS["thermal"] if f not in CATEGORICAL_FEATURES],
        "thermal_categorical_features": list(CATEGORICAL_FEATURES),
        "regionwise_zscore_numeric_feature_pool": list(NUMERIC_FEATURE_POOL),
        "prohibited_leakage_columns": list(FORBIDDEN_MODEL_COLUMNS),
        "model_classes": {"baseline": "RandomForestClassifier", "thermal": "RandomForestClassifier"},
        "model_hyperparameters": model_hyperparameters,
        "random_state": STEP10_RANDOM_STATE,
        "adaptation_methods": {
            "raw_source_only": {
                "definition": (
                    "Reuses Step9B's exact preprocessing (build_pipeline: source-fitted "
                    "median imputer + one-hot encoder) and model fit; no additional "
                    "transform. Must reproduce Step9B metrics within 1e-6."
                ),
            },
            "regionwise_zscore": {
                "definition": "z = (x - region_mean) / region_std, computed independently per region.",
                "ddof": 0,
                "missing_value_rule": "fill with region's own feature mean before standardizing (becomes 0 after transform)",
                "zero_variance_rule": "if std < 1e-12, use divisor 1.0 and record a constant-feature guard",
                "clipping_or_winsorization": False,
                "categorical_handling": "untouched; source-fitted one-hot encoding preserved",
                "metadata_classification": REGIONWISE_ZSCORE_METADATA_CLASS,
                "never_uses_target_labels": True,
            },
            "coral_after_regionwise_zscore": {
                "definition": (
                    "CORAL (Sun & Saenko) applied to numeric features AFTER regionwise "
                    "z-score. Cs=cov(Xs_z)+lambda*I, Ct=cov(Xt_z)+lambda*I, "
                    "A=Cs^(-1/2) @ Ct^(1/2) via symmetric eigendecomposition, "
                    "Xs_coral = Xs_z @ A, Xt_coral = Xt_z (target unchanged)."
                ),
                "lambda": STEP10_CORAL_LAMBDA,
                "eigenvalue_floor": 1e-12,
                "categorical_handling": "excluded from covariance alignment",
                "never_uses_target_labels": True,
            },
        },
        "adaptation_method_names": list(ADAPTATION_METHODS),
        "threshold_policy": "none -- ROC-AUC and PR-AUC are threshold-free; no target threshold is fit.",
        "bootstrap": {
            "replicates": STEP10_BOOTSTRAP_REPLICATES,
            "ci_method": f"{STEP10_BOOTSTRAP_CI_LOWER_PERCENTILE}/{STEP10_BOOTSTRAP_CI_UPPER_PERCENTILE} percentile",
            "random_state": STEP10_RANDOM_STATE,
            "min_valid_replicates": STEP10_MIN_VALID_BOOTSTRAP_REPLICATES,
            "resampling_unit": "target spatial_block_id (existing, not redefined)",
            "pairing": "identical sampled target blocks across all methods within a replicate",
            "invalid_replicate_rule": "single-class target y invalidates the replicate for ALL methods jointly",
        },
        "interpretation_rules": INTERPRETATION_RULES,
        "prohibited_actions": PROHIBITED_ACTIONS,
        "not_probability_calibration": True,
        "not_operational_transfer_proof": True,
    }


def planned_output_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / "step10_preregistration.json", output_dir / "step10_preregistration.md",
        output_dir / "step10_input_audit.json", output_dir / "step10_adaptation_statistics.json",
        output_dir / "step10_predictions.parquet", output_dir / "step10_metrics.json",
        output_dir / "step10_metrics.csv", output_dir / "step10_bootstrap_replicates.parquet",
        output_dir / "step10_bootstrap_summary.json", output_dir / "step10_bootstrap_summary.csv",
        output_dir / "step10_decomposition.csv", output_dir / "step10_final_report.json",
        output_dir / "step10_final_report.md",
    ]


# =============================================================================
# Girdi/sema denetimi (input audit) -- salt-okunur, SHA256 hash'leri kaydeder
# =============================================================================
def run_input_audit(source_id: str, target_id: str) -> dict:
    audit: dict = {"created_at": datetime.now(timezone.utc).isoformat(), "checks": {}}

    for experiment_id in (source_id, target_id):
        step8a_path = resolve_step8a_dataset_path(experiment_id)
        step8b_pred_path = resolve_step8b_predictions_path(experiment_id)
        step8b_metrics_path = resolve_step8b_metrics_path(experiment_id)
        for label, path in (
            ("step8a_dataset", step8a_path), ("step8b_predictions", step8b_pred_path),
            ("step8b_metrics", step8b_metrics_path),
        ):
            assert_paths_are_safely_namespaced(experiment_id, experiment_id, path)
        audit["checks"][experiment_id] = {
            "step8a_dataset_path": str(step8a_path), "step8a_dataset_exists": step8a_path.exists(),
            "step8a_dataset_sha256": sha256_file(step8a_path),
            "step8b_predictions_path": str(step8b_pred_path), "step8b_predictions_exists": step8b_pred_path.exists(),
            "step8b_predictions_sha256": sha256_file(step8b_pred_path),
            "step8b_metrics_path": str(step8b_metrics_path), "step8b_metrics_exists": step8b_metrics_path.exists(),
            "step8b_metrics_sha256": sha256_file(step8b_metrics_path),
        }
        if step8a_path.exists():
            import pandas as pd
            schema_cols = set(pd.read_parquet(step8a_path).columns)
            # NOT: spatial_block_id burada ARANMAZ -- ham Step8A parquet
            # dosyasinda BULUNMAZ; load_step8a_dataset() tarafindan
            # row_500m/col_500m'den TUREME (add_spatial_block_id) olarak
            # calisma-zamaninda hesaplanir.
            required = {"cell_id", "row_500m", "col_500m", "valid_for_modeling", PRIMARY_POPULATION, "burned"} | set(FEATURE_LISTS["thermal"])
            missing = sorted(required - schema_cols)
            audit["checks"][experiment_id]["step8a_missing_required_columns"] = missing
            if missing:
                raise Step10Error(f"'{experiment_id}' Step8A veri setinde eksik zorunlu kolonlar: {missing}")

    step9b_metrics_path = resolve_step9b_metrics_path(source_id, target_id)
    step9b_pred_path = resolve_step9b_predictions_path(source_id, target_id)
    assert_paths_are_safely_namespaced(source_id, target_id, step9b_metrics_path)
    audit["checks"]["step9b_reference"] = {
        "metrics_path": str(step9b_metrics_path), "metrics_exists": step9b_metrics_path.exists(),
        "metrics_sha256": sha256_file(step9b_metrics_path),
        "predictions_path": str(step9b_pred_path), "predictions_exists": step9b_pred_path.exists(),
        "predictions_sha256": sha256_file(step9b_pred_path),
    }
    audit["package_versions"] = package_versions()
    return audit


# =============================================================================
# Orkestrasyon
# =============================================================================
def main(source_id: str, target_id: str, force: bool = False, dry_run: bool = False) -> dict:
    if source_id == target_id:
        raise Step10Error("--source ve --target ayni deney OLAMAZ.")

    output_dir = step10_output_dir(source_id, target_id)
    manifest_path = output_dir / "step10_preregistration.json"
    audit_path = output_dir / "step10_input_audit.json"

    scientific_config = build_scientific_config(source_id, target_id)
    computed_analysis_id = compute_analysis_id(scientific_config)

    if dry_run:
        log.info("[dry-run] Step10A: on-kayit + girdi denetimi ONIZLEME.")
        log.info("[dry-run] output_dir: %s", output_dir)
        log.info("[dry-run] computed analysis_id (henuz yazilmadi): %s", computed_analysis_id)
        log.info("[dry-run] primary_population: %s", scientific_config["primary_population"])
        log.info("[dry-run] adaptation_methods: %s", scientific_config["adaptation_method_names"])
        log.info("[dry-run] bootstrap: %s", scientific_config["bootstrap"])
        log.info("[dry-run] Hicbir dosya yazilmadi, hicbir model fit edilmedi.")
        return {"analysis_id": computed_analysis_id, "scientific_config": scientific_config, "dry_run": True}

    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_config = existing["scientific_config"]
        if canonical_json(existing_config) != canonical_json(scientific_config):
            raise Step10Error(
                "Step10 bilimsel konfigurasyonu, MEVCUT dondurulmus (frozen) "
                f"on-kayit ({manifest_path}) ile UYUSMUYOR. Bu, bilimsel bir "
                "degisikliktir; on-kayit UZERINE YAZILMAZ -- yeni bir acik "
                "analiz versiyonu (farkli output_dir) gereklidir."
            )
        analysis_id = existing["analysis_id"]
        log.info("Mevcut on-kayit (immutable) dogrulandi ve reuse edildi: analysis_id=%s", analysis_id)
    else:
        analysis_id = computed_analysis_id
        manifest = {
            "analysis_id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_if_available(), "scientific_config": scientific_config,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        _write_preregistration_md(manifest, output_dir)
        log.info("YENI on-kayit olusturuldu (immutable): analysis_id=%s (%s)", analysis_id, manifest_path)

    if not audit_path.exists() or force:
        audit = run_input_audit(source_id, target_id)
        audit["analysis_id"] = analysis_id
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.info("Girdi denetimi yazildi: %s", audit_path)
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        log.info("Girdi denetimi zaten var; --force verilmedigi icin atlaniyor.")

    return {"analysis_id": analysis_id, "scientific_config": scientific_config, "audit": audit, "output_dir": output_dir}


def _write_preregistration_md(manifest: dict, output_dir: Path) -> Path:
    cfg = manifest["scientific_config"]
    lines = [
        "# Step10 Preregistration (IMMUTABLE)", "",
        f"- analysis_id: `{manifest['analysis_id']}`",
        f"- created_at: {manifest['created_at']}",
        f"- git_commit: {manifest.get('git_commit')}",
        f"- source: `{cfg['source_experiment_id']}` / target: `{cfg['target_experiment_id']}`",
        f"- directions: {cfg['directions']}",
        f"- primary_population: `{cfg['primary_population']}`",
        "",
        "## Primary estimand", "", cfg["primary_estimand"], "",
        "## Secondary estimands", "",
    ]
    for e in cfg["secondary_estimands"]:
        lines.append(f"- {e}")
    lines.extend(["", "## Adaptation methods (fixed)", ""])
    for name, spec in cfg["adaptation_methods"].items():
        lines.append(f"### `{name}`")
        lines.append(spec.get("definition", ""))
        lines.append("")
    lines.extend([
        "## Bootstrap", "", f"- replicates: {cfg['bootstrap']['replicates']}",
        f"- CI: {cfg['bootstrap']['ci_method']}", f"- random_state: {cfg['bootstrap']['random_state']}",
        f"- min_valid_replicates: {cfg['bootstrap']['min_valid_replicates']}", "",
        "## Prohibited actions", "",
    ])
    for a in cfg["prohibited_actions"]:
        lines.append(f"- {a}")
    lines.extend([
        "", "## Note", "",
        "This preregistration is IMMUTABLE once created. `--force` may overwrite "
        "downstream Step10B-D outputs, but never this file or `analysis_id`. A "
        "scientific configuration change requires a new analysis (new output "
        "directory), not an overwrite.",
    ])
    path = output_dir / "step10_preregistration.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step10A: preregistration + input audit.")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(source_id=args.source, target_id=args.target, force=args.force, dry_run=args.dry_run)