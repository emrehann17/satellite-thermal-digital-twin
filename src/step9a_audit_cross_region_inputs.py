"""
step9a_audit_cross_region_inputs.py

Step9A: iki bagimsiz deneyin (or. manavgat_2021 <-> bejis_2022) Step8A
500 m modeling dataset'lerini, cross-region transfer degerlendirmesine
(Step9B-D) GECMEDEN ONCE denetler (audit).

AMAC
----
Step9, Step8'in "burned-area association" modelini (baseline + baseline+
thermal) BIR bolgede egitip BASKA bir bolgede test ederek genelleme
kapasitesini olcer. Bu, ne 30 m'lik bir yangin tahmin modeli, ne
operasyonel bir yangin tespit sistemi, ne de Step7 downscaling modelinin
kendisinin transferidir -- yalnizca Step8'in ~500 m hucre-seviyesi
burned-area ASOSIASYON modelinin cross-region davranisidir.

Bu modul, iki veri setinin GERCEKTEN karsilastirilabilir olup olmadigini
egitim/transfer baslamadan ONCE dogrular; herhangi bir uyumsuzlukta
FAIL-FAST yapar.

CIKTILAR:
    outputs/cross_region/<source>__<target>/step9a/cross_region_input_audit.json
    outputs/cross_region/<source>__<target>/step9a/cross_region_input_audit.md

CLI:
    python src/step9a_audit_cross_region_inputs.py --source manavgat_2021 --target bejis_2022
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

import pandas as pd

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.regions import get_experiment, get_experiment_output_root

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step9a_audit_cross_region_inputs")


class Step9AError(SystemExit):
    """Fail-fast error for Step9A (diğer step'lerle aynı konvansiyon)."""


# =============================================================================
# Paylaşılan Step9 şeması (step9b/9c/9d bu sabitleri BURADAN import eder --
# tek kaynak, iki farklı/divergent tanım OLMAZ).
# =============================================================================
TARGET_COLUMN = "burned"
CELL_LEVEL_REQUIRED = "500m_reconstructed_mcd64a1_cell"
CATEGORICAL_FEATURES = ["landcover_dominant"]

SHARED_BASELINE_FEATURES = [
    "ndvi_mean",
    "elevation_mean",
    "slope_mean",
    "landcover_dominant",
]
SHARED_THERMAL_FEATURES = [
    "lst_anomaly_mean",
    "current_lst_mean",
    "current_tvdi_mean",
    "tvdi_difference_mean",
    "downscaled_lst_mean",
    "fused_lst_mean",
]
SHARED_THERMAL_MODEL_FEATURES = SHARED_BASELINE_FEATURES + SHARED_THERMAL_FEATURES

# Prompt'ta ("Forbidden model inputs") birebir listelenen kolonlar. Bu
# kolonlar model FEATURE SETLERINE asla giremez (dataset'te metadata/label
# olarak bulunmalari NORMALDIR -- yasak olan onlari X'e dahil etmektir).
FORBIDDEN_MODEL_COLUMNS = [
    "burned",
    "burn_date",
    "burn_month",
    "burn_day_of_year",
    "label_source",
    "cell_id",
    "row_500m",
    "col_500m",
    "lon",
    "lat",
    "experiment_id",
    "region_key",
    "spatial_block_id",
    "fold_id",
    "valid_for_modeling",
    "invalid_reason",
    "source_mask_majority",
    "observed_fraction",
    "gapfilled_fraction",
    "invalid_source_fraction",
    "valid_30m_fraction",
    "burn_date_pixel_agreement_fraction",
    "out_of_window_burndate",
]

# Populasyon maskesi olarak kullanilacak boolean kolonlar (Step8A'nin kendi
# ciktisinda zaten mevcut -- Step9B'de YENIDEN hesaplanmaz).
PRIMARY_POPULATIONS = ["burnable_tree_shrub_grass"]
SECONDARY_POPULATIONS = ["all_valid", "burnable_tree_shrub"]
ALL_POPULATIONS = PRIMARY_POPULATIONS + SECONDARY_POPULATIONS

MIN_POSITIVES_PER_REGION = 10
MIN_NEGATIVES_PER_REGION = 10


def cross_region_output_root(source_id: str, target_id: str) -> Path:
    return BASE_DIR / "outputs" / "cross_region" / f"{source_id}__{target_id}"


def resolve_step8a_dataset_path(experiment_id: str) -> Path:
    return get_experiment_output_root(experiment_id) / "step8a" / "step8a_500m_modeling_dataset.parquet"


def resolve_step8a_stats_path(experiment_id: str) -> Path:
    return get_experiment_output_root(experiment_id) / "step8a" / "step8a_dataset_stats.json"


def resolve_gate_path(experiment_id: str) -> Path:
    return get_experiment_output_root(experiment_id) / "validation" / "labels" / "burned_landcover_gate.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def audit_single_experiment(experiment_id: str) -> dict:
    """
    Tek bir deney icin tum girdi-uygunluk kontrollerini calistirir. Hicbir
    sey yazmaz; yalnizca bir sonuc/hata listesi dondurur (caller fail-fast
    kararini verir).
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []
    info: dict = {"experiment_id": experiment_id}

    exp = get_experiment(experiment_id)
    info["region_key"] = exp["region_key"]
    info["role"] = exp["role"]
    info["predictor_window"] = [exp["predictor_start_date"], exp["predictor_end_date"]]
    info["label_window"] = [exp["label_start_date"], exp["label_end_date"]]
    info["baseline_years"] = exp["baseline_years"]

    # --- predictor window bitisi label window baslangicindan ONCE mi? ---
    pred_end = exp["predictor_end_date"]
    label_start = exp["label_start_date"]
    checks["predictor_window_ends_before_label_window"] = bool(
        pred_end is not None and label_start is not None and pred_end < label_start
    )
    if not checks["predictor_window_ends_before_label_window"]:
        errors.append(
            f"'{experiment_id}': predictor_end_date ({pred_end}) label_start_date "
            f"({label_start})'tan ONCE degil."
        )

    # --- gate karari wildfire_candidate_pass mi? ---
    gate_path = resolve_gate_path(experiment_id)
    gate = _load_json(gate_path)
    info["gate_path"] = str(gate_path)
    info["gate_decision"] = gate.get("decision") if gate else None
    checks["gate_exists"] = gate is not None
    checks["gate_is_wildfire_candidate_pass"] = bool(gate and gate.get("decision") == "wildfire_candidate_pass")
    if not checks["gate_exists"]:
        errors.append(f"'{experiment_id}': burned-landcover gate sonucu bulunamadi ({gate_path}).")
    elif not checks["gate_is_wildfire_candidate_pass"]:
        errors.append(
            f"'{experiment_id}': gate karari 'wildfire_candidate_pass' DEGIL "
            f"(bulunan: {gate.get('decision')}). Cross-region transfer icin "
            "her iki bolge de gate'i gecmis olmalidir."
        )

    # --- Step8A stats: cell_level + no_30m_label_claim ---
    stats_path = resolve_step8a_stats_path(experiment_id)
    stats = _load_json(stats_path)
    info["step8a_stats_path"] = str(stats_path)
    checks["step8a_stats_exists"] = stats is not None
    if stats is not None:
        checks["cell_level_correct"] = stats.get("cell_level") == CELL_LEVEL_REQUIRED
        checks["no_30m_label_claim_true"] = stats.get("no_30m_label_claim") is True
        info["cell_level"] = stats.get("cell_level")
        info["no_30m_label_claim"] = stats.get("no_30m_label_claim")
        if not checks["cell_level_correct"]:
            errors.append(
                f"'{experiment_id}': cell_level='{stats.get('cell_level')}' "
                f"(beklenen: '{CELL_LEVEL_REQUIRED}')."
            )
        if not checks["no_30m_label_claim_true"]:
            errors.append(f"'{experiment_id}': no_30m_label_claim True degil.")
    else:
        checks["cell_level_correct"] = False
        checks["no_30m_label_claim_true"] = False
        errors.append(f"'{experiment_id}': Step8A stats dosyasi bulunamadi ({stats_path}).")

    # --- Step8A dataset: var mi, gerekli kolonlar, MCD64A1 primary label ---
    dataset_path = resolve_step8a_dataset_path(experiment_id)
    info["dataset_path"] = str(dataset_path)
    checks["dataset_exists"] = dataset_path.exists()
    if not checks["dataset_exists"]:
        errors.append(f"'{experiment_id}': Step8A modeling dataset bulunamadi ({dataset_path}).")
        info["row_count"] = None
        info["positive_count"] = None
        info["negative_count"] = None
        return {"checks": checks, "errors": errors, "info": info}

    df = pd.read_parquet(dataset_path)
    info["row_count"] = int(len(df))

    required_columns = list(dict.fromkeys(
        SHARED_BASELINE_FEATURES + SHARED_THERMAL_FEATURES + [
            TARGET_COLUMN, "row_500m", "col_500m", "cell_id", "valid_for_modeling",
            "label_source",
        ] + [p for p in ALL_POPULATIONS if p != "all_valid"]
    ))
    missing_columns = [c for c in required_columns if c not in df.columns]
    checks["required_shared_columns_present"] = len(missing_columns) == 0
    info["missing_columns"] = missing_columns
    if missing_columns:
        errors.append(f"'{experiment_id}': gerekli kolonlar eksik: {missing_columns}.")

    checks["target_column_is_burned"] = TARGET_COLUMN in df.columns
    if not checks["target_column_is_burned"]:
        errors.append(f"'{experiment_id}': hedef kolon '{TARGET_COLUMN}' bulunamadi.")

    # --- primary label kaynagi MCD64A1 mi? ---
    if "label_source" in df.columns and len(df) > 0:
        sources = set(df["label_source"].dropna().unique().tolist())
        checks["primary_label_is_mcd64a1"] = sources.issubset({"MCD64A1"}) if sources else True
        info["label_sources_seen"] = sorted(sources)
        if not checks["primary_label_is_mcd64a1"]:
            errors.append(f"'{experiment_id}': label_source MCD64A1 disinda deger iceriyor: {sources}.")
    else:
        checks["primary_label_is_mcd64a1"] = True
        info["label_sources_seen"] = []

    # --- spatial_block_id hesaplanabilir mi? (row_500m/col_500m mevcut) ---
    checks["spatial_block_id_computable"] = (
        "row_500m" in df.columns and "col_500m" in df.columns
    )
    if not checks["spatial_block_id_computable"]:
        errors.append(f"'{experiment_id}': row_500m/col_500m eksik; spatial_block_id hesaplanamaz.")

    # --- forbidden kolonlar model feature setine SIZMAMIS mi? (static self-check) ---
    leaked = set(SHARED_THERMAL_MODEL_FEATURES).intersection(FORBIDDEN_MODEL_COLUMNS)
    checks["no_forbidden_columns_in_feature_sets"] = len(leaked) == 0
    if leaked:
        errors.append(f"'{experiment_id}': YASAK kolonlar feature setine sizmis: {leaked}.")

    # --- her populasyon icin yeterli pozitif/negatif hucre var mi? ---
    pop_counts: dict[str, dict] = {}
    valid_df = df[df["valid_for_modeling"] == True] if "valid_for_modeling" in df.columns else df  # noqa: E712
    for pop in ALL_POPULATIONS:
        if pop == "all_valid":
            pop_df = valid_df
        elif pop in df.columns:
            pop_df = valid_df[valid_df[pop].astype(bool)]
        else:
            pop_df = valid_df.iloc[0:0]
        n_pos = int((pop_df[TARGET_COLUMN] == 1).sum()) if TARGET_COLUMN in pop_df.columns else 0
        n_neg = int((pop_df[TARGET_COLUMN] == 0).sum()) if TARGET_COLUMN in pop_df.columns else 0
        sufficient = n_pos >= MIN_POSITIVES_PER_REGION and n_neg >= MIN_NEGATIVES_PER_REGION
        pop_counts[pop] = {"positive_count": n_pos, "negative_count": n_neg, "sufficient": sufficient}
    info["population_counts"] = pop_counts
    checks["primary_population_sufficient"] = all(
        pop_counts[p]["sufficient"] for p in PRIMARY_POPULATIONS if p in pop_counts
    )
    if not checks["primary_population_sufficient"]:
        errors.append(
            f"'{experiment_id}': birincil populasyon(lar) ({PRIMARY_POPULATIONS}) icin "
            f"yeterli pozitif/negatif hucre yok (>= {MIN_POSITIVES_PER_REGION} her biri): "
            f"{pop_counts}."
        )

    # --- TVDI bu bolge icin AYRI hesaplanmis mi? (dosya bazli kanit: Step5C
    # kendi namespaced dizininde calisti mi -- current_tvdi_mean kolonunun
    # varliği + NaN olmayan degerler tasimasi dolayli kanittir) ---
    if "current_tvdi_mean" in df.columns:
        checks["tvdi_computed_for_region"] = bool(df["current_tvdi_mean"].notna().any())
    else:
        checks["tvdi_computed_for_region"] = False
    if not checks["tvdi_computed_for_region"]:
        errors.append(f"'{experiment_id}': current_tvdi_mean tamamen eksik/NaN -- TVDI bu bolge icin hesaplanmamis olabilir.")

    return {"checks": checks, "errors": errors, "info": info}


def audit_pair(source_id: str, target_id: str) -> dict:
    """Iki deneyi (source + target) denetler ve birlesik bir sonuc dondurur."""
    log.info("Step9A audit basliyor: source=%s, target=%s", source_id, target_id)

    source_result = audit_single_experiment(source_id)
    target_result = audit_single_experiment(target_id)

    all_errors = list(source_result["errors"]) + list(target_result["errors"])

    # --- shared features gercekten HER IKI veri setinde de mevcut mu? ---
    shared_ok = (
        not source_result["info"].get("missing_columns")
        and not target_result["info"].get("missing_columns")
    )
    if not shared_ok:
        all_errors.append(
            "Paylasilan zorunlu ozellikler her iki veri setinde de mevcut degil "
            f"(source eksik: {source_result['info'].get('missing_columns')}, "
            f"target eksik: {target_result['info'].get('missing_columns')})."
        )

    passed = len(all_errors) == 0

    result = {
        "source_experiment_id": source_id,
        "target_experiment_id": target_id,
        "shared_baseline_features": SHARED_BASELINE_FEATURES,
        "shared_thermal_features": SHARED_THERMAL_FEATURES,
        "shared_thermal_model_features": SHARED_THERMAL_MODEL_FEATURES,
        "forbidden_model_columns": FORBIDDEN_MODEL_COLUMNS,
        "primary_populations": PRIMARY_POPULATIONS,
        "secondary_populations": SECONDARY_POPULATIONS,
        "target_column": TARGET_COLUMN,
        "cell_level_required": CELL_LEVEL_REQUIRED,
        "source": source_result,
        "target": target_result,
        "shared_features_present_in_both": shared_ok,
        "errors": all_errors,
        "passed": passed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def write_audit_outputs(result: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cross_region_input_audit.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# Cross-Region Input Compatibility Audit",
        "",
        f"- source: `{result['source_experiment_id']}`",
        f"- target: `{result['target_experiment_id']}`",
        f"- **passed: `{result['passed']}`**",
        "",
        "This audit checks whether two independently generated Step8A ~500 m "
        "MCD64A1-cell modeling datasets are compatible for cross-region "
        "association-model transfer. It does NOT train or evaluate any model.",
        "",
        "## Per-experiment checks",
        "",
        "| check | source | target |",
        "|---|---|---|",
    ]
    all_check_names = list(result["source"]["checks"].keys())
    for name in all_check_names:
        s = result["source"]["checks"].get(name)
        t = result["target"]["checks"].get(name)
        lines.append(f"| {name} | {s} | {t} |")

    lines.extend(["", "## Population counts (positive/negative cells)", ""])
    lines.append("| population | source pos/neg | target pos/neg |")
    lines.append("|---|---|---|")
    for pop in ALL_POPULATIONS:
        sp = result["source"]["info"].get("population_counts", {}).get(pop, {})
        tp = result["target"]["info"].get("population_counts", {}).get(pop, {})
        lines.append(
            f"| {pop} | {sp.get('positive_count')}/{sp.get('negative_count')} | "
            f"{tp.get('positive_count')}/{tp.get('negative_count')} |"
        )

    if result["errors"]:
        lines.extend(["", "## Errors", ""])
        for e in result["errors"]:
            lines.append(f"- {e}")

    lines.extend([
        "", "## Scope note", "",
        "This is a cross-region transfer of the Step8 ~500 m MCD64A1-cell "
        "burned-area **association** model. It is NOT a 30 m fire prediction "
        "model, NOT an operational fire detection system, and does NOT "
        "transfer the Step7 downscaling model itself.",
    ])
    md_path = output_dir / "cross_region_input_audit.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main(source_id: str, target_id: str, force: bool = False) -> dict:
    output_dir = cross_region_output_root(source_id, target_id) / "step9a"
    json_path = output_dir / "cross_region_input_audit.json"
    if json_path.exists() and not force:
        log.info("Step9A ciktisi zaten var (%s); --force verilmedigi icin atlaniyor.", json_path)
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if not existing.get("passed"):
            raise Step9AError(
                f"Var olan Step9A audit sonucu passed=False ({json_path}). "
                "--force ile yeniden calistirin veya girdileri duzeltin."
            )
        return existing

    result = audit_pair(source_id, target_id)
    json_path, md_path = write_audit_outputs(result, output_dir)
    log.info("Step9A tamamlandi: passed=%s (%s, %s)", result["passed"], json_path, md_path)

    if not result["passed"]:
        raise Step9AError(
            f"Step9A cross-region input audit BASARISIZ ({len(result['errors'])} hata). "
            f"Detaylar: {json_path}. Ilk hatalar: {result['errors'][:5]}"
        )
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step9A: iki deneyin Step8A veri setlerini cross-region "
        "transfer icin denetler (audit). Model EGITMEZ, tahmin YAPMAZ."
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(source_id=args.source, target_id=args.target, force=args.force)