"""
run_step7_downscaling_only.py

Step0E: deney-farkinda (experiment-aware) Step7A-E (MODIS->Landsat LST
downscaling + fusion) calistiricisi.

Yalnizca Step7A-E'yi calistirir. Step8 (burned-area/fire-risk modelleme)
KESINLIKLE CALISTIRILMAZ, hicbir yangin/burned-area modeli EGITILMEZ.

IKI FARKLI DAVRANIS MODU
-------------------------
kozan_2023:
    Legacy davranis BIREBIR korunur. src/step7a-e_*.py'nin ctx-farkinda
    run_step7X(ctx=None) fonksiyonlari, ctx=None oldugunda mevcut main()'i
    degistirmeden cagirir; tum girdi/cikti yollari core/config.py'nin legacy
    sabitleridir (outputs/step7a .. outputs/step7e, outputs/step5,
    outputs/step5c, data/...).

kozan_2023 DISINDAKI her deney (or. manavgat_2021):
    TAMAMEN NAMESPACED calisir:
        outputs/experiments/<experiment_id>/step7a/
        outputs/experiments/<experiment_id>/step7b/
        outputs/experiments/<experiment_id>/step7c/
        outputs/experiments/<experiment_id>/step7d/
        outputs/experiments/<experiment_id>/step7e/
    Girdiler:
        - Step5/Step5C: ctx["step5_output_dir"] / ctx["step5c_output_dir"]
          (namespaced, ASLA outputs/step5 veya outputs/step5c okunmaz).
        - DEM (elevation/slope): BILEREK PAYLASILAN/salt-okunur global asset
          (Option B) -- data/dem/elevation.tif, data/dem/slope.tif. Manavgat
          icin ayri bir DEM export'u bu yamada YAPILMAZ; Manavgat-turetilmis
          hicbir DEM ciktisi bu paylasilan dizine YAZILMAZ (yalnizca okuma).
        - Landcover: Step6A gate-input asamasinda zaten uretilmis, referans
          gride hizali landcover (ctx["landcover_aligned_path"]) reuse edilir.
        - MODIS: Kozan-dışı deneyler için (bu düzeltmeden itibaren) MODIS
          mean/std ZORUNLUDUR -- Step7 amacı MODIS->Landsat downscaling/fusion
          ürettiği için, eksik MODIS'i sessizce atlamak GÜVENSİZDİR. Bu
          script önce namespaced MODIS export'unu (best-effort) DENER
          (core/step2_modis_5year_mean.py:process_summer_mean() +
          run_predictors_only.py:export_image_direct_or_tiled() reuse
          edilerek). Export sonrası hâlâ MODIS mean/std yoksa VE
          `--allow-no-modis` VERİLMEDİYSE, Step7B/Step7D çalıştırılmadan ÖNCE
          net bir hata ile durur. `--allow-no-modis` açıkça verilirse (yalnızca
          diagnostic/no-downscaling modu için), zincir MODIS'siz devam eder
          ve ürettiği metadata'ya `modis_available=false`,
          `downscaling_mode="no_modis_diagnostic"` ve açık bir uyarı
          ("bu geçerli bir MODIS->Landsat downscaling koşusu DEĞİLDİR")
          yazılır.

Kozan icin MODIS legacy davranışı (opsiyonel, required=False) DEĞİŞMEDİ --
bu kısıtlama yalnızca Kozan-dışı deneyler içindir.

Her Kozan-disi calistirmadan once, TUM hesaplanan yollarin
outputs/experiments/<experiment_id>/ altinda kaldigi VE legacy Kozan paylasilan
dizinleriyle (outputs/step7a..e, outputs/step5, outputs/step5c, data/modis)
CAKISMADIGI dogrulanir. Ihlalde hicbir adim calismaz.

CLI:
    python scripts/run_step7_downscaling_only.py --experiment kozan_2023 --dry-run
    python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 --dry-run
    python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 --force
    python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 \
        --from-step step7a --to-step step7e --force
    python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 \
        --force --allow-no-modis   # yalnızca diagnostic/no-downscaling modu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context, get_region, log_context_summary
from core.io_utils import setup_logger

log, log_file = setup_logger("run_step7_downscaling_only")

BASE_DIR = _PROJECT_ROOT

STEP_ORDER = ["step7a", "step7b", "step7c", "step7d", "step7e"]

# Legacy Kozan paylaşılan dizinleri: Kozan-dışı deneyler bunlara ASLA yazamaz/okuyamaz.
_LEGACY_SHARED_DIRS = [
    (BASE_DIR / "outputs" / "step7a").resolve(),
    (BASE_DIR / "outputs" / "step7b").resolve(),
    (BASE_DIR / "outputs" / "step7c").resolve(),
    (BASE_DIR / "outputs" / "step7d").resolve(),
    (BASE_DIR / "outputs" / "step7e").resolve(),
    (BASE_DIR / "outputs" / "step5").resolve(),
    (BASE_DIR / "outputs" / "step5c").resolve(),
    (BASE_DIR / "data" / "modis").resolve(),
    (BASE_DIR / "data" / "dem").resolve() if False else None,  # bkz. NOT asagida
]
_LEGACY_SHARED_DIRS = [d for d in _LEGACY_SHARED_DIRS if d is not None]
# NOT: data/dem/ BILEREK bu listeye eklenmedi -- Option B geregi bu dizin
# Kozan/Manavgat arasinda PAYLASILAN, SALT-OKUNUR bir kaynaktir; iki deney de
# ORAYA sadece okur, hicbiri oraya YAZMAZ (bkz. asagidaki yazma-yolu kontrolu
# zaten yalnizca ctx'in kendi cikti dizinlerini kontrol eder, dem_input_dir'i
# DEGIL).


class Step7RunnerError(SystemExit):
    """Fail-fast error for this runner (diğer step'lerle aynı konvansiyon)."""


_STEP_METADATA_FILENAMES = {
    "step7a": "tiling_test_summary.json",
    "step7b": "downscaling_dataset_stats.json",
    "step7c": "downscaling_model_metadata.json",
    "step7d": "downscaling_prediction_metadata.json",
    "step7e": "fused_lst_metadata.json",
}


def _enrich_metadata_with_experiment_context(metadata_path: Path, ctx: dict) -> None:
    """
    Bir step'in kendi ürettiği metadata JSON dosyasına deney bağlamını
    (experiment_id, region_key, role, predictor window, baseline years) ekler.

    Mevcut metadata şemasını BOZMAZ; yalnızca eksik anahtarları ekler (var
    olan bir anahtara asla dokunmaz). Kozan için (ctx=None) hiçbir şey
    yapmaz -- legacy metadata dosyaları BİREBİR korunur.
    """
    if ctx is None or not metadata_path.exists():
        return
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    changed = False
    for key, value in [
        ("experiment_id", ctx["experiment_id"]),
        ("region_key", ctx["region_key"]),
        ("role", ctx["role"]),
        ("predictor_window", [ctx["predictor_start_date"], ctx["predictor_end_date"]]),
        ("baseline_years", ctx["baseline_years"]),
    ]:
        if key not in data:
            data[key] = value
            changed = True
    if changed:
        metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Metadata deney bağlamıyla zenginleştirildi: %s", metadata_path)


def _enrich_metadata_with_modis_status(metadata_path: Path, modis_requirement: dict) -> None:
    """
    Step7B/7C/7D/7E metadata'sına MODIS zorunluluk-kontrolü sonucunu ekler:
        modis_available: bool | None
        downscaling_mode: "modis_downscaling" | "no_modis_diagnostic" | None
        (varsa) warning: "No MODIS input; this is not a valid ... run."

    Kozan icin (modis_requirement["checked"]=False, downscaling_mode=None)
    hicbir sey YAZMAZ -- legacy metadata semasi BOZULMAZ.
    """
    if not modis_requirement.get("checked") or not metadata_path.exists():
        return
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    data["modis_available"] = modis_requirement.get("modis_available")
    data["downscaling_mode"] = modis_requirement.get("downscaling_mode")
    if modis_requirement.get("warning"):
        data["warning"] = modis_requirement["warning"]
    metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _step7_output_dirs(ctx: dict) -> dict:
    return {
        "step7a": ctx["step7a_output_dir"],
        "step7b": ctx["step7b_output_dir"],
        "step7c": ctx["step7c_output_dir"],
        "step7d": ctx["step7d_output_dir"],
        "step7e": ctx["step7e_output_dir"],
    }


def _assert_paths_are_safely_namespaced(ctx: dict) -> None:
    experiment_id = ctx["experiment_id"]
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    check_paths = {
        **_step7_output_dirs(ctx),
        "step5_dir_read": ctx["step5_output_dir"],
        "step5c_dir_read": ctx["step5c_output_dir"],
        "modis_input_dir": ctx["modis_input_dir"],
    }
    for key, p in check_paths.items():
        resolved = Path(p).resolve()
        for legacy_dir in _LEGACY_SHARED_DIRS:
            if resolved == legacy_dir or legacy_dir in resolved.parents:
                raise Step7RunnerError(
                    f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                    f"({resolved}) Kozan'ın legacy paylaşılan dizinine ({legacy_dir}) "
                    "düşüyor. Bu deney bu dizine ASLA yazamaz/okuyamaz. İşlem DURDURULDU."
                )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise Step7RunnerError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için '{key}' yolu "
                f"({resolved}) outputs/experiments/{experiment_id}/ dışında. "
                "İşlem DURDURULDU."
            )


def _log_planned_paths(ctx: dict) -> None:
    dirs = _step7_output_dirs(ctx)
    for name, d in dirs.items():
        log.info("  %s_output_dir: %s", name, d)
    log.info("  step5_dir (okuma): %s", ctx["step5_output_dir"])
    log.info("  step5c_dir (okuma): %s", ctx["step5c_output_dir"])
    log.info(
        "  dem_input_dir (%s): %s",
        "Kozan legacy, paylaşılan/salt-okunur" if ctx["is_kozan"] else "namespaced",
        ctx["dem_input_dir"],
    )
    log.info(
        "  landcover_aligned_path: %s",
        ctx["landcover_aligned_path"] if ctx["landcover_aligned_path"] else "(Kozan legacy keşif)",
    )
    log.info("  modis_input_dir: %s", ctx["modis_input_dir"])


def _dry_run_input_check(ctx: dict, allow_no_modis: bool = False) -> None:
    log.info("Beklenen girdiler:")
    step5_files = [
        "current_period_median_celsius.tif",
        "anomaly_zscore.tif",
        "baseline_lst_mean_celsius.tif",
        "baseline_lst_std_celsius.tif",
    ]
    for name in step5_files:
        p = Path(ctx["step5_output_dir"]) / name
        log.info("  %s %s (step5)", "[VAR]" if p.exists() else "[EKSİK]", name)

    step5c_files = ["current_tvdi.tif", "tvdi_difference.tif"]
    for name in step5c_files:
        p = Path(ctx["step5c_output_dir"]) / name
        log.info("  %s %s (step5c)", "[VAR]" if p.exists() else "[EKSİK]", name)

    modis_mean = Path(ctx["modis_input_dir"]) / "modis_lst_mean_celsius.tif"
    modis_std = Path(ctx["modis_input_dir"]) / "modis_lst_std_celsius.tif"
    modis_available = modis_mean.exists() and modis_std.exists()

    if ctx["is_kozan"]:
        log.info(
            "  MODIS: mean=%s std=%s (Kozan legacy: opsiyonel/required=False -- "
            "eksikse Step7B/7D sessizce bu özellikleri atlar)",
            "[VAR]" if modis_mean.exists() else "[EKSİK]",
            "[VAR]" if modis_std.exists() else "[EKSİK]",
        )
    else:
        log.info("  MODIS required=True for %s (Kozan-dışı deneyler için ZORUNLU)", ctx["experiment_id"])
        log.info(
            "  MODIS: mean=%s std=%s",
            "[VAR]" if modis_mean.exists() else "[EKSİK]",
            "[VAR]" if modis_std.exists() else "[EKSİK]",
        )
        if modis_available:
            log.info("  -> Gerçek çalıştırma (--force) MODIS ile devam edecek.")
        elif allow_no_modis:
            log.info(
                "  -> MODIS EKSİK, ama --allow-no-modis verildi: gerçek çalıştırma "
                "'no_modis_diagnostic' modunda (downscaling GEÇERSİZ) devam edecek."
            )
        else:
            log.info(
                "  -> MODIS EKSİK ve --allow-no-modis VERİLMEDİ: gerçek çalıştırma "
                "(--force) Step7B/7D'den ÖNCE FAIL-FAST ile DURACAK. Önce namespaced "
                "MODIS export edin, veya yalnızca diagnostic/no-downscaling modu için "
                "--allow-no-modis ekleyin."
            )

    dem_elev = Path(ctx["dem_input_dir"]) / "elevation.tif"
    dem_slope = Path(ctx["dem_input_dir"]) / "slope.tif"
    dem_available = dem_elev.exists() and dem_slope.exists()

    if ctx["is_kozan"]:
        log.info(
            "  DEM (paylaşılan/salt-okunur): elevation=%s slope=%s",
            "[VAR]" if dem_elev.exists() else "[EKSİK]",
            "[VAR]" if dem_slope.exists() else "[EKSİK]",
        )
    else:
        log.info(
            "  DEM required=True for %s (namespaced, dem_input_dir=%s)",
            ctx["experiment_id"], ctx["dem_input_dir"],
        )
        log.info(
            "  DEM: elevation=%s slope=%s",
            "[VAR]" if dem_elev.exists() else "[EKSİK]",
            "[VAR]" if dem_slope.exists() else "[EKSİK]",
        )
        if dem_available:
            log.info("  -> Gerçek çalıştırma (--force) namespaced DEM ile devam edecek.")
        else:
            log.info(
                "  -> DEM EKSİK: gerçek çalıştırma (--force) Step7B'den ÖNCE FAIL-FAST "
                "ile DURACAK. Önce: python scripts/prepare_dem_for_experiment.py "
                f"--experiment {ctx['experiment_id']} --export --force"
            )

    if ctx["landcover_aligned_path"] is not None:
        lc = Path(ctx["landcover_aligned_path"])
        log.info(
            "  Landcover (Step6A gate-input): %s %s",
            "[VAR]" if lc.exists() else "[EKSİK]", lc,
        )
    else:
        log.info("  Landcover: Kozan legacy keşif (data/landcover/...)")


# =============================================================================
# MODIS hazırlığı + zorunluluk kontrolü -- Manavgat (Kozan-dışı) için
# =============================================================================
def prepare_manavgat_modis_context(ctx: dict, force: bool = False) -> dict:
    """
    Manavgat (Kozan-dışı) için MODIS LST mean/std'yi (deneyin PREDICTOR
    penceresi için) export etmeyi DENER (best-effort). BASARISIZ olursa hata
    FIRLATMAZ -- yalnizca uyari loglar ve {"status": "failed", ...} doner;
    NIHAI zorunluluk kontrolu (_require_modis_or_fail) BU fonksiyonun
    DISINDA, cagiran main() icinde yapilir -- boylece export basarisiz olsa
    bile mean/std dosyalari baska bir yoldan (or.
    scripts/prepare_modis_for_step7.py ile ayrica calistirilmis) zaten
    mevcutsa akis dogru sekilde devam edebilir.

    Asil mantik artik scripts/prepare_modis_for_step7.py:prepare_modis_for_step7()
    icinde yasiyor -- burasi yalnizca onu (best-effort/try-except ile
    sarmalanmis olarak) cagiran ince bir koprudur; iki farkli/divergent MODIS
    hazirlama implementasyonu YOKTUR.
    """
    from scripts.prepare_modis_for_step7 import prepare_modis_for_step7

    try:
        return prepare_modis_for_step7(ctx, force=force)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "MODIS export başarısız/atlandı: %s: %s", type(exc).__name__, exc,
        )
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


_MODIS_MISSING_MESSAGE = (
    "Manavgat MODIS inputs are missing. Prepare/export MODIS for this "
    "experiment before Step7, or run with --allow-no-modis only for "
    "diagnostic/no-downscaling mode."
)


def _modis_available(ctx: dict) -> bool:
    mean_path = ctx["modis_input_dir"] / "modis_lst_mean_celsius.tif"
    std_path = ctx["modis_input_dir"] / "modis_lst_std_celsius.tif"
    return mean_path.exists() and std_path.exists()


def _require_modis_or_fail(ctx: dict, steps_to_run: list[str], allow_no_modis: bool) -> dict:
    """
    Kozan-dışı deneyler için MODIS mean/std'yi Step7B/Step7D çalışacaksa
    ZORUNLU kılar.

    - MODIS mevcutsa: {"modis_available": True, "downscaling_mode": "modis_downscaling"}.
    - MODIS eksik VE --allow-no-modis verilmediyse: Step7RunnerError (fail-fast,
      Step7B/7D ÇALIŞTIRILMADAN ÖNCE) -- tam olarak istenen mesajla.
    - MODIS eksik VE --allow-no-modis verildiyse: hata FIRLATILMAZ; dönen
      dict, sonradan her step metadata'sına yazılacak diagnostic-mode
      bilgisini taşır (bkz. main()).

    Kozan için (ctx["is_kozan"]) HİÇBİR ŞEY YAPMAZ -- legacy opsiyonel MODIS
    davranışı DEĞİŞMEZ.
    """
    if ctx["is_kozan"]:
        return {"modis_available": None, "downscaling_mode": None, "checked": False}

    needs_modis = ("step7b" in steps_to_run) or ("step7d" in steps_to_run)
    if not needs_modis:
        return {"modis_available": _modis_available(ctx), "downscaling_mode": None, "checked": False}

    modis_available = _modis_available(ctx)
    if modis_available:
        log.info("MODIS mevcut (namespaced): Step7B/7D normal downscaling modunda çalışacak.")
        return {"modis_available": True, "downscaling_mode": "modis_downscaling", "checked": True}

    if not allow_no_modis:
        log.error(_MODIS_MISSING_MESSAGE)
        raise Step7RunnerError(_MODIS_MISSING_MESSAGE)

    log.warning(
        "MODIS eksik ama --allow-no-modis verildi: 'no_modis_diagnostic' modunda "
        "devam ediliyor. Bu GEÇERLİ bir MODIS->Landsat downscaling koşusu DEĞİLDİR."
    )
    return {
        "modis_available": False,
        "downscaling_mode": "no_modis_diagnostic",
        "checked": True,
        "warning": (
            "No MODIS input; this is not a valid MODIS-to-Landsat "
            "downscaling run."
        ),
    }


_DEM_MISSING_MESSAGE = (
    "Experiment-aware DEM/slope missing. Run prepare_dem_for_experiment.py first."
)


def _dem_available(ctx: dict) -> bool:
    elevation_path = ctx["dem_input_dir"] / "elevation.tif"
    slope_path = ctx["dem_input_dir"] / "slope.tif"
    return elevation_path.exists() and slope_path.exists()


def _require_dem_or_fail(ctx: dict, steps_to_run: list[str]) -> dict:
    """
    Kozan-dışı deneyler için deney-özel (namespaced) DEM elevation/slope'u
    Step7B çalışacaksa ZORUNLU kılar (elevation/slope, Step7B'de
    STEP7B_REQUIRE_DEM=True ile zaten zorunlu bir feature'dır; ancak eksik
    olduğunda Step7B'nin kendi içindeki final_sample_count=0 fail-fast'ine
    kadar beklemek yerine, burada -- Step7B'yi hiç çağırmadan önce -- çok
    daha erken ve net bir hata verilir).

    Kozan için (ctx["is_kozan"]) HİÇBİR ŞEY YAPMAZ -- legacy paylaşılan
    data/dem/ davranışı DEĞİŞMEZ (orada zaten dosyalar mevcuttur).
    """
    if ctx["is_kozan"]:
        return {"dem_available": None, "checked": False}

    needs_dem = "step7b" in steps_to_run
    if not needs_dem:
        return {"dem_available": _dem_available(ctx), "checked": False}

    dem_available = _dem_available(ctx)
    if dem_available:
        log.info("DEM mevcut (namespaced): Step7B normal şekilde devam edecek.")
        return {"dem_available": True, "checked": True}

    log.error(_DEM_MISSING_MESSAGE)
    raise Step7RunnerError(_DEM_MISSING_MESSAGE)


# =============================================================================
# Orkestrasyon
# =============================================================================
def _run_one_step(step_name: str, ctx: dict, force: bool) -> dict:
    if step_name == "step7a":
        import src.step7a_tiling_infrastructure as step7a
        return step7a.run_step7a(ctx=ctx, force=force)
    if step_name == "step7b":
        import src.step7b_prepare_downscaling_dataset as step7b
        return step7b.run_step7b(ctx=ctx, force=force)
    if step_name == "step7c":
        import src.step7c_train_downscaling_model as step7c
        return step7c.run_step7c(ctx=ctx, force=force)
    if step_name == "step7d":
        import src.step7d_predict_downscaled_lst as step7d
        return step7d.run_step7d(ctx=ctx, force=force)
    if step_name == "step7e":
        import src.step7e_fuse_landsat_downscaled_lst as step7e
        return step7e.run_step7e(ctx=ctx, force=force)
    raise Step7RunnerError(f"Bilinmeyen step: {step_name}")


def main(
    experiment_id: str = "kozan_2023",
    dry_run: bool = False,
    force: bool = False,
    from_step: str = "step7a",
    to_step: str = "step7e",
    allow_no_modis: bool = False,
) -> dict:
    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx)

    if dry_run:
        log.info("[dry-run] Planlanan yollar:")
        _log_planned_paths(ctx)
        _dry_run_input_check(ctx, allow_no_modis=allow_no_modis)
        log.info("[dry-run] Hiçbir raster okuma/yazma, hiçbir model eğitimi ÇALIŞTIRILMADI.")
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run"}

    if from_step not in STEP_ORDER or to_step not in STEP_ORDER:
        raise Step7RunnerError(f"--from-step/--to-step şunlardan biri olmalı: {STEP_ORDER}")
    start_idx = STEP_ORDER.index(from_step)
    end_idx = STEP_ORDER.index(to_step)
    if start_idx > end_idx:
        raise Step7RunnerError("--from-step, --to-step'ten sonra olamaz.")
    steps_to_run = STEP_ORDER[start_idx:end_idx + 1]

    modis_result = None
    if not ctx["is_kozan"] and "step7b" in steps_to_run:
        # MODIS export'unu best-effort dene -- Step7B'den önce (Step7B/7D
        # namespaced modis_input_dir'i zaten kontrol ediyor). Basarisiz olsa
        # bile, asagidaki _require_modis_or_fail zorunluluk kontrolunu ayrica
        # yapar (export basarisiz olsa da dosyalar baska bir yoldan mevcut
        # olabilir).
        modis_result = prepare_manavgat_modis_context(ctx, force=force)

    # ZORUNLULUK KONTROLÜ (Kozan-dışı, Step7B veya Step7D çalışacaksa):
    # MODIS eksikse VE --allow-no-modis verilmediyse, Step7B/7D ÇALIŞTIRILMADAN
    # ÖNCE fail-fast. Kozan için hiçbir şey değişmez (legacy opsiyonel MODIS).
    modis_requirement = _require_modis_or_fail(ctx, steps_to_run, allow_no_modis)

    # ZORUNLULUK KONTROLÜ (Kozan-dışı, Step7B çalışacaksa): namespaced DEM
    # elevation/slope eksikse, Step7B ÇALIŞTIRILMADAN ÖNCE fail-fast.
    dem_requirement = _require_dem_or_fail(ctx, steps_to_run)

    results: dict = {
        "experiment_id": experiment_id, "ran": True, "steps": {},
        "modis": modis_result, "modis_requirement": modis_requirement,
        "dem_requirement": dem_requirement,
    }
    diagnostic_mode = (
        not ctx["is_kozan"]
        and modis_requirement.get("downscaling_mode") == "no_modis_diagnostic"
    )
    for step_name in steps_to_run:
        log.info("=" * 70)
        log.info("ÇALIŞIYOR: %s [experiment=%s]", step_name, experiment_id)
        log.info("=" * 70)
        results["steps"][step_name] = _run_one_step(step_name, ctx, force)
        if not ctx["is_kozan"]:
            metadata_filename = _STEP_METADATA_FILENAMES.get(step_name)
            if metadata_filename:
                metadata_path = ctx[f"{step_name}_output_dir"] / metadata_filename
                _enrich_metadata_with_experiment_context(metadata_path, ctx)
                if step_name in ("step7b", "step7c", "step7d", "step7e"):
                    _enrich_metadata_with_modis_status(metadata_path, modis_requirement)

    if diagnostic_mode:
        log.warning(
            "TAMAMLANDI ama 'no_modis_diagnostic' modunda: %s", modis_requirement["warning"],
        )
    log.info("Step7 zinciri tamamlandı: %s", steps_to_run)
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0E: deney-farkında (experiment-aware) Step7A-E "
        "(MODIS->Landsat LST downscaling + fusion) çalıştırıcısı. Step8'i "
        "ÇALIŞTIRMAZ, hiçbir burned-area/fire-risk modeli EĞİTMEZ. "
        "kozan_2023 legacy davranışını korur; diğer deneyler (örn. "
        "manavgat_2021) tamamen namespaced çalışır ve MODIS'i (Step7B/7D "
        "çalışacaksa) ZORUNLU kılar."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hiçbir şey çalıştırma; deney özetini + planlanan tüm yolları + "
        "girdi dosya durumunu (MODIS zorunluluk durumu dahil) bas.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Her adımın çıktıları zaten varsa üzerine yaz.",
    )
    parser.add_argument(
        "--from-step", type=str, default="step7a", choices=STEP_ORDER,
        help="Zincirin başlayacağı adım (varsayılan: step7a).",
    )
    parser.add_argument(
        "--to-step", type=str, default="step7e", choices=STEP_ORDER,
        help="Zincirin biteceği adım (varsayılan: step7e).",
    )
    parser.add_argument(
        "--allow-no-modis", action="store_true",
        help="YALNIZCA diagnostic/no-downscaling modu için: Kozan-dışı bir "
        "deneyde MODIS eksikse bile Step7B/7D'nin MODIS'siz devam etmesine "
        "izin verir. Üretilen metadata'ya modis_available=false, "
        "downscaling_mode='no_modis_diagnostic' ve açık bir uyarı yazılır. "
        "Bu GEÇERLİ bir MODIS->Landsat downscaling koşusu ÜRETMEZ.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        force=args.force,
        from_step=args.from_step,
        to_step=args.to_step,
        allow_no_modis=args.allow_no_modis,
    )