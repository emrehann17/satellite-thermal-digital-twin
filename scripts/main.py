"""
main.py

Yapılanlar:
    - Step1'den Step8E'ye kadar TÜM pipeline'ı sırasıyla çalıştırır: GEE
      erişimi gereken online kısım (Step1-7B; Step6'nın hemen ardından
      label export cleanup + burned-landcover gate dahil), ardından
      yerel/offline kısım (Step7C downscaling model eğitimi -> Step7D
      downscaled LST tahmini -> Step7E füzyon -> Step8A native 500 m
      modelleme verisi -> Step8B baseline vs thermal belirleyici deney ->
      Step8C spatial-block bootstrap -> Step8D termal özellik ablation ->
      Step8E nihai birleşik rapor).

NOTLAR:
    - Step6 (burned-area association testi) hata-toleranslıdır: GEE/veri
      erişimi başarısız olursa yalnızca uyarı verir, pipeline'ın geri
      kalanını DURDURMAZ (Step6'nın çıktısı sonraki adımların hiçbiri için
      zorunlu girdi değildir).
    - Label export cleanup (src.step6_validate_fire_relation.export_raw_mcd64a1_labels,
      Step6'nın HEMEN ardından çalışır) Step8A için ZORUNLU bir GEE adımıdır
      ve hata-toleranslı DEĞİLDİR: başarısız olursa pipeline burada durur,
      çünkü Step8A gerçek BurnDate DOY değerleri olmadan (yalnızca binary
      maskeyle) yanlış/geçersiz bir etiket kullanmaya çalışır ve zaten kendi
      içinde net hata ile durur. Bu adım Step6'nın kendi association
      testinden BAĞIMSIZDIR (Step6 başarısız olsa/atlansa bile çalışır).
    - Burned-landcover gate (src.step6b_burned_landcover_gate, label export
      cleanup'ın hemen ardından çalışır) DIAGNOSTIC'tir: MCD64A1-burned
      hücrelerin landcover kompozisyonunu özetler ve
      wildfire_candidate_pass / cropland_dominated_control /
      insufficient_burned_positives / mixed_or_uncertain olarak sınıflar.
      cropland_dominated_control sonucu (Kozan 2023 için beklenen) pipeline'ı
      DURDURMAZ; yalnızca raw BurnDate binary görünüyorsa veya gerekli girdi
      rasterları eksikse hata verir.
    - Step7C/7D/7E ve Step8A-8E, önceki çalıştırmadan kalan çıktılar zaten
      varsa varsayılan olarak NET HATA ile durur (üzerine yazma güvenliği).
      Bu betiği tekrar/yeniden çalıştırmak için `--force` verin; bu, ilgili
      tüm adımlara (burned-landcover gate dahil) iletilir.
    - Step7C (model eğitimi) ve özellikle Step8D (popülasyon başına 11
      model x spatial-block CV) gerçek AOI verisiyle uzun sürebilir
      (Step8D tek başına gerçek ~48k satırlık veride tahminen 15-40 dakika).
      Bu, tüm zinciri tek komutla çalıştırmanın doğal maliyetidir.

KULLANIM:
    python scripts/main.py            # ilk calistirma
    python scripts/main.py --force    # ciktilar zaten varsa uzerine yaz
"""

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from datetime import datetime
import traceback

from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.regions import get_active_experiment, get_experiment_output_root

import src.step1_fetch_modis as step1_fetch_modis
import src.step2_modis_5year_mean as step2_modis_5year_mean
import src.step2b_dem as step2b_dem
import src.step3_landsat_lst as step3_landsat_lst
import src.step4_export_geotiff as step4_export_geotiff
import src.step4b_download_drive_export as step4b_download_drive_export
import src.step5_preprocess_timeseries as step5_preprocess_timeseries
import src.step5b_diagnostic_report as step5b_diagnostic_report
import src.step5c_tvdi as step5c_tvdi
import src.step6_validate_fire_relation as step6_validate_fire_relation
import src.step6b_burned_landcover_gate as step6b_burned_landcover_gate
import src.step7a_tiling_infrastructure as step7a_tiling_infrastructure
import src.step7b_prepare_downscaling_dataset as step7b_prepare_downscaling_dataset
import src.step7c_train_downscaling_model as step7c_train_downscaling_model
import src.step7d_predict_downscaled_lst as step7d_predict_downscaled_lst
import src.step7e_fuse_landsat_downscaled_lst as step7e_fuse_landsat_downscaled_lst
import src.step8a_prepare_500m_modeling_dataset as step8a_prepare_500m_modeling_dataset
import src.step8b_train_baseline_vs_thermal_model as step8b_train_baseline_vs_thermal_model
import src.step8c_spatial_block_bootstrap_uncertainty as step8c_spatial_block_bootstrap_uncertainty
import src.step8d_thermal_feature_ablation as step8d_thermal_feature_ablation
import src.step8e_final_report as step8e_final_report


BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("main")


def log_step0_banner(experiment_id: str) -> dict:
    """Step0: aktif deneyi cozer, logo/konsola yazdirir ve deney dict'ini dondurur.

    NOT: Bu fonksiyon YALNIZCA bilgilendirme/namespacing amaclidir. Step1-8E
    script'leri hala core/config.py'deki legacy REGION_NAME / PREDICTOR_*_DATE
    / LABEL_*_DATE sabitlerini kullanir (bkz. core/config.py Step0 koprusu
    yorumlari). Bu ilk implementasyonda yalnizca "kozan_2023" deneyi gercekten
    calistirilir; farkli bir --experiment ile tam pipeline'i calistirmaya
    calismak fail-fast bir hata verir (bkz. main() cagrisi altindaki kontrol).
    """
    exp = get_active_experiment(experiment_id)
    output_root = get_experiment_output_root(experiment_id)
    baseline_years_str = ", ".join(str(y) for y in exp["baseline_years"])

    log.info("[Step0] Active experiment: %s", experiment_id)
    log.info("[Step0] Region: %s", exp["region_key"])
    log.info("[Step0] Role: %s", exp["role"])
    log.info(
        "[Step0] Predictor window: %s -> %s",
        exp["predictor_start_date"], exp["predictor_end_date"],
    )
    log.info(
        "[Step0] Label window: %s -> %s",
        exp["label_start_date"], exp["label_end_date"],
    )
    log.info("[Step0] Baseline years: %s", baseline_years_str)
    log.info("[Step0] Output root: %s", output_root)

    print(f"[Step0] Active experiment: {experiment_id}")
    print(f"[Step0] Region: {exp['region_key']}")
    print(f"[Step0] Role: {exp['role']}")
    print(f"[Step0] Predictor window: {exp['predictor_start_date']} -> {exp['predictor_end_date']}")
    print(f"[Step0] Label window: {exp['label_start_date']} -> {exp['label_end_date']}")
    print(f"[Step0] Baseline years: {baseline_years_str}")
    print(f"[Step0] Output root: {output_root}")

    return exp


def run_step(step_name: str, step_func) -> None:
    log.info("=" * 70)
    log.info(f"{step_name} BAŞLATILIYOR")
    log.info("=" * 70)

    result = step_func()

    log.info("=" * 70)
    log.info(f"{step_name} TAMAMLANDI")
    log.info("=" * 70)
    return result


def main(force: bool = False) -> None:
    start_time = datetime.now()

    log.info("#" * 80)
    log.info("PIPELINE BAŞLIYOR (STEP1 -> STEP8E, uçtan uca)")
    log.info(f"Başlangıç zamanı: {start_time.isoformat()}")
    log.info(f"force={force}")
    log.info(
        "NOT: Step6 hata-toleranslıdır (basarisiz olursa yalnizca uyarir). "
        "Label export cleanup (raw MCD64A1 BurnDate) ise ZORUNLUDUR; "
        "basarisiz olursa pipeline burada durur (Step8A gecerli DOY etiketi "
        "olmadan calisamaz). Burned-landcover gate DIAGNOSTIC'tir; "
        "cropland_dominated_control sonucu pipeline'i durdurmaz."
    )
    log.info("#" * 80)

    try:
        run_step("STEP 1", step1_fetch_modis.main)
        run_step("STEP 2", step2_modis_5year_mean.main)
        run_step("STEP 2B", step2b_dem.main)
        step3_result = run_step("STEP 3", step3_landsat_lst.main)
        run_step(
            "STEP 4",
            lambda: step4_export_geotiff.main(step3_result=step3_result),
        )
        run_step("STEP 4B", step4b_download_drive_export.main)
        run_step("STEP 5", step5_preprocess_timeseries.main)
        run_step("STEP 5C", step5c_tvdi.main)
        run_step("STEP 5B", step5b_diagnostic_report.main)

        # Step6 burned-area association testi. Yangın etiketleri GEE'den çekilir;
        # veri yoksa veya GEE erişimi başarısız olursa pipeline'ın geri kalanı
        # etkilenmesin diye hata-toleranslı çağrılır (hiçbir sonraki adım
        # Step6'nın çıktısına bağımlı değildir).
        try:
            run_step("STEP 6", step6_validate_fire_relation.main)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "STEP 6 atlandı (burned-area validation başarısız): %s", exc
            )

        # Label export cleanup: gerçek MCD64A1 BurnDate DOY değerlerini (+
        # opsiyonel binary maskeyi) canonical konuma (outputs/validation/labels/)
        # yazar. ZORUNLUDUR ve hata-toleranslı DEĞİLDİR -- Step8A geçerli DOY
        # etiketi olmadan çalışamaz. Step6'nın kendi association testinden
        # (yukarıda) BAĞIMSIZDIR: Step6 başarısız/atlanmış olsa bile bu adım
        # çalışır ve kendi GEE erişimini kurar.
        run_step(
            "LABEL EXPORT CLEANUP (raw MCD64A1 BurnDate)",
            lambda: step6_validate_fire_relation.export_raw_mcd64a1_labels(also_binary=True),
        )

        # Burned-landcover gate: MCD64A1-burned ~500 m hücrelerin landcover
        # kompozisyonunu özetler (wildfire_candidate_pass /
        # cropland_dominated_control / insufficient_burned_positives /
        # mixed_or_uncertain). DIAGNOSTIC'tir; cropland_dominated_control
        # sonucu (Kozan için beklenen) pipeline'ı DURDURMAZ. Yalnızca raw
        # BurnDate binary görünüyorsa veya gerekli girdi rasterları eksikse
        # hata verir.
        run_step(
            "BURNED-LANDCOVER GATE",
            lambda: step6b_burned_landcover_gate.main(force=force),
        )

        run_step("STEP 7A", step7a_tiling_infrastructure.main)
        run_step("STEP 7B", step7b_prepare_downscaling_dataset.main)

        # --- Offline/yerel devam: downscaling model + füzyon + Step8 ailesi ---
        run_step("STEP 7C", lambda: step7c_train_downscaling_model.main(force=force))
        run_step("STEP 7D", lambda: step7d_predict_downscaled_lst.main(force=force))
        run_step("STEP 7E", lambda: step7e_fuse_landsat_downscaled_lst.main(force=force))

        run_step("STEP 8A", lambda: step8a_prepare_500m_modeling_dataset.main(force=force))
        run_step("STEP 8B", lambda: step8b_train_baseline_vs_thermal_model.main(force=force))
        run_step("STEP 8C", lambda: step8c_spatial_block_bootstrap_uncertainty.main(force=force))
        run_step("STEP 8D", lambda: step8d_thermal_feature_ablation.main(force=force))
        run_step("STEP 8E", lambda: step8e_final_report.main(force=force))

        end_time = datetime.now()

        log.info("#" * 80)
        log.info("PIPELINE TAMAMLANDI (STEP1 -> STEP8E)")
        log.info(f"Bitiş zamanı: {end_time.isoformat()}")
        log.info(f"Toplam sure: {end_time - start_time}")
        log.info(f"Log dosyası: {log_file}")
        log.info(
            "Nihai rapor: outputs/step8e/step8e_summary.md, "
            "step8e_summary.json, step8e_results_tables.xlsx, "
            "step8e_key_findings.csv."
        )
        log.info("#" * 80)

    except Exception as e:
        log.error("Pipeline çalışırken hata oluştu.")
        log.error(str(e))
        log.error(traceback.format_exc())

        print("\n" + "=" * 80)
        print("HATA OLUŞTU")
        print(str(e))
        print("Detaylar log dosyasında mevcut:")
        print(log_file)
        print("=" * 80 + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="satellite-thermal-digital-twin: Step1'den Step8E'ye "
        "kadar tüm pipeline'ı çalıştırır."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Step7C/7D/7E ve Step8A-8E icin ciktilar zaten varsa uzerine yaz.",
    )
    parser.add_argument(
        "--experiment", default="kozan_2023",
        help=(
            "Step0 deney kimligi (core/regions.py EXPERIMENTS kaydi). "
            "Varsayilan: kozan_2023. NOT: bu Step0 implementasyonunda "
            "yalnizca kozan_2023 gercekten calistirilabilir; baska bir "
            "deney icin --dry-run kullanin."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Yalnizca Step0 aktif deney bilgisini (region, pencereler, "
            "baseline yillari, cikti koku) yazdirir ve pipeline'i "
            "CALISTIRMADAN cikar."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    active_exp = log_step0_banner(args.experiment)

    if args.dry_run:
        print("[Step0] --dry-run: pipeline calistirilmadi.")
    elif args.experiment != "kozan_2023":
        raise SystemExit(
            f"'{args.experiment}' deneyi icin tam pipeline calistirma bu "
            "Step0 implementasyonunda henuz desteklenmiyor (Step1-8E hala "
            "legacy kozan_2023 config sabitlerini kullaniyor). Bu deneyi "
            "onizlemek icin --dry-run ekleyin; gercek calistirma icin "
            "Step1-8E'nin experiment-aware hale getirilmesi gereken "
            "sonraki bir refactor asamasini bekleyin."
        )
    else:
        main(force=args.force)