"""
main.py

Yapılanlar:
    - Step1'den Step8E'ye kadar TÜM pipeline'ı sırasıyla çalıştırır: GEE
      erişimi gereken online kısım (Step1-7B), ardından yerel/offline kısım
      (Step7C downscaling model eğitimi -> Step7D downscaled LST tahmini ->
      Step7E füzyon -> raw MCD64A1 BurnDate export -> Step8A native 500 m
      modelleme verisi -> Step8B baseline vs thermal belirleyici deney ->
      Step8C spatial-block bootstrap -> Step8D termal özellik ablation ->
      Step8E nihai birleşik rapor).

NOTLAR:
    - Step6 (burned-area association testi) hata-toleranslıdır: GEE/veri
      erişimi başarısız olursa yalnızca uyarı verir, pipeline'ın geri
      kalanını DURDURMAZ (Step6'nın çıktısı sonraki adımların hiçbiri için
      zorunlu girdi değildir).
    - Raw MCD64A1 BurnDate export (scripts/export_mcd64a1_raw_burndate.py)
      Step8A için ZORUNLU bir GEE adımıdır ve hata-toleranslı DEĞİLDİR:
      başarısız olursa pipeline burada durur, çünkü Step8A gerçek BurnDate
      DOY değerleri olmadan (yalnızca binary maskeyle) yanlış/geçersiz bir
      etiket kullanmaya çalışır ve zaten kendi içinde net hata ile durur.
    - Step7C/7D/7E ve Step8A-8E, önceki çalıştırmadan kalan çıktılar zaten
      varsa varsayılan olarak NET HATA ile durur (üzerine yazma güvenliği).
      Bu betiği tekrar/yeniden çalıştırmak için `--force` verin; bu, ilgili
      tüm adımlara iletilir.
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
import src.step7a_tiling_infrastructure as step7a_tiling_infrastructure
import src.step7b_prepare_downscaling_dataset as step7b_prepare_downscaling_dataset
import src.step7c_train_downscaling_model as step7c_train_downscaling_model
import src.step7d_predict_downscaled_lst as step7d_predict_downscaled_lst
import src.step7e_fuse_landsat_downscaled_lst as step7e_fuse_landsat_downscaled_lst
import scripts.export_mcd64a1_raw_burndate as export_mcd64a1_raw_burndate
import src.step8a_prepare_500m_modeling_dataset as step8a_prepare_500m_modeling_dataset
import src.step8b_train_baseline_vs_thermal_model as step8b_train_baseline_vs_thermal_model
import src.step8c_spatial_block_bootstrap_uncertainty as step8c_spatial_block_bootstrap_uncertainty
import src.step8d_thermal_feature_ablation as step8d_thermal_feature_ablation
import src.step8e_final_report as step8e_final_report


BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("main")


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
        "Raw MCD64A1 BurnDate export ise ZORUNLUDUR; basarisiz olursa "
        "pipeline burada durur (Step8A gecerli DOY etiketi olmadan calisamaz)."
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

        run_step("STEP 7A", step7a_tiling_infrastructure.main)
        run_step("STEP 7B", step7b_prepare_downscaling_dataset.main)

        # --- Offline/yerel devam: downscaling model + füzyon + Step8 ailesi ---
        run_step("STEP 7C", lambda: step7c_train_downscaling_model.main(force=force))
        run_step("STEP 7D", lambda: step7d_predict_downscaled_lst.main(force=force))
        run_step("STEP 7E", lambda: step7e_fuse_landsat_downscaled_lst.main(force=force))

        # Raw MCD64A1 BurnDate export (GEE) -- ZORUNLU, hata-toleranslı DEĞİL.
        run_step(
            "RAW MCD64A1 BURNDATE EXPORT",
            lambda: export_mcd64a1_raw_burndate.main(argv=["--also-binary"]),
        )

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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(force=args.force)