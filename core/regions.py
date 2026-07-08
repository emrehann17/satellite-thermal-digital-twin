"""
core/regions.py

Step0: bolge (region) ve deney (experiment) kayit defteri.

Kavramsal ayrim:
    - region     = yalnizca geometri / AOI (Area Of Interest)
    - experiment = region + yil + predictor penceresi + label penceresi +
                   baseline yillari + rol (role) + cikti namespace'i

Bu dosya once mevcut `build_regions()` fonksiyonunu (geriye donuk uyumlu
sekilde) korur, ardindan Step0 deney kayit defterini (EXPERIMENTS) ve
yardimci fonksiyonlari ekler. Step1-Step8 script'leri hala eskisi gibi
`build_regions()` + `core.config.REGION_NAME` uzerinden calisir; bu dosyadaki
yeni fonksiyonlar SADECE ek/opsiyonel bir katmandir, mevcut davranisi
DEGISTIRMEZ.

ONEMLI (Step0 kapsami):
    - Bu asamada Manavgat/Valencia/Zamora pipeline'i CALISTIRILMAZ.
    - Bilimsel hesaplamalar (Step1-Step8) bu dosyadaki degisiklikten
      ETKILENMEZ.
    - Kozan 2023 varsayilan deney olarak kalir (geriye donuk uyumluluk).
"""

from pathlib import Path
from typing import Optional

import ee

from core.paths import PROJECT_ROOT

# =============================================================================
# 1) Region geometrileri (mevcut + yeni placeholder'lar)
# =============================================================================


def build_regions() -> dict:
    """Geriye donuk uyumlu bolge geometrisi sozlugu.

    Mevcut anahtarlar (`dogu_akdeniz`, `kozan_aoi`) DEGISTIRILMEDI.
    Yeni eklenenler (`manavgat_aoi`, `manavgat_aoi_refined_bbox`,
    `manavgat_aoi_wide_buffer`, `valencia_2022_aoi`, `zamora_2022_aoi`)
    yalnizca Step0 deney kayit defteri tarafindan referans verildiginde
    kullanilir; varsayilan pipeline (REGION_NAME="kozan_aoi") bunlardan
    ETKILENMEZ.

    Manavgat icin UC anahtar vardir:
        manavgat_aoi              -> deney kaydinin (manavgat_2021) kullandigi
                                      AKTIF/varsayilan AOI (su an = refined bbox).
        manavgat_aoi_refined_bbox -> ayni geometri, acik isimle (dogrudan
                                      referans vermek/karsilastirmak icin).
        manavgat_aoi_wide_buffer  -> eski simetrik nokta+50 km buffer stili;
                                      yalnizca fallback/debug amaçli, hicbir
                                      deney varsayilan olarak KULLANMAZ.
    """
    kozan_merkez = ee.Geometry.Point([35.82, 37.45])
    kozan_aoi = kozan_merkez.buffer(50000).bounds()

    dogu_akdeniz = ee.Geometry.BBox(33.8, 36.0, 36.7, 38.0)

    # --- Manavgat / Antalya 2021 (ilk anchor natural-vegetation wildfire AOI) ---
    # Manavgat, Kozan'in aksine cropland/aniz-yakma degil, GERCEK
    # dogal-bitki-ortusu orman yangini AOI'si olmasi beklenen ilk deneydir
    # (bkz. EXPERIMENTS["manavgat_2021"]["role"] = "anchor_wildfire").
    #
    # Manavgat ilcesi kendisi kiyi seridinde, tarim/seracilik agirlikli bir
    # bolgededir (narenciye, sera). 2021 yangini ise byuk olcude Manavgat'in
    # KUZEYINDE, Toros Daglari'nin orman/makilik yamaclarina (Akseki,
    # Gundogmus yonunde) dogru yayilmistir. Bu yuzden AOI'yi Manavgat merkez
    # noktasi etrafinda SIMETRIK bir buffer olarak degil, KUZEYE/KUZEYDOGUYA
    # KAYDIRILMIS (kiyidaki tarim kusagini minimize eden, orman/makilik
    # alanlari daha fazla iceren) bir dikdortgen (bbox) olarak taniml uyoruz.
    #
    # ONEMLI: Bu KESIN yangin perimetri DEGILDIR -- MCD64A1/FIRMS/fire-scar
    # verisine gore turetilmemistir. Elle, kabaca cizilmis bir CALISMA AOI'si
    # (working AOI)'dir; Step6B burned-landcover gate ile dogrulanmalidir
    # (gate_level=500m_reconstructed_mcd64a1_cell). Gate "cropland_dominated_control"
    # donerse bu bbox'in kiyi tarim kusagini hala fazla icerdigi, "insufficient_
    # burned_positives" donerse AOI'nin cok kucuk/yanlis konumlandirilmis
    # olabilecegi anlamina gelir -- her iki durumda da bu geometri (asagida,
    # TEK YERDE) yeniden ayarlanabilir.
    #
    # Referans nokta (kiyidaki Manavgat ilce merkezi; sadece dokumantasyon/
    # fallback amaçli, AOI hesaplamasinda dogrudan KULLANILMAZ):
    manavgat_merkez = ee.Geometry.Point([31.44, 36.79])

    # Fallback/debug: eski simetrik nokta+50 km buffer stili, KORUNDU. Hicbir
    # deney varsayilan olarak bunu KULLANMAZ; yalnizca karsilastirma/hata
    # ayiklama icin saklanir.
    manavgat_aoi_wide_buffer = manavgat_merkez.buffer(50000).bounds()

    # Varsayilan, netlestirilmis (refined) Manavgat AOI'si.
    # Refined manually for Manavgat 2021 anchor AOI; should be checked
    # against MCD64A1/FIRMS/fire scar outputs.
    #   - Guney sinir (lat_min=36.72): Manavgat kiyi seridinin hemen kuzeyi;
    #     kiyidaki yogun sera/narenciye kusagini AOI'nin BUYUK COGUNLUGUNDAN
    #     DISLAMAK icin kasitli olarak dar tutuldu (tamamen sifir degil,
    #     kontrol/unburned komsuluk icin bir miktar kiyi seridi birakildi).
    #   - Kuzey sinir (lat_max=37.35): Toros Daglari'nin orman/makilik
    #     yamaclarina (Akseki/Gundogmus yonu) dogru genisletildi.
    #   - Dogu/bati sinirlari (lon 31.05-31.85): Manavgat vadisini ve
    #     dogu/bati komsu orman alanlarini kapsayacak sekilde secildi; asiri
    #     genis tutulmadi (Antalya kentsel alanina tasmamak icin).
    manavgat_aoi_refined_bbox = ee.Geometry.BBox(31.05, 36.72, 31.85, 37.35)

    # `manavgat_aoi`: experiment kaydi (EXPERIMENTS["manavgat_2021"]["region_key"])
    # bu anahtari kullanir. Su anki varsayilan = refined bbox (yukarida).
    manavgat_aoi = manavgat_aoi_refined_bbox

    # --- Disabled placeholder'lar (Valencia / Zamora) ---
    # Bu iki AOI su an EXPERIMENTS kaydinda enabled=False'dur; pipeline
    # tarafindan KULLANILMAZ. Yalnizca ileride external validation /
    # hard transfer test asamasi icin yer tutucu olarak eklenmistir.
    # TODO(step0): Gercek kullanim oncesi kesin AOI sinirlari tanimlanmali.
    valencia_merkez = ee.Geometry.Point([-0.38, 39.47])
    valencia_2022_aoi = valencia_merkez.buffer(50000).bounds()

    zamora_merkez = ee.Geometry.Point([-6.35, 41.85])
    zamora_2022_aoi = zamora_merkez.buffer(50000).bounds()

    return {
        "dogu_akdeniz": dogu_akdeniz,
        "kozan_aoi": kozan_aoi,
        "manavgat_aoi": manavgat_aoi,
        "manavgat_aoi_refined_bbox": manavgat_aoi_refined_bbox,
        "manavgat_aoi_wide_buffer": manavgat_aoi_wide_buffer,
        "valencia_2022_aoi": valencia_2022_aoi,
        "zamora_2022_aoi": zamora_2022_aoi,
    }


# =============================================================================
# 2) Step0 deney (experiment) kayit defteri
# =============================================================================
# Her deney: region + yil + predictor penceresi + label penceresi +
# baseline yillari + rol + cikti namespace'i.
#
# kozan_2023: mevcut, dogrulanmis, negative/control AOI (cropland/aniz-yakma
# agirlikli). Varsayilan (default) deney -- geriye donuk uyumluluk icin.
#
# manavgat_2021: bir sonraki anchor wildfire AOI. Bu Step0 asamasinda
# YALNIZCA kayit defterine eklenir; pipeline HENUZ calistirilmaz.
#
# valencia_2022 / zamora_2022: ileriki asamalar icin disabled placeholder.

EXPERIMENTS = {
    "kozan_2023": {
        "enabled": True,
        "region_key": "kozan_aoi",
        "display_name": "Kozan 2023",
        "role": "negative_control",
        "country": "Turkey",
        "predictor_start_date": "2023-06-01",
        "predictor_end_date": "2023-07-31",
        "label_start_date": "2023-08-01",
        "label_end_date": "2023-10-31",
        "baseline_years": [2019, 2020, 2021, 2022],
        "output_namespace": "kozan_2023",
        "notes": "Cropland-dominated burned labels; retained as negative/control AOI.",
    },
    "manavgat_2021": {
        "enabled": True,
        "region_key": "manavgat_aoi",
        "display_name": "Manavgat / Antalya 2021",
        "role": "anchor_wildfire",
        "country": "Turkey",
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-27",
        "label_start_date": "2021-07-28",
        "label_end_date": "2021-08-31",
        "baseline_years": [2017, 2018, 2019, 2020],
        "output_namespace": "manavgat_2021",
        "notes": "Anchor natural-vegetation wildfire AOI.",
    },
    "valencia_2022": {
        "enabled": False,
        "region_key": "valencia_2022_aoi",
        "display_name": "Valencia / Castellon 2022",
        "role": "external_validation",
        "country": "Spain",
        "predictor_start_date": None,
        "predictor_end_date": None,
        "label_start_date": None,
        "label_end_date": None,
        "baseline_years": [],
        "output_namespace": "valencia_2022",
        "notes": "Placeholder for later Mediterranean external validation.",
    },
    "zamora_2022": {
        "enabled": False,
        "region_key": "zamora_2022_aoi",
        "display_name": "Sierra de la Culebra / Zamora 2022",
        "role": "hard_transfer_test",
        "country": "Spain",
        "predictor_start_date": None,
        "predictor_end_date": None,
        "label_start_date": None,
        "label_end_date": None,
        "baseline_years": [],
        "output_namespace": "zamora_2022",
        "notes": "Placeholder for later harder transfer test.",
    },
}

# Geriye donuk uyumluluk: mevcut pipeline'in varsayilan deneyi.
DEFAULT_EXPERIMENT_ID = "kozan_2023"


# =============================================================================
# 3) Yardimci fonksiyonlar
# =============================================================================


def get_experiment(experiment_id: str) -> dict:
    """Verilen experiment_id icin deney konfigurasyonunu dondurur.

    Sozlugun bir KOPYASINI dondurur (cagiran taraf yanlislikla global
    kayit defterini mutasyona ugratamasin diye), ve kolaylik olsun diye
    "experiment_id" alanini ekler.

    Raises:
        ValueError: experiment_id kayit defterinde yoksa.
    """
    if experiment_id not in EXPERIMENTS:
        valid_ids = ", ".join(sorted(EXPERIMENTS.keys()))
        raise ValueError(
            f"Bilinmeyen experiment_id: '{experiment_id}'. "
            f"Gecerli degerler: {valid_ids}."
        )
    exp = dict(EXPERIMENTS[experiment_id])
    exp["experiment_id"] = experiment_id
    return exp


def get_active_experiment(experiment_id: Optional[str] = None, allow_disabled: bool = False) -> dict:
    """Aktif deneyi cozer ve dondurur.

    Args:
        experiment_id: Secilecek deney kimligi. None ise geriye donuk
            uyumluluk icin DEFAULT_EXPERIMENT_ID ("kozan_2023") kullanilir.
        allow_disabled: True degilse, enabled=False olan bir deney secilirse
            ValueError firlatilir (yanlislikla henuz hazir olmayan bir
            deneyin secilmesini engellemek icin).

    Raises:
        ValueError: experiment_id bilinmiyorsa veya disabled bir deney
            allow_disabled=False iken secilmeye calisilirsa.
    """
    resolved_id = experiment_id if experiment_id is not None else DEFAULT_EXPERIMENT_ID
    exp = get_experiment(resolved_id)
    if not exp["enabled"] and not allow_disabled:
        raise ValueError(
            f"'{resolved_id}' deneyi su an disabled (enabled=False). "
            "Bilerek secmek istiyorsan get_active_experiment(experiment_id, "
            "allow_disabled=True) kullan."
        )
    return exp


def list_experiments(include_disabled: bool = False) -> dict:
    """Kayit defterindeki deneylerin listesini dondurur (loglama/debug icin).

    Args:
        include_disabled: True ise enabled=False olanlar da dahil edilir.
    """
    if include_disabled:
        return {k: dict(v, experiment_id=k) for k, v in EXPERIMENTS.items()}
    return {
        k: dict(v, experiment_id=k)
        for k, v in EXPERIMENTS.items()
        if v["enabled"]
    }


def get_region_for_experiment(experiment_id: str):
    """experiment_id -> region_key -> ee.Geometry cozer.

    Mevcut `build_regions()` geometri uretecini kullanir; yeni bir geometri
    tanimlama mantigi EKLEMEZ.

    Raises:
        ValueError: experiment_id bilinmiyorsa veya region_key
            build_regions() ciktisinda yoksa.
    """
    exp = get_experiment(experiment_id)
    region_key = exp["region_key"]
    regions = build_regions()
    if region_key not in regions:
        raise ValueError(
            f"'{experiment_id}' deneyinin region_key'i ('{region_key}') "
            "build_regions() ciktisinda bulunamadi."
        )
    return regions[region_key]


def get_experiment_output_root(experiment_id: str) -> Path:
    """`outputs/experiments/<output_namespace>/` yolunu dondurur.

    Dizini OLUSTURMAZ; yalnizca yolu hesaplar. Mevcut legacy `outputs/step8a`
    vb. yollari bu fonksiyondan ETKILENMEZ.
    """
    exp = get_experiment(experiment_id)
    return PROJECT_ROOT / "outputs" / "experiments" / exp["output_namespace"]


def get_step_output_dir(experiment_id: str, step_name: str, create: bool = False) -> Path:
    """`outputs/experiments/<output_namespace>/<step_name>/` yolunu dondurur.

    Args:
        experiment_id: Deney kimligi.
        step_name: Ornegin "step8a", "step8b" gibi bir alt dizin adi.
        create: True ise dizin (parents dahil) olusturulur; False ise
            (varsayilan) yalnizca yol hesaplanir, olusturma cagirana
            birakilir (proje genelindeki "step script'i kendi dizinini
            olusturur" konvansiyonuna uygun).
    """
    step_dir = get_experiment_output_root(experiment_id) / step_name
    if create:
        step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir