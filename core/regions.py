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
    - Bilimsel hesaplamalar (Step1-Step8) bu dosyadaki degisiklikten
      ETKILENMEZ.
    - Kozan 2023 varsayilan deney olarak kalir (geriye donuk uyumluluk).
"""

from pathlib import Path
from typing import Optional

import ee

from core.paths import PROJECT_ROOT

# =============================================================================
# 0) AOI bbox sabitleri (test-edilebilir; ee.Geometry OLMADAN da erisilebilir)
# =============================================================================
# ee.Geometry.BBox nesnesinden koordinatlari cekmek genelde bir GEE server
# cagrisi (.getInfo) gerektirir. Testlerin (ve provenance/hash mantiginin)
# GEE auth OLMADAN AOI koordinatlarini dogrulayabilmesi icin, Muğla bbox'i
# ONCE burada (lon_min, lat_min, lon_max, lat_max) sırasında -- ee.Geometry.BBox
# ile AYNI argüman sırası -- bir Python tuple olarak tanimlanir; build_regions()
# geometriyi bu sabitten uretir. CRS = EPSG:4326 (projenin EXPORT_CRS'i).
#
# Muğla 2021: Marmaris / Bodrum / Milas / Köyceğiz 2021 yaz yanginlarini ve
# cevredeki dogal-bitki-ortusu (orman/makilik) alanlari kapsayan CALISMA AOI'si.
# KESIN yangin perimetri DEGILDIR; Step6B burned-landcover gate ile
# dogrulanmalidir (Manavgat/Bejís gibi, gate sonrasi netlestirilebilir).
#
# Hedef sehir merkezleri (yaklasik) -- bbox bunlarin TAMAMINI marjla icermeli:
#   Bodrum    ~ (lon 27.43, lat 37.03)
#   Milas     ~ (lon 27.78, lat 37.32)
#   Marmaris  ~ (lon 28.27, lat 36.85)
#   Köyceğiz  ~ (lon 28.69, lat 36.97)
# Secilen sinirlar:
#   lon_min 27.10  : Bodrum yarimadasinin batisi (Turgutreis ~27.25) dahil.
#   lat_min 36.60  : Marmaris'in guneyindeki kiyi/orman seridini kapsar.
#   lon_max 28.90  : Köyceğiz'in ve Marmaris'in dogu yayilimini kapsar.
#   lat_max 37.45  : Milas/Yeniköy yanginlarinin kuzey yayilimini kapsar.
# Not: bbox genis gorunse de buyuk bolumu Ege/Akdeniz DENIZIdir (Bodrum ve
# Marmaris yarimadalari); province-wide bir AOI DEGILDIR. Dort hedef yangin
# bu araliga gercekten yayilmistir (Bodrum <-> Köyceğiz ~110 km).
MUGLA_AOI_BBOX = (27.10, 36.60, 28.90, 37.45)

# Kuzey Evia (Euboea), Yunanistan -- 2021 yaz yangini. Manavgat/Bejís/Muğla
# gibi ayni-yil Akdeniz "mediterranean_transfer_wildfire" AOI'lerine bir
# yenisi (bkz. EXPERIMENTS["evia_2021"]). Genis, yer-tabanli (place-based)
# bir bolgesel dikdortgendir -- KESIN yangin perimetri DEGILDIR, MCD64A1
# etiketlerine gore AYARLANMAMISTIR. Step6B burned-landcover gate ile
# dogrulanmalidir, tipki digerleri gibi.
NORTH_EVIA_AOI_BBOX = (23.12, 38.68, 23.52, 39.08)

# Kuzey Evia GENISLETILMIS (extended) AOI -- ayri bir deney icin
# (bkz. EXPERIMENTS["evia_2021_extended"]). Mevcut NORTH_EVIA_AOI_BBOX'i
# DEGISTIRMEZ; evia_2021 aynen kalir. Bu bbox, evia_2021 AOI'sini GEOMETRIK
# OLARAK TAMAMEN ICERIR (her dort kenarda disariya dogru genisletilmistir).
#
# Amac: 2021 Kuzey Evia yangin koridorunun daha genis COGRAFI BAGLAMINI ve
# cevredeki YANMAMIS dogal vejetasyonu kapsamak. Geometri YALNIZCA yer
# (place) bilgisiyle -- Limni-Rovies-Agia Anna-Pefki koridoru ve Evia
# adasinin kuzey ucu -- tanimlanmistir; MCD64A1/Copernicus burned-area
# etiketlerine, burned prevalence'a, gate sonucuna, AUC/PR-AUC'ye veya
# herhangi bir model metrigine BAKILARAK optimize EDILMEMISTIR ve birden
# fazla aday geometri denenip en iyisi SECILMEMISTIR (tek, deterministik
# tanim).
#
# Koridorun tasarim capalari (yaklasik yerlesim koordinatlari; yalnizca
# dokumantasyon amacli, hesaplamada KULLANILMAZ):
#   Rovies     ~ (lon 23.22, lat 38.78)
#   Limni      ~ (lon 23.33, lat 38.77)
#   Agia Anna  ~ (lon 23.42, lat 38.87)
#   Pefki      ~ (lon 23.67, lat 39.02)   <-- mevcut evia_2021 bbox'inin
#                                             DISINDA (lon_max = 23.52).
# Kenar gerekceleri (mevcut -> genisletilmis):
#   lon_min 23.12 -> 23.05 : Rovies kiyi yamacini marjla icine alir; Kuzey
#                            Evia Korfezi'nin karsi kiyisindaki Fthiotida
#                            ANAKARASINA (Arkitsa ~23.03) tasmamak icin
#                            kasitli olarak dar tutuldu.
#   lat_min 38.68 -> 38.55 : Prokopi/Kirinthos ic kesimine ve Dirfys'e bakan
#                            cam/makilik yamaclara dogru ~14 km guneye
#                            genisler -- YANMAMIS dogal vejetasyon baglami.
#   lon_max 23.52 -> 23.85 : Pefki'yi (23.67) ve kuzeydogudaki Ege'ye bakan
#                            kiyi seridini icine alir; koridorun eski bbox
#                            tarafindan KESILEN kuzeydogu ucu.
#   lat_max 39.08 -> 39.15 : Adanin kuzey ucunun (Artemisio Burnu ~39.03)
#                            kuzeyinde marj birakir.
# KESIN yangin perimetri DEGILDIR. Step6B burned-landcover gate ile
# dogrulanmalidir -- tipki manavgat_2021 / bejis_2022 / mugla_2021 /
# evia_2021 gibi.
NORTH_EVIA_EXTENDED_AOI_BBOX = (23.05, 38.55, 23.85, 39.15)


# =============================================================================
# 1) Region geometrileri (mevcut + yeni placeholder'lar)
# =============================================================================


def build_regions() -> dict:
    """Geriye donuk uyumlu bolge geometrisi sozlugu.

    Mevcut anahtarlar (`dogu_akdeniz`, `kozan_aoi`) DEGISTIRILMEDI.
    Yeni eklenenler (`manavgat_aoi`, `manavgat_aoi_refined_bbox`,
    `manavgat_aoi_wide_buffer`, `valencia_2022_aoi`,
    `bejis_aoi`) yalnizca Step0 deney kayit defteri tarafindan referans
    verildiginde kullanilir; varsayilan pipeline (REGION_NAME="kozan_aoi")
    bunlardan ETKILENMEZ.

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

    # --- Disabled placeholder (Valencia) ---
    # Bu AOI su an EXPERIMENTS kaydinda enabled=False'dur; pipeline
    # tarafindan KULLANILMAZ. Yalnizca ileride external validation /
    # hard transfer test asamasi icin yer tutucu olarak eklenmistir.
    # TODO(step0): Gercek kullanim oncesi kesin AOI sinirlari tanimlanmali.
    valencia_merkez = ee.Geometry.Point([-0.38, 39.47])
    valencia_2022_aoi = valencia_merkez.buffer(50000).bounds()

    # --- Bejis / Castellon 2022 (Akdeniz transfer wildfire AOI, Ispanya) ---
    # Bejis yangini (Castellon, Valencian Community, Ispanya), 2022-08-15'te
    # baslamistir. Bu, Manavgat 2021 ile karsilastirilabilir bir "Akdeniz
    # transfer wildfire" vaka calismasi olarak eklenmistir; Kozan gibi bir
    # kontrol bolgesi DEGILDIR (bkz. EXPERIMENTS["bejis_2022"]["role"] =
    # "mediterranean_transfer_wildfire").
    #
    # Initial Bejís candidate bbox; refine after AOI preview and MCD64A1
    # burned-landcover gate.
    #
    # ONEMLI: Bu KESIN yangin perimetri DEGILDIR -- yalnizca ilk aday
    # (candidate) bir dikdortgendir. AOI onizleme (scripts/preview_experiment_aoi.py)
    # ve Step6B burned-landcover gate calistirildiktan sonra, yanan hucre
    # kumesi kirpiliyorsa veya ilgisiz yanginlar/cropland baskin cikiyorsa
    # bu geometri (asagida, TEK YERDE) yeniden ayarlanmalidir -- tipki
    # Manavgat AOI'sinin gate sonrasi netlestirilmesi gibi.
    bejis_aoi = ee.Geometry.BBox(-1.05, 39.68, -0.35, 40.15)

    # --- Muğla 2021 (ayni ulke/ayni yil transfer wildfire; internship sorusu:
    # transfer basarisizligi BOLGESEL mi yoksa YANGIN-OLAYINA-OZGU mu?) ---
    # Candidate bbox; module-seviyesi MUGLA_AOI_BBOX sabitinden (TEK KAYNAK)
    # uretilir. Manavgat/Bejís gibi KESIN yangin perimetri DEGILDIR; Step6B
    # burned-landcover gate ile dogrulanmali, gate sonrasi netlestirilebilir.
    # AOI, gate GORULMEDEN once tanimlanir; sonuca gore covertly ayarlanmaz.
    mugla_aoi_candidate_bbox = ee.Geometry.BBox(*MUGLA_AOI_BBOX)
    mugla_aoi = mugla_aoi_candidate_bbox

    # --- North Evia (Euboea) 2021 (Akdeniz transfer wildfire, Yunanistan) ---
    # Manavgat/Bejís/Muğla ile karsilastirilabilir, ayni-yil (2021) bir baska
    # Akdeniz dogal-bitki-ortusu yangin AOI'si. Genis, yer-tabanli (place-
    # based) bolgesel bir dikdortgendir; module-seviyesi NORTH_EVIA_AOI_BBOX
    # sabitinden (TEK KAYNAK) uretilir. KESIN yangin perimetri DEGILDIR --
    # Copernicus burned-area perimetrine veya MCD64A1 etiketlerine gore
    # KIRPILMAMISTIR/AYARLANMAMISTIR. Step6B burned-landcover gate ile
    # dogrulanmali, gate sonrasi (gorulduyse) netlestirilebilir -- tipki
    # Manavgat/Bejís/Muğla gibi.
    north_evia_candidate_bbox = ee.Geometry.BBox(*NORTH_EVIA_AOI_BBOX)
    north_evia = north_evia_candidate_bbox

    # --- North Evia GENISLETILMIS AOI (evia_2021_extended) ---
    # AYRI bir region anahtari; yukaridaki `north_evia` (evia_2021) geometrisi
    # DEGISTIRILMEZ. Module-seviyesi NORTH_EVIA_EXTENDED_AOI_BBOX sabitinden
    # (TEK KAYNAK) uretilir ve o sabitin uzerinde belgelenen coğrafi
    # gerekceyle (Limni-Rovies-Agia Anna-Pefki koridoru + cevredeki yanmamis
    # dogal vejetasyon) tanimlanmistir. `north_evia` bbox'ini her dort
    # kenarda ICERIR.
    north_evia_extended_candidate_bbox = ee.Geometry.BBox(*NORTH_EVIA_EXTENDED_AOI_BBOX)
    north_evia_extended = north_evia_extended_candidate_bbox

    return {
        "dogu_akdeniz": dogu_akdeniz,
        "kozan_aoi": kozan_aoi,
        "manavgat_aoi": manavgat_aoi,
        "manavgat_aoi_refined_bbox": manavgat_aoi_refined_bbox,
        "manavgat_aoi_wide_buffer": manavgat_aoi_wide_buffer,
        "valencia_2022_aoi": valencia_2022_aoi,
        "bejis_aoi": bejis_aoi,
        "mugla_aoi": mugla_aoi,
        "mugla_aoi_candidate_bbox": mugla_aoi_candidate_bbox,
        "north_evia": north_evia,
        "north_evia_candidate_bbox": north_evia_candidate_bbox,
        "north_evia_extended": north_evia_extended,
        "north_evia_extended_candidate_bbox": north_evia_extended_candidate_bbox,
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
# bejis_2022: Manavgat 2021 ile karsilastirilabilir ikinci bir Akdeniz
# transfer wildfire vaka calismasi (Ispanya, Bejis/Castellon). Bu Step0
# asamasinda YALNIZCA kayit defterine + ilk aday AOI'ye eklenir; gate/
# predictor/Step7/Step8/transfer modelleme HENUZ calistirilmaz.
#
# valencia_2022: ileriki asamalar icin disabled placeholder.

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
    
        "bejis_2022": { #valenica
        "enabled": True,
        "region_key": "bejis_aoi",
        "display_name": "Bejís / Castellón 2022",
        "role": "mediterranean_transfer_wildfire",
        "country": "Spain",
        "predictor_start_date": "2022-06-15",
        "predictor_end_date": "2022-08-14",
        "label_start_date": "2022-08-15",
        "label_end_date": "2022-09-30",
        "baseline_years": [2018, 2019, 2020, 2021],
        "output_namespace": "bejis_2022",
        "notes": (
            "Bejís wildfire, Castellón / Valencian Community, Spain; fire start "
            "2022-08-15. Mediterranean transfer wildfire case comparable to "
            "Manavgat 2021 -- NOT a negative/control AOI. Initial Bejís "
            "candidate bbox; refine after AOI preview and MCD64A1 "
            "burned-landcover gate. Must pass the same MCD64A1 "
            "burned-landcover gate before modeling, exactly like Manavgat 2021."
        ),
    },

    "mugla_2021": {
        "enabled": True,
        "region_key": "mugla_aoi",
        "display_name": "Muğla 2021",
        "role": "same_country_same_year_transfer_wildfire",
        "country": "Turkey",
        "predictor_start_date": "2021-06-01",
        "predictor_end_date": "2021-07-28",
        "label_start_date": "2021-07-29",
        "label_end_date": "2021-09-15",
        "baseline_years": [2017, 2018, 2019, 2020],
        "output_namespace": "mugla_2021",
        # --- LEAKAGE-SAFE PRE-LABEL EXCLUSION (Muğla-specific) ---------------
        # A separate fire around Bördübet / Marmaris burned ~2021-06-21..25,
        # which is INSIDE the predictor window (2021-06-01..2021-07-28). Cells
        # that already burned BEFORE label_start (2021-07-29) carry post-fire
        # hot/dry/bare predictor signatures and would contaminate the predictor
        # population (temporal leakage). They must be EXCLUDED from the whole
        # analysis universe -- NOT treated as unburned negatives.
        #
        # The canonical label raster (mcd64a1_raw.tif) is DOY-masked to the
        # label window, so it sets pre-label BurnDate to 0 (indistinguishable
        # from genuinely-unburned). Excluding pre-label burns therefore
        # REQUIRES a SEPARATE pre-label BurnDate raster over the pre-label
        # window below. When exclude_pre_label_burns is True, the gate reads
        # that extra raster and drops any cell with a positive pre-label
        # BurnDate before evaluating natural-vegetation composition.
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2021-06-01", "2021-07-28"],
        # Optional, purely-diagnostic secondary sub-window (generic field;
        # any experiment may set it): reports how many pre-label-excluded
        # cells fall specifically inside this narrower range. For Muğla this
        # isolates the known Bördübet/Marmaris fire (~2021-06-21..25) within
        # the wider pre-label window. Absent for every other experiment --
        # scripts/run_label_gate_only.py reads this generically (never a
        # hardcoded per-AOI literal) and passes None when unset.
        "pre_label_diagnostic_window": ["2021-06-21", "2021-06-25"],
        "notes": (
            "Muğla 2021 (Marmaris/Bodrum/Milas/Köyceğiz). Same-country, "
            "same-year Mediterranean pine wildfire, added as the first/highest-"
            "priority NEW event to test whether cross-region transfer failure "
            "is REGIONAL or WILDFIRE-EVENT-SPECIFIC. NOT a negative/control AOI. "
            "Candidate bbox (MUGLA_AOI_BBOX); refine after AOI preview + MCD64A1 "
            "burned-landcover gate, exactly like Manavgat 2021 / Bejís 2022. "
            "LEAKAGE: a Bördübet/Marmaris fire (~2021-06-21..25) lies inside the "
            "predictor window; exclude_pre_label_burns=True removes any cell "
            "that burned before label_start (2021-07-29) from the analysis "
            "universe. Gate result must be sent to the advisor; passing the gate "
            "does NOT authorize downstream predictor/Step7/Step8/Step9/Step10 "
            "execution (downstream_authorized=false)."
        ),
    },

    "evia_2021": {
        "enabled": True,
        "region_key": "north_evia",
        "display_name": "North Evia (Euboea), Greece",
        "role": "mediterranean_transfer_wildfire",
        "country": "Greece",
        "predictor_start_date": "2021-06-05",
        "predictor_end_date": "2021-08-02",
        "label_start_date": "2021-08-03",
        "label_end_date": "2021-09-30",
        "baseline_years": [2017, 2018, 2019, 2020],
        "output_namespace": "evia_2021",
        # --- LEAKAGE-SAFE PRE-LABEL EXCLUSION (generic contract; same
        # mechanism as mugla_2021, applied here via the SAME declarative
        # fields -- no code branch added for Evia) -----------------------
        # The canonical label raster is DOY-masked to the label window, so
        # any cell that burned BEFORE label_start would show BurnDate=0
        # (indistinguishable from genuinely-unburned) unless excluded via a
        # SEPARATE pre-label BurnDate raster covering [predictor_start,
        # label_start). exclude_pre_label_burns=True makes the generic
        # gate/Step8A machinery export that raster, exclude affected ~500 m
        # cells from the analysis universe, and record their identities in
        # the canonical manifest Step8A reads -- exactly as already built
        # for mugla_2021, now also applied to evia_2021.
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2021-06-05", "2021-08-02"],
        "notes": (
            "Same-year cross-country Mediterranean wildfire replication "
            "relative to manavgat_2021; broad regional AOI selected from "
            "geographic place coverage rather than an observed "
            "burned-area perimeter. Pre-label leakage exclusion enabled "
            "(exclude_pre_label_burns=True) over [predictor_start, "
            "label_start) = 2021-06-05..2021-08-02, using the same generic "
            "Gate -> Step8A contract already established for mugla_2021."
        ),
    },

    # evia_2021 ile AYNI olay/pencere, YALNIZCA daha genis bir cografi AOI.
    # evia_2021 kaydi ve ciktilari DEGISTIRILMEZ/TASINMAZ/OVERWRITE EDILMEZ;
    # bu AYRI bir deneydir ve kendi namespace'ine
    # (outputs/experiments/evia_2021_extended/) yazar. Predictor tarihleri,
    # label tarihleri, baseline yillari, role, country ve pre-label leakage
    # sozlesmesi evia_2021'den BIREBIR kopyalanmistir -- yalnizca AOI
    # (region_key), display_name, output_namespace ve aciklayici notes
    # farklidir.
    "evia_2021_extended": {
        "enabled": True,
        "region_key": "north_evia_extended",
        "display_name": "North Evia (Euboea), Greece -- extended AOI",
        "role": "mediterranean_transfer_wildfire",
        "country": "Greece",
        "predictor_start_date": "2021-06-05",
        "predictor_end_date": "2021-08-02",
        "label_start_date": "2021-08-03",
        "label_end_date": "2021-09-30",
        "baseline_years": [2017, 2018, 2019, 2020],
        "output_namespace": "evia_2021_extended",
        # Ayni jenerik, deklaratif leakage sozlesmesi (evia_2021 ile birebir;
        # yeni bir kod dali EKLENMEZ).
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2021-06-05", "2021-08-02"],
        "notes": (
            "Geographically EXTENDED counterpart of evia_2021: same event, "
            "same predictor/label windows and same baseline years, but a "
            "wider AOI (NORTH_EVIA_EXTENDED_AOI_BBOX) that geometrically "
            "CONTAINS the evia_2021 AOI and adds the surrounding unburned "
            "natural vegetation plus the Limni-Rovies-Agia Anna-Pefki north "
            "Evia corridor. AOI defined from geographic place coverage ONLY "
            "-- never tuned on burned labels, burned prevalence, gate "
            "outcome, AUC/PR-AUC or any model metric, and not selected from "
            "several trial geometries. evia_2021 remains registered and "
            "unchanged; this experiment writes exclusively to "
            "outputs/experiments/evia_2021_extended/. Must pass the same "
            "MCD64A1 burned-landcover gate before modeling; registration "
            "does NOT authorize downstream predictor/Step7/Step8/Step9/"
            "Step10 execution."
        ),
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