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

# Montiferru / Oristano, Sardinia (Italy) -- 24 July 2021 wildfire
# (bkz. EXPERIMENTS["montiferru_2021"]). Manavgat/Bejís/Muğla/Evia ile ayni
# "mediterranean_transfer_wildfire" ailesindendir.
#
# Bu bbox YALNIZCA YER (place) kapsamindan tanimlanmistir: dort capa belediye
# (comune) -- Bonarcado, Santu Lussurgiu, Cuglieri, Scano di Montiferro --
# TERRITORYALARI ile birlikte icerilir. KESIN yangin perimetri DEGILDIR ve
# yangin footprint'ine gore KIRPILMAMISTIR; MCD64A1 etiketlerine, burned
# prevalence'a, gate sonucuna, AUC/PR-AUC'ye veya herhangi bir model
# metrigine BAKILARAK ayarlanmamistir. Tek, deterministik aday olarak
# dondurulmustur (birden fazla geometri denenip en iyisi SECILMEMISTIR).
#
# Capa belediye merkezleri (yaklasik; yalnizca dokumantasyon amacli,
# hesaplamada KULLANILMAZ):
#   Cuglieri              ~ (lon 8.567, lat 40.183)
#   Bonarcado             ~ (lon 8.650, lat 40.150)
#   Santu Lussurgiu       ~ (lon 8.650, lat 40.217)
#   Scano di Montiferro   ~ (lon 8.583, lat 40.233)
# Kenar turetme yontemi (KANONIK; makine-okunur karsiligi
# EXPERIMENTS["montiferru_2021"]["aoi_provenance"] icindedir):
#   ISTAT 2021 non-generalized administrative boundaries (Limiti2021.zip,
#   referans tarihi 2021-12-31) arsivinden yukaridaki dort resmi comune
#   poligonu secilir, EPSG:4326'ya donusturulur, birlesimlerinin (union)
#   total bounds'u hesaplanir ve her kenar 2 ondalik dereceye DISA DOGRU
#   (outward) yuvarlanir:
#     raw union total bounds =
#       (8.454041028057148, 40.053824525866645,
#        8.745509975002108, 40.264205506896815)
#     outward round -> (8.45, 40.05, 8.75, 40.27) = MONTIFERRU_AOI_BBOX
#   Yani lon_min/lat_min asagi, lon_max/lat_max yukari yuvarlanir; ham
#   birlesim sinirlari her zaman final bbox'in ICINDE kalir.
# Step6B burned-landcover gate ile dogrulanmalidir -- tipki manavgat_2021 /
# bejis_2022 / mugla_2021 / evia_2021 gibi. Gate FAIL ederse AOI, populasyon
# veya esik DEGISTIRILMEZ.
MONTIFERRU_AOI_BBOX = (
    8.45,
    40.05,
    8.75,
    40.27,
)


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

    # --- Montiferru / Oristano 2021 (Akdeniz transfer wildfire, Italya) ---
    # Module-seviyesi MONTIFERRU_AOI_BBOX sabitinden (TEK KAYNAK) uretilir;
    # cografi gerekce o sabitin uzerinde belgelenmistir (dort capa belediye:
    # Bonarcado, Santu Lussurgiu, Cuglieri, Scano di Montiferro). KESIN yangin
    # perimetri DEGILDIR ve yangin footprint'ine gore kirpilmamistir; Step6B
    # burned-landcover gate ile dogrulanmalidir.
    montiferru_aoi_candidate_bbox = ee.Geometry.BBox(*MONTIFERRU_AOI_BBOX)
    montiferru_aoi = montiferru_aoi_candidate_bbox

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
        "montiferru_aoi": montiferru_aoi,
        "montiferru_aoi_candidate_bbox": montiferru_aoi_candidate_bbox,
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
        # kozan_2023 is NOT a legacy variant: it is the canonical
        # cropland/stubble-burning negative control and stays selectable.
        # It is filtered out of natural-vegetation cohort discovery by its
        # ROLE ("negative_control"), never by variant_status.
        "variant_status": "canonical",
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
        "variant_status": "canonical",
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
        "variant_status": "canonical",
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
        "variant_status": "canonical",
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

    # --- HISTORICAL / SUPERSEDED: provisional calendar-shift attempt ---------
    # mugla_2021 ile AYNI MEKAN, YALNIZCA BIR YIL SONRAKI ZAMAN PENCERESI
    # (takvim kaydirmasi / calendar shift). Bu formulasyon, danisman
    # (supervisor) protokolun OLAY-GORELI (event-relative) oldugunu
    # netlestirdikten SONRA bilimsel olarak GECERSIZ KILINMISTIR; yerini
    # mugla_2022_event_relative almistir (bkz. superseded_by).
    #
    # Kayit SILINMEZ ve outputs/experiments/mugla_2022/ altindaki calendar-
    # shift gate denemesinin ciktilari ASLA overwrite/migrate EDILMEZ --
    # tarihsel kayit olarak oldugu gibi kalir. variant_status yalnizca bu
    # kaydi YENI kanonik secimlerden cikarir (evia_2021 ile ayni mekanizma).
    #
    # AOI SABITTIR: ayni region_key ("mugla_aoi") uzerinden MEVCUT Muğla
    # geometrisi (module sabiti MUGLA_AOI_BBOX -> (27.10, 36.60, 28.90, 37.45))
    # YENIDEN KULLANILIR -- yeni bir bbox/geometri TANIMLANMAZ.
    "mugla_2022": {
        "enabled": True,
        "variant_status": "legacy_superseded",
        "superseded_by": "mugla_2022_event_relative",
        "region_key": "mugla_aoi",
        "display_name": "Muğla 2022",
        # Temporal (ayni AOI, farkli yil) transfer deneyi: mekan sabit
        # tutuldugunda transfer basarisizliginin YILA/OLAYA mi bagli oldugunu
        # test eder. Yeni bir kod dali gerektirmez -- rol yalnizca
        # aciklayici metadata olarak tasinir (bkz. list_canonical_enabled_
        # experiments(role=...) jenerik filtresi).
        "role": "temporal_transfer_wildfire",
        "country": "Turkey",
        "predictor_start_date": "2022-06-01",
        "predictor_end_date": "2022-07-28",
        "label_start_date": "2022-07-29",
        "label_end_date": "2022-09-15",
        "baseline_years": [2018, 2019, 2020, 2021],
        "output_namespace": "mugla_2022",
        # Ayni jenerik, deklaratif leakage sozlesmesi (mugla_2021 / evia_2021
        # ile ayni mekanizma; yeni bir kod dali EKLENMEZ): label rasteri label
        # penceresine DOY-maskeli oldugu icin, label_start ONCESI yanan
        # hucreler ayri bir pre-label BurnDate rasteri ile analiz evreninden
        # DISLANIR (unburned negatif olarak SAYILMAZ).
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2022-06-01", "2022-07-28"],
        # NOT: mugla_2021'deki `pre_label_diagnostic_window` BILEREK
        # kopyalanmamistir -- o alan Bördübet/Marmaris (~2021-06-21..25)
        # yangina ozgu bir TESHIS alt-penceresidir ve 2022 icin boyle bir
        # onceden belirlenmis olay yoktur. Alan yoksa
        # scripts/run_label_gate_only.py jenerik olarak None gecer.
        "notes": (
            "SUPERSEDED (historical calendar-shift attempt; retained, never "
            "deleted or migrated). This was the PROVISIONAL calendar-shift "
            "formulation of Muğla 2022: mugla_2021's windows shifted forward by "
            "exactly one calendar year (predictor 2022-06-01..2022-07-28, label "
            "2022-07-29..2022-09-15, baseline 2018-2021) on the identical AOI "
            "(region_key='mugla_aoi', MUGLA_AOI_BBOX). It is SCIENTIFICALLY "
            "SUPERSEDED after the supervisor clarified that the protocol is "
            "EVENT-RELATIVE: the windows must be anchored on the 2022 fire "
            "ignition date, not on the 2021 calendar dates. The canonical "
            "replacement is mugla_2022_event_relative (see superseded_by), "
            "which writes to its OWN namespace. The calendar-shift gate attempt "
            "under outputs/experiments/mugla_2022/ is PRESERVED as-is and is "
            "never overwritten, migrated, or reused as input for the "
            "event-relative experiment. mugla_2021 remains untouched."
        ),
    },

    # --- CANONICAL: event-relative, same-geography event-to-event -----------
    # Replaces the calendar-shift attempt above (mugla_2022). SAME AOI as
    # mugla_2021/mugla_2022 -- region_key="mugla_aoi", the existing
    # MUGLA_AOI_BBOX geometry is REUSED; no new bbox/geometry is defined.
    # Windows are anchored on the 2022 dominant MCD64A1 event ignition
    # (2022-06-21) instead of on 2021's calendar dates, preserving the
    # same-AOI 2021 window DURATIONS exactly (58-day predictor, 49-day label).
    #
    # Writes exclusively to outputs/experiments/mugla_2022_event_relative/;
    # the frozen outputs/experiments/mugla_2022/ artifacts are never read as
    # input, overwritten, or migrated.
    "mugla_2022_event_relative": {
        "enabled": True,
        "variant_status": "canonical",
        "region_key": "mugla_aoi",
        "display_name": "Muğla 2022 -- event-relative",
        # Role is deliberately KEPT as temporal_transfer_wildfire: existing
        # discovery logic (src/burned_pattern_audit.py NON_COHORT_ROLES and the
        # generic role filters) excludes this role from the canonical
        # five-AOI spatial/domain cohort. Changing the role would silently
        # enrol a SECOND year of an AOI that is already represented
        # (mugla_2021) into a spatial comparison. The role name is descriptive
        # metadata only -- see transfer_framing below for the actual claim.
        "role": "temporal_transfer_wildfire",
        "country": "Turkey",
        "predictor_start_date": "2022-04-24",
        "predictor_end_date": "2022-06-20",
        "label_start_date": "2022-06-21",
        "label_end_date": "2022-08-08",
        "baseline_years": [2018, 2019, 2020, 2021],
        "output_namespace": "mugla_2022_event_relative",
        # --- Event-relative framing (explicit, machine-readable) -------------
        # This is NOT a pure temporal-transfer experiment: YEAR and SEASONAL
        # PHASE are confounded (the 2022 event ignites ~5 weeks earlier in the
        # season than the 2021 event), so a performance difference cannot be
        # attributed to the year alone. The intended claim is SAME-GEOGRAPHY
        # EVENT-TO-EVENT transfer.
        "transfer_framing": "same_geography_event_to_event",
        "event_anchor_date": "2022-06-21",
        "event_anchor_basis": (
            "dominant MCD64A1 event identified independently before window "
            "revision"
        ),
        "event_window_rule": (
            "same-AOI 2021 duration preserved: 58-day predictor ending one day "
            "before ignition; 49-day label beginning on ignition"
        ),
        # --- Leakage-safe PRE-LABEL exclusion (same generic, declarative
        # contract as mugla_2021 / evia_2021 / montiferru_2021; no new code
        # branch). Cells that burned inside THIS experiment's predictor window
        # (before label_start) leave post-fire predictor signatures and are
        # removed from the analysis universe -- never counted as unburned. ---
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2022-04-24", "2022-06-20"],
        # --- HISTORICAL (2021) BURN EXCLUSION (separate, independent contract)
        # A cell that burned in the 2021 Muğla event is excluded from the 2022
        # analysis universe entirely, because its 2022 NDVI/LST/TVDI may
        # reflect POST-FIRE RECOVERY rather than a normal vegetation state.
        # This is a DIFFERENT exclusion axis from the pre-label one above: the
        # two are derived independently, reported separately, and unioned by
        # the gate (a cell may carry both flags). Generic, declarative fields
        # -- no code path keys off this experiment_id.
        #
        # Source mask definition is EXACTLY `burned == 1` in the source
        # experiment's canonical Step8A parquet; it is NOT additionally
        # restricted by TSG, analysis_eligible, valid_for_modeling or
        # landcover. The source artifact is READ-ONLY and never rewritten.
        "exclude_historical_burns": True,
        "historical_burn_source_experiment": "mugla_2021",
        "historical_burn_source_kind": "canonical_step8a_physical_burned_cells",
        # Frozen expectation for the CURRENT canonical source artifact; the
        # manifest builder fails closed if the source yields a different count
        # (a silently changed/regenerated source must never pass unnoticed).
        "historical_burn_source_expected_count": 3073,
        # --- POST-GATE primary-population sample-size rule (SEPARATE from the
        # generic burned-landcover gate's min_positives=30, which stays the
        # total-burned feasibility threshold and is NOT modified). Evaluated
        # AFTER all exclusions + landcover classification, on the primary
        # population only. Falling short is a STOP for downstream
        # authorization -- it is NOT a failure of the natural/cropland
        # composition gate. ---
        "primary_population": "burnable_tree_shrub_grass",
        "min_primary_population_burned": 300,
        "notes": (
            "Muğla 2022 EVENT-RELATIVE (canonical; supersedes the provisional "
            "calendar-shift record mugla_2022). Same AOI as mugla_2021 "
            "(region_key='mugla_aoi', MUGLA_AOI_BBOX -- reused, not "
            "redefined). Windows are anchored on the 2022 dominant MCD64A1 "
            "event ignition date 2022-06-21, which was identified "
            "INDEPENDENTLY, BEFORE the window revision, and preserve the "
            "same-AOI 2021 window durations exactly: a 58-day predictor "
            "(2022-04-24..2022-06-20, ending one day before ignition) and a "
            "49-day label (2022-06-21..2022-08-08, beginning on ignition). "
            "FRAMING: this is NOT a pure temporal-transfer experiment. Year "
            "and seasonal phase are CONFOUNDED -- the 2022 event ignites "
            "earlier in the season than the 2021 event, so predictor/label "
            "windows sit in a different seasonal phase as well as a different "
            "year, and no year-only attribution is possible. The intended "
            "claim is SAME-GEOGRAPHY EVENT-TO-EVENT transfer "
            "(transfer_framing='same_geography_event_to_event'). Two "
            "INDEPENDENT exclusion contracts apply and are unioned by the "
            "gate: (1) pre-label burns inside this experiment's own predictor "
            "window, and (2) historical 2021 burns (all burned==1 cells of "
            "mugla_2021's canonical Step8A dataset), excluded because their "
            "2022 vegetation/thermal state may reflect post-fire recovery. "
            "The frozen outputs/experiments/mugla_2022/ calendar-shift "
            "artifacts are never overwritten, migrated or reused here. "
            "Registration does NOT authorize downstream predictor/Step7/Step8/"
            "Step9/Step10 execution; the same MCD64A1 burned-landcover gate "
            "must pass first, AND the separate primary-population "
            "sample-size rule (>= 300 burned burnable_tree_shrub_grass cells) "
            "must be satisfied."
        ),
    },

    "evia_2021": {
        "enabled": True,
        # --- FROZEN VARIANT DECISION -------------------------------------
        # evia_2021 is the LEGACY North Evia AOI variant, superseded by
        # evia_2021_extended (same event, same predictor/label windows, same
        # baseline years, same predictor/label/model contract -- the ONLY
        # contract difference is AOI geometry). It stays `enabled` and fully
        # registered, and its frozen outputs under
        # outputs/experiments/evia_2021/ are NEVER deleted or overwritten;
        # `variant_status` only removes it from NEW canonical cohort
        # selections (discovery mode, cross-region synthesis, multi-AOI
        # univariate comparison). Explicitly requesting it fails closed with
        # the superseded_by pointer -- it is never silently dropped.
        # Quantitative variant comparison: docs/evia_2021_prevalence_audit.json
        "variant_status": "legacy_superseded",
        "superseded_by": "evia_2021_extended",
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
        # Canonical North Evia variant; supersedes evia_2021 (see that
        # record's `superseded_by`). A canonical record must NEVER carry a
        # `superseded_by` key -- see validate_variant_record().
        "variant_status": "canonical",
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

    "montiferru_2021": {
        "enabled": True,
        "variant_status": "canonical",
        "region_key": "montiferru_aoi",
        "display_name": "Montiferru / Oristano, Sardinia 2021",
        "role": "mediterranean_transfer_wildfire",
        "country": "Italy",
        "predictor_start_date": "2021-05-25",
        "predictor_end_date": "2021-07-23",
        "label_start_date": "2021-07-24",
        "label_end_date": "2021-08-31",
        "baseline_years": [2017, 2018, 2019, 2020],
        "output_namespace": "montiferru_2021",
        "exclude_pre_label_burns": True,
        "pre_label_burn_window": ["2021-05-25", "2021-07-23"],
        # Kanonik, JSON-serializable AOI turetme zinciri. MONTIFERRU_AOI_BBOX
        # bu kaydin "final_bbox_epsg4326" alanina birebir esittir; bbox'in
        # kendisi TEK KAYNAK olarak modul sabitinde kalir. Bu dict gate
        # manifestine scientific.aoi.derivation olarak tasinir ve boylece
        # analysis_id'nin parcasi olur.
        "aoi_provenance": {
            "source_authority": "ISTAT",
            "source_dataset": "2021 non-generalized administrative boundaries",
            "source_archive": "Limiti2021.zip",
            "source_reference_date": "2021-12-31",
            "administrative_level": "municipality/comune",
            "municipalities": [
                "Bonarcado",
                "Cuglieri",
                "Santu Lussurgiu",
                "Scano di Montiferro",
            ],
            "derivation_method": (
                "Select the four official municipality polygons, transform to "
                "EPSG:4326, compute their union total bounds, then outward-round "
                "each edge to two decimal degrees."
            ),
            "raw_union_total_bounds_epsg4326": [
                8.454041028057148,
                40.053824525866645,
                8.745509975002108,
                40.264205506896815,
            ],
            "rounding": {
                "mode": "outward",
                "decimal_places": 2,
            },
            "final_bbox_epsg4326": [
                8.45,
                40.05,
                8.75,
                40.27,
            ],
            "selection_constraint": (
                "Defined only from official municipal boundaries; not tuned "
                "using the fire footprint, MCD64A1 labels, prevalence, gate "
                "outcome or model metrics."
            ),
        },
        "notes": (
            "Montiferru / Oristano 2021 wildfire AOI covering Bonarcado, "
            "Santu Lussurgiu, Cuglieri and Scano di Montiferro. The AOI is "
            "defined from geographic place coverage only and was not tuned on "
            "MCD64A1 labels, burned prevalence, gate outcome or model metrics. "
            "Mixed forest, olive-grove and grassland landscape; the canonical "
            "burned-landcover gate may fail. Gate passage does not authorize "
            "downstream execution."
        ),
    },
}

# Geriye donuk uyumluluk: mevcut pipeline'in varsayilan deneyi.
DEFAULT_EXPERIMENT_ID = "kozan_2023"


# =============================================================================
# 2b) Variant status (canonical / legacy_superseded) -- generic, fail-closed
# =============================================================================
# Bir deney kaydinin AOI/kontrat varyant durumu. YALNIZCA iki deger gecerlidir;
# bilinmeyen veya eksik bir deger HATA verir (fail-closed) -- boylece yeni bir
# deney kaydi eklendiginde varyant durumunu belirtmeyi unutmak sessizce
# "canonical" varsayilmaz.
#
#   canonical          : yeni cohort/synthesis secimlerinde kullanilabilir.
#   legacy_superseded  : baska bir deney tarafindan gecersiz kilinmistir;
#                        kaydi ve ciktilari SILINMEZ/OVERWRITE EDILMEZ, ama
#                        yeni kanonik secimlere GIRMEZ. Acikca istenirse
#                        superseded_by isaretcisiyle fail-closed hata verir.
#
# ONEMLI: `variant_status` `enabled` alanindan BAGIMSIZDIR. legacy_superseded
# bir deney hala enabled kalabilir (tekil, acik, geriye donuk analizler icin);
# bu iki alan birbirinin yerine KULLANILMAZ.
#
# ONEMLI: `role` de `variant_status`tan BAGIMSIZDIR. kozan_2023 canonical bir
# kayittir; dogal-vejetasyon cohort'larindan `role == "negative_control"`
# oldugu icin ayrilir, legacy oldugu icin DEGIL.
VARIANT_STATUS_CANONICAL = "canonical"
VARIANT_STATUS_LEGACY_SUPERSEDED = "legacy_superseded"
ALLOWED_VARIANT_STATUSES = (
    VARIANT_STATUS_CANONICAL,
    VARIANT_STATUS_LEGACY_SUPERSEDED,
)


class VariantStatusError(ValueError):
    """Raised when a registry record's variant contract is invalid or when a
    superseded experiment is requested where only canonical ones are allowed.

    Subclasses ValueError so existing callers that already catch ValueError
    from `get_experiment()` keep working unchanged.
    """


def validate_variant_record(record: dict, experiment_id: str) -> str:
    """Validate ONE registry record's variant contract and return its status.

    Pure function over a record dict -- it does not consult EXPERIMENTS, so
    callers that already hold a record (e.g. from `list_experiments()`) can
    validate it without a second registry lookup.

    Rules (all fail-closed):
        - `variant_status` must be present and one of ALLOWED_VARIANT_STATUSES;
          missing or unknown values raise (never defaulted to canonical).
        - `legacy_superseded` requires a `superseded_by` naming a DIFFERENT,
          registered experiment.
        - `canonical` must NOT carry a `superseded_by` key at all.

    The `superseded_by` target is resolved against EXPERIMENTS only when that
    registry actually contains `experiment_id`, so record-level validation of
    synthetic/test records stays self-contained.
    """
    status = record.get("variant_status")
    if status is None:
        raise VariantStatusError(
            f"'{experiment_id}': registry record has no 'variant_status'. "
            f"Every experiment must declare one of {list(ALLOWED_VARIANT_STATUSES)} "
            "explicitly (missing status is never defaulted to canonical)."
        )
    if status not in ALLOWED_VARIANT_STATUSES:
        raise VariantStatusError(
            f"'{experiment_id}': unknown variant_status {status!r}. "
            f"Allowed values: {list(ALLOWED_VARIANT_STATUSES)}."
        )

    superseded_by = record.get("superseded_by")
    if status == VARIANT_STATUS_LEGACY_SUPERSEDED:
        if not superseded_by or not isinstance(superseded_by, str):
            raise VariantStatusError(
                f"'{experiment_id}': variant_status='{VARIANT_STATUS_LEGACY_SUPERSEDED}' "
                "requires a non-empty 'superseded_by' experiment_id."
            )
        if superseded_by == experiment_id:
            raise VariantStatusError(
                f"'{experiment_id}': 'superseded_by' must not point at itself."
            )
        
        # Only enforce successor registry constraints when this record is itself
        # a real registry entry; synthetic records validate standalone.
        if experiment_id in EXPERIMENTS:
            if superseded_by not in EXPERIMENTS:
                raise VariantStatusError(
                    f"'{experiment_id}': superseded_by='{superseded_by}' is not a "
                    f"registered experiment. Valid IDs: "
                    f"{', '.join(sorted(EXPERIMENTS))}."
                )

            successor_status = EXPERIMENTS[superseded_by].get("variant_status")
            if successor_status != VARIANT_STATUS_CANONICAL:
                raise VariantStatusError(
                    f"'{experiment_id}': superseded_by='{superseded_by}' "
                    "must reference a canonical experiment."
                )
        
    elif "superseded_by" in record:
        raise VariantStatusError(
            f"'{experiment_id}': variant_status='{VARIANT_STATUS_CANONICAL}' "
            f"must not carry 'superseded_by' (found {superseded_by!r}). "
            "Only superseded records point at a successor."
        )
    return status


def get_variant_status(experiment_id: str) -> str:
    """Return the validated variant status for `experiment_id`.

    Raises:
        ValueError: experiment_id is not registered.
        VariantStatusError: the record's variant contract is invalid
            (missing/unknown status, superseded without a valid
            `superseded_by`, or canonical carrying one).
    """
    record = get_experiment(experiment_id)
    return validate_variant_record(record, experiment_id)


def get_superseded_by(experiment_id: str) -> Optional[str]:
    """Return the successor experiment_id for a superseded experiment, or
    None when `experiment_id` is canonical. Validates the record first."""
    record = get_experiment(experiment_id)
    status = validate_variant_record(record, experiment_id)
    if status != VARIANT_STATUS_LEGACY_SUPERSEDED:
        return None
    return record["superseded_by"]


def assert_not_superseded_experiments(
    experiment_ids, context: str,
) -> tuple[str, ...]:
    """Fail closed if any caller-supplied experiment_id is superseded.

    Generic guard for entry points that accept an EXPLICIT experiment list.
    A superseded ID is NEVER silently dropped -- doing so would quietly change
    the scientific cohort behind the caller's back. The raised error names the
    successor so the caller can fix the request.

    Every ID is also variant-validated, so an unknown/missing status fails
    here too.

    Args:
        experiment_ids: iterable of experiment IDs (validated in order).
        context: short human-readable name of the calling entry point, echoed
            in the error message (e.g. "multi-AOI transfer synthesis").

    Returns:
        The validated IDs as a tuple, unchanged.

    Raises:
        ValueError: an ID is unknown.
        VariantStatusError: an ID is superseded or has an invalid record.
    """
    validated: list[str] = []
    offenders: list[str] = []
    for experiment_id in experiment_ids:
        status = get_variant_status(experiment_id)  # ValueError if unknown
        if status == VARIANT_STATUS_LEGACY_SUPERSEDED:
            successor = EXPERIMENTS[experiment_id]["superseded_by"]
            offenders.append(f"'{experiment_id}' (superseded_by='{successor}')")
        validated.append(experiment_id)
    if offenders:
        raise VariantStatusError(
            f"{context}: refusing superseded experiment(s): {', '.join(offenders)}. "
            f"Only variant_status='{VARIANT_STATUS_CANONICAL}' experiments may enter a "
            "new canonical selection. Legacy outputs stay on disk and are never "
            "deleted; use the successor experiment_id instead."
        )
    return tuple(validated)


def list_canonical_enabled_experiments(role: Optional[str] = None) -> dict:
    """Return the enabled + canonical experiments, optionally filtered by role.

    Generic replacement for ad-hoc `enabled minus <hardcoded legacy id>`
    filtering. No experiment ID is hard-coded here.

    Args:
        role: when given, keep only records whose "role" equals it (e.g.
            "negative_control" or "mediterranean_transfer_wildfire"). When
            None, every role is kept -- including negative controls, which
            callers building natural-vegetation cohorts must exclude
            themselves by role (kozan_2023 is canonical, not legacy).

    Raises:
        VariantStatusError: any enabled record has an invalid variant contract.
    """
    selected: dict = {}
    for experiment_id, record in EXPERIMENTS.items():
        if not record.get("enabled"):
            continue
        status = validate_variant_record(record, experiment_id)
        if status != VARIANT_STATUS_CANONICAL:
            continue
        if role is not None and record.get("role") != role:
            continue
        selected[experiment_id] = dict(record, experiment_id=experiment_id)
    return selected


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