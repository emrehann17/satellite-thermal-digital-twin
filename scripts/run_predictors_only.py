"""
run_predictors_only.py

Step0D: deney-farkinda (experiment-aware) predictor uretim calistiricisi.

Yalnizca Step1-Step5/5C'nin (Landsat LST/NDVI current+baseline + Step5 termal
anomaly + Step5C TVDI/dryness) namespaced/experiment-aware calistirilmasini
saglar. Step7 VE Step8'i KESINLIKLE CALISTIRMAZ, model EGITMEZ.

IKI FARKLI DAVRANIS MODU
-------------------------
kozan_2023:
    Legacy davranis. `--export` ile Step3->Step4->Step4b->Step5->Step5C
    legacy zincirini (core/config.py sabitleriyle, namespace'siz) calistirir;
    `--local-only` ile yalnizca Step5->Step5C'yi (girdi GeoTIFF'lerin zaten
    var oldugu varsayilarak) calistirir. Kozan'in legacy paylasilan
    dosyalarina ETKISI, scripts/main.py'nin Step3-5C bolumuyle AYNIDIR.

kozan_2023 DISINDAKI her deney (or. manavgat_2021):
    TAMAMEN NAMESPACED calisir (bkz. core/experiment_context.py):
        outputs/experiments/<experiment_id>/data/...
        outputs/experiments/<experiment_id>/step5/
        outputs/experiments/<experiment_id>/step5b/
        outputs/experiments/<experiment_id>/step5c/
    `--export`, Step4/Step4b'nin Drive-export+polling+download zincirini
    REPLIKE ETMEZ -- bunun yerine Step3'un zaten parametrik olan GEE
    fonksiyonlarini (get_current_period_median, get_current_period_ndvi_median,
    get_landsat_baseline_window_median_collection,
    get_landsat_baseline_window_ndvi_collection) DOGRUDAN, geemap.ee_export_image
    ile yerel diske (Drive'a ugramadan) export eder -- Step6/Step6A'da zaten
    kullanilan ayni desen. `--local-only`, bu dosyalarin zaten var oldugunu
    varsayar ve yalnizca Step5->Step5C'yi (step5.run_step5(ctx) /
    step5c.run_step5c(ctx)) calistirir.

    Bazi AOI/urun kombinasyonlari (or. Manavgat'in genis AOI'si + tam
    cozunurluklu current-period LST) GEE'nin senkron getPixels indirme
    boyutu limitini (~50 MB) asabilir. Bu durumda `export_image_direct_or_tiled()`
    otomatik olarak 2x2 -> 4x4 -> 6x6 tile grid'ine kademeli olarak geri
    duser (bkz. asagida); COZUNURLUK ASLA DUSURULMEZ (hala 30 m), yalnizca
    AOI parcalara bolunup ayri ayri indirilip rasterio.merge ile birlestirilir.

Her Kozan-disi calistirmadan once, TUM hesaplanan yollarin
outputs/experiments/<experiment_id>/ altinda kaldigi VE legacy Kozan paylasilan
dizinleriyle (data/current_period, data/baseline_period/landsat_timeseries,
data/ndvi_*, outputs/step5, outputs/step5c) CAKISMADIGI dogrulanir
(_assert_paths_are_safely_namespaced). Ihlalde hicbir export/isleme calismaz.

CLI:
    python scripts/run_predictors_only.py --experiment kozan_2023 --dry-run
    python scripts/run_predictors_only.py --experiment manavgat_2021 --dry-run
    python scripts/run_predictors_only.py --experiment manavgat_2021 --export --force
    python scripts/run_predictors_only.py --experiment manavgat_2021 --local-only --force
    python scripts/run_predictors_only.py --experiment manavgat_2021 --export --force --cleanup-tiles
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.experiment_context import build_experiment_context, get_region, log_context_summary
from core.io_utils import setup_logger

log, log_file = setup_logger("run_predictors_only")

BASE_DIR = _PROJECT_ROOT

# Kademeli tile grid denemeleri: 2x2 yetmezse 4x4, o da yetmezse 6x6, 8x8.
_TILE_GRID_ESCALATION = [(2, 2), (4, 4), (6, 6), (8, 8)]

# GEE'nin senkron getPixels/download (Image.getDownloadURL / geemap tabanli
# direct export) icin sabit sunucu-taraf boyut siniri. Bu, projede uc kez
# gozlemlenen hatanin ("Total request size (NNNNNNNN bytes) must be <=
# 50331648 bytes.") sabit ustsinirdir -- 81561538, 69438600 ve 63226200
# byte'lik farkli istekler HEPSI ayni 50331648 byte sinirina carpmistir.
GEE_DIRECT_DOWNLOAD_LIMIT_BYTES = 50_331_648  # 48 MiB

# ONFLIGHT (pre-download) tahmin icin KONSERVATIF esik: sabit sinirin
# ALTINDA, kasti bir marj birakir. Marj GEREKLIDIR cunku bu tahmin GEE'nin
# gercek encode edilmis indirme boyutunu DEGIL, basit bir piksel-sayisi x
# byte/piksel yaklasimini kullanir -- ve gozlemlenen veriyle karsilastirma
# bunun bazen GERCEK boyutu KUCUMSEDIGINI gosteriyor (bkz. Mugla reference_30m
# ornegi: int16 kabulu ~42 MB tahmin ederken gercek istek 63.2 MB olmustu).
# Bu yuzden varsayilan tahmin, dtype ne olursa olsun 4 byte/piksel (float32
# esdegeri) kullanir (bkz. DEFAULT_ESTIMATE_BYTES_PER_PIXEL) VE esik sabit
# sinirin %80'inde tutulur -- boylece sinira yakin/belirsiz durumlarda
# guvenli tarafta kalinir. Bu tahmin yalnizca bir ON-FILTRE'dir (gereksiz
# bir "once dogrudan dene, basarisiz olacagini bil" agi cagrisindan
# kacinmak icin); asagidaki mevcut reaktif dogrulama (_file_ok + tiled
# fallback) HALA nihai guvenlik agidir ve tahmin yanlis olsa bile dogru
# sonucu garanti eder.
DIRECT_EXPORT_SAFE_THRESHOLD_BYTES = int(GEE_DIRECT_DOWNLOAD_LIMIT_BYTES * 0.80)  # ~38.4 MiB

# GEE, cografi (EPSG:4326 gibi) bir CRS'e metrik bir "scale" ile export
# yaparken, dereceyi enleme gore (cos(lat)) AYARLAMAZ -- ekvatoral bir
# metre/derece sabiti KULLANIR (hem lon hem lat ekseninde AYNI). Bu, yalnizca
# ON-FILTRE tahmini icin kullanilir; gercek export'un piksel gridini HALA
# GEE'nin kendisi belirler (bu sabit onu DEGISTIRMEZ/ETKILEMEZ).
_ESTIMATE_METERS_PER_DEGREE = 111_319.49

# Tahminde varsayilan byte/piksel: dtype'a bakmaksizin float32-esdegeri (4
# byte). Cagiran taraf gercek dtype'ini/bant sayisini biliyorsa acikca
# gecebilir; ama yukaridaki gozlemlenen kucumsenme riski nedeniyle bu proje
# icin varsayilan KUCUK bir dtype (or. int16/uint8) VARSAYMAZ.
DEFAULT_ESTIMATE_BYTES_PER_PIXEL = 4

# Legacy Kozan paylaşılan dizinleri: Kozan-dışı deneyler bunlara ASLA yazamaz.
_LEGACY_SHARED_DIRS = [
    (BASE_DIR / "data" / "landsat_timeseries").resolve(),
    (BASE_DIR / "data" / "landsat_qa").resolve(),
    (BASE_DIR / "data" / "current_period").resolve(),
    (BASE_DIR / "data" / "modis").resolve(),
    (BASE_DIR / "data" / "ndvi_timeseries").resolve(),
    (BASE_DIR / "data" / "ndvi_current_period").resolve(),
    (BASE_DIR / "outputs" / "step5").resolve(),
    (BASE_DIR / "outputs" / "step5b").resolve(),
    (BASE_DIR / "outputs" / "step5c").resolve(),
    (BASE_DIR / "outputs" / "validation" / "labels").resolve(),
]


class PredictorRunnerError(SystemExit):
    """Fail-fast error for this runner (diğer step'lerle aynı konvansiyon)."""


def _assert_paths_are_safely_namespaced(ctx: dict) -> None:
    """
    GÜVENLİK KONTROLÜ (Kozan-dışı deneyler için ZORUNLU):
        1) Hiçbir hesaplanan yol legacy paylaşılan dizinlerin (yukarıda)
           altına DÜŞMEMELİDİR.
        2) Yazılabilir TÜM yollar outputs/experiments/<experiment_id>/
           altında OLMALIDIR.
    İhlal varsa PredictorRunnerError fırlatır; hiçbir export/işleme çalışmaz.
    """
    experiment_id = ctx["experiment_id"]
    experiments_root = (BASE_DIR / "outputs" / "experiments" / experiment_id).resolve()

    check_keys = [
        "data_root", "baseline_input_dir", "qa_dir", "current_period_dir",
        "modis_input_dir", "ndvi_baseline_dir", "ndvi_current_dir",
        "step5_output_dir", "step5b_output_dir", "step5c_output_dir",
        "gate_labels_dir",
    ]
    for key in check_keys:
        p = ctx.get(key)
        if p is None:
            continue
        resolved = Path(p).resolve()

        for legacy_dir in _LEGACY_SHARED_DIRS:
            if resolved == legacy_dir or legacy_dir in resolved.parents:
                raise PredictorRunnerError(
                    f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için hesaplanan "
                    f"'{key}' yolu ({resolved}) Kozan'ın legacy paylaşılan "
                    f"dizinine ({legacy_dir}) düşüyor. Bu deney bu dizine ASLA "
                    "yazamaz/okuyamaz. İşlem DURDURULDU."
                )
        if resolved != experiments_root and experiments_root not in resolved.parents:
            raise PredictorRunnerError(
                f"GÜVENLİK İHLALİ: '{experiment_id}' deneyi için hesaplanan "
                f"'{key}' yolu ({resolved}) outputs/experiments/{experiment_id}/ "
                "dışında. İşlem DURDURULDU."
            )


def _log_planned_paths(ctx: dict) -> None:
    log.info("  data_root: %s", ctx["data_root"])
    log.info("  baseline_input_dir (Landsat LST, yıllık): %s", ctx["baseline_input_dir"])
    log.info("  current_period_dir (Landsat LST current): %s", ctx["current_period_dir"])
    log.info("  ndvi_baseline_dir: %s", ctx["ndvi_baseline_dir"])
    log.info("  ndvi_current_dir: %s", ctx["ndvi_current_dir"])
    log.info("  step5_output_dir: %s", ctx["step5_output_dir"])
    log.info("  step5b_output_dir: %s", ctx["step5b_output_dir"])
    log.info("  step5c_output_dir: %s", ctx["step5c_output_dir"])


# =============================================================================
# Tiled export fallback (GEE senkron getPixels boyut limiti asilirsa)
# =============================================================================
def _is_size_related_error(exc: Exception) -> bool:
    """
    GEE'nin senkron getPixels/download boyut limiti hatalarini tanir
    (or. "Total request size (69438600 bytes) must be <= 50331648 bytes.").
    Baska tur hatalar (auth, geometry, bant adi vb.) icin False doner --
    bunlar normal sekilde firlatilmaya devam eder (sessizce yutulmaz).

    ONEMLI: Bu fonksiyon artik SADECE tile-grid ESKALASYONUNDA (bir sonraki
    daha ince grid'e gecilsin mi, yoksa hata direkt firlatilsin mi) kullanilir.
    Direct export'un fallback'e DUSUP DUSMEYECEGINE bu fonksiyon KARAR VERMEZ
    -- bkz. export_image_direct_or_tiled(): dosyanin gercekten var olup
    olmadigi/boyutunun >0 olup olmadigi tek basina yeterli sinyaldir, cunku
    geemap.ee_export_image bazen hata firlatmadan (yalnizca konsola log
    basarak) sessizce basarisiz olabilir.
    """
    msg = str(exc).lower()
    size_markers = (
        "request size", "must be <=", "getpixels", "download size",
        "total request", "user memory limit", "image.getdownloadurl",
    )
    return any(marker in msg for marker in size_markers)


class TiledExportError(PredictorRunnerError):
    """Tek bir tile'in export'u basarisiz oldugunda (dosya yok/0 byte) firlatilir.

    Row/col/bounds bilgisini tasir ki disaridaki grid-eskalasyon dongusu
    (export_image_direct_or_tiled) bunu daha ince bir grid ile tekrar
    denemek icin kullanabilsin.
    """

    def __init__(self, label: str, r: int, c: int, bounds: tuple, tile_path: Path, detail: str = ""):
        self.label = label
        self.r = r
        self.c = c
        self.bounds = bounds
        self.tile_path = tile_path
        message = (
            f"[{label}] tile (r{r},c{c}) bounds={bounds} export basarisiz "
            f"(dosya yok/bos): {tile_path}. {detail}"
        )
        super().__init__(message)


def _bbox_from_region(region) -> tuple[float, float, float, float]:
    """region.bounds().getInfo() üzerinden (xmin, ymin, xmax, ymax) döner."""
    bounds_info = region.bounds().getInfo()
    coords = bounds_info["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _tile_bboxes(xmin: float, ymin: float, xmax: float, ymax: float, rows: int, cols: int):
    """(xmin,ymin,xmax,ymax) dikdörtgenini rows x cols eşit alt-bbox'a böler."""
    x_edges = [xmin + (xmax - xmin) * i / cols for i in range(cols + 1)]
    y_edges = [ymin + (ymax - ymin) * i / rows for i in range(rows + 1)]
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append(
                {
                    "r": r, "c": c,
                    "bbox": (x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1]),
                }
            )
    return tiles


# =============================================================================
# On-flight (pre-download) boyut tahmini -- GEE'ye HIC dokunmadan, saf Python
# =============================================================================
def _estimate_pixel_grid(
    xmin: float, ymin: float, xmax: float, ymax: float, scale_m: float,
    meters_per_degree: float = _ESTIMATE_METERS_PER_DEGREE,
) -> tuple[int, int]:
    """
    GEE'nin cografi (EPSG:4326) bir export icin URETECEGI piksel gridinin
    KONSERVATIF bir tahminini, GEE'ye HIC dokunmadan (tamamen saf Python)
    hesaplar. Deterministiktir: ayni girdiler HER ZAMAN ayni sonucu verir.

    NOT: Bu, GEE'nin gercek dahili grid hesaplamasinin BIREBIR AYNISI
    DEGILDIR (kesin piksel sinirlari GEE'nin kendi reproject/snap mantigina
    baglidir) -- yalnizca direct-vs-tiled KARARI icin yeterince dogru,
    konservatif bir ON-FILTREDIR. Gercek export HER ZAMAN GEE'nin kendi
    grid mantigini kullanir; bu fonksiyon onu degistirmez.
    """
    deg_per_pixel = scale_m / meters_per_degree
    width_px = max(1, math.ceil((xmax - xmin) / deg_per_pixel))
    height_px = max(1, math.ceil((ymax - ymin) / deg_per_pixel))
    return width_px, height_px


def _estimate_request_bytes_from_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, scale_m: float,
    bytes_per_pixel: int = DEFAULT_ESTIMATE_BYTES_PER_PIXEL, band_count: int = 1,
) -> int:
    """Saf Python: bbox + scale + varsayilan byte-genisligi -> tahmini byte sayisi."""
    width_px, height_px = _estimate_pixel_grid(xmin, ymin, xmax, ymax, scale_m)
    return width_px * height_px * bytes_per_pixel * band_count


def _estimate_request_bytes(
    region, scale_m: float,
    bytes_per_pixel: int = DEFAULT_ESTIMATE_BYTES_PER_PIXEL, band_count: int = 1,
) -> int:
    """
    GEE'ye DOKUNAN sarmalayici: `region.bounds().getInfo()` ile AOI bbox'ini
    ceker (ucuz bir metadata cagrisi -- piksel indirmez), sonra saf tahmine
    devreder. `_export_tiled` zaten AYNI `_bbox_from_region` cagrisini
    yapiyor; burada ONCEDEN yapilmasi yalnizca "once basarisiz olacagini
    bilerek direct dene" agini onlemek icindir.
    """
    xmin, ymin, xmax, ymax = _bbox_from_region(region)
    return _estimate_request_bytes_from_bbox(xmin, ymin, xmax, ymax, scale_m, bytes_per_pixel, band_count)


def _file_ok(path: Path) -> bool:
    """Dosya gercekten var mi VE 0 byte'tan buyuk mu (bos/yarim dosya da basarisizlik sayilir)."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _atomic_replace(tmp_path: Path, out_path: Path) -> None:
    """
    `tmp_path` -> `out_path` icin ATOMIK tasima (ayni dosya sistemi/dizin
    altinda `os.replace` POSIX'te atomiktir). Yarim/bozuk bir dosyanin ASLA
    `out_path`'te GORUNMEMESINI garanti eder -- ya tam tasima basarili olur
    ya da `out_path` hic degismez (eski dosya varsa OYNANMADAN kalir; hata
    firlatilir).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(tmp_path), str(out_path))


def _validate_export_alignment(
    out_path: Path, region, scale_m: float, crs: str,
    expected_band_count: int = 1, bounds_tolerance_px: float = 2.0,
) -> dict:
    """
    Export SONRASI (direct veya tiled -- HER IKI YOLDAN SONRA da) hizalama
    QA'si. Yalniz "dosya var mi" degil, USTELIK:
        - CRS beklenenle uyusuyor mu,
        - piksel boyutu (transform) beklenen scale ile (derece-tahmini
          toleransla) uyusuyor mu,
        - sinirlar istenen AOI'yi kapsiyor mu (kucuk bir piksel toleransiyla
          -- GEE/rasterio grid-snap farklari icin),
        - bant sayisi ve dtype beklenen gibi mi (null/tamamen bos DEGIL).
    Herhangi biri basarisiz olursa PredictorRunnerError firlatir -- yanlis
    hizalanmis bir dosya SESSIZCE basarili sayilamaz. Basarili olursa QA
    raporunu (dict) dondurur (metadata/log icin).
    """
    import numpy as np
    import rasterio

    if not _file_ok(out_path):
        raise PredictorRunnerError(f"Alignment QA: dosya yok/bos: {out_path}")

    xmin, ymin, xmax, ymax = _bbox_from_region(region)
    expected_deg_per_px = scale_m / _ESTIMATE_METERS_PER_DEGREE

    with rasterio.open(out_path) as src:
        actual_crs = src.crs
        transform = src.transform
        px_w, px_h = abs(transform.a), abs(transform.e)
        bounds = src.bounds
        band_count = src.count
        dtype = src.dtypes[0] if src.count else None
        sample = src.read(1, masked=True) if src.count else None

    # --- CRS ---
    if actual_crs is None:
        raise PredictorRunnerError(f"Alignment QA: cikti CRS'siz: {out_path}")
    try:
        crs_ok = actual_crs.to_string().upper() == str(crs).upper() or (
            actual_crs.to_epsg() is not None and str(actual_crs.to_epsg()) in str(crs)
        )
    except Exception:  # noqa: BLE001
        crs_ok = str(actual_crs).upper() == str(crs).upper()
    if not crs_ok:
        raise PredictorRunnerError(
            f"Alignment QA: beklenmeyen CRS ({actual_crs} != {crs}) -> {out_path}"
        )

    # --- Piksel boyutu (cozunurluk) -- %25 gorece tolerans (GEE'nin kendi
    # grid/snap mantigi bu ON-tahminden hafifce sapabilir; asil kontrol
    # "makul ayni buyuklukte mi", piksel-mukemmel esitlik DEGIL) ---
    rel_tol = 0.25
    if not (
        math.isclose(px_w, expected_deg_per_px, rel_tol=rel_tol)
        and math.isclose(px_h, expected_deg_per_px, rel_tol=rel_tol)
    ):
        raise PredictorRunnerError(
            f"Alignment QA: piksel boyutu beklenenden sapiyor "
            f"(actual=({px_w:.8f},{px_h:.8f}) deg, expected~={expected_deg_per_px:.8f} deg) "
            f"-> {out_path}"
        )

    # --- Sinirlar istenen AOI'yi kapsiyor mu (kucuk piksel toleransiyla) ---
    tol = bounds_tolerance_px * expected_deg_per_px
    covers = (
        bounds.left <= xmin + tol and bounds.bottom <= ymin + tol
        and bounds.right >= xmax - tol and bounds.top >= ymax - tol
    )
    if not covers:
        raise PredictorRunnerError(
            f"Alignment QA: cikti sinirlari istenen AOI'yi KAPSAMIYOR "
            f"(bounds={tuple(bounds)}, requested_aoi=({xmin},{ymin},{xmax},{ymax})) "
            f"-> {out_path}"
        )

    # --- Bant sayisi ---
    if band_count != expected_band_count:
        raise PredictorRunnerError(
            f"Alignment QA: beklenmeyen bant sayisi ({band_count} != "
            f"{expected_band_count}) -> {out_path}"
        )

    # --- Tamamen bos/NaN cikti degil ---
    if sample is not None:
        valid = int(np.asarray(~np.ma.getmaskarray(sample)).sum()) if hasattr(sample, "mask") else sample.size
        if valid == 0:
            raise PredictorRunnerError(f"Alignment QA: cikti tamamen bos/nodata: {out_path}")

    return {
        "crs": str(actual_crs), "dtype": dtype, "band_count": band_count,
        "pixel_size_deg": [px_w, px_h], "bounds": list(bounds),
        "requested_aoi": [xmin, ymin, xmax, ymax],
    }


def _assert_tile_transforms_compatible(srcs: list, label: str) -> None:
    """
    Merge'den ONCE: tum tile'larin CRS + piksel boyutu birbiriyle uyumlu
    (ayni ortak grid uzerinde) oldugunu dogrular. Uyumsuzsa TiledExportError
    yerine dogrudan PredictorRunnerError firlatir (bu bir tile-indirme
    basarisizligi degil, bir GRID TUTARSIZLIGI'dir -- farkli bir hata
    sinifi, ayri tanilanmali).
    """
    if not srcs:
        return
    ref = srcs[0]
    ref_crs, ref_px = ref.crs, (abs(ref.transform.a), abs(ref.transform.e))
    for i, s in enumerate(srcs[1:], start=1):
        if s.crs != ref_crs:
            raise PredictorRunnerError(
                f"[{label}] tile {i} CRS uyumsuz (ilk tile={ref_crs}, bu tile={s.crs}); "
                "tile'lar ORTAK bir grid uzerinde OLMALI."
            )
        px = (abs(s.transform.a), abs(s.transform.e))
        if not (math.isclose(px[0], ref_px[0], rel_tol=1e-6) and math.isclose(px[1], ref_px[1], rel_tol=1e-6)):
            raise PredictorRunnerError(
                f"[{label}] tile {i} piksel boyutu uyumsuz (ilk tile={ref_px}, bu tile={px}); "
                "tile'lar AYNI cozunurlukte OLMALI."
            )


def _export_tiled(
    image,
    out_path: Path,
    region,
    scale: int,
    crs: str,
    label: str,
    force: bool,
    tile_rows: int,
    tile_cols: int,
    tiles_dir: Path,
) -> Path:
    """
    AOI'yi tile_rows x tile_cols dikdörtgen tile'a böler, her tile'ı ayrı ayrı
    ee.Geometry.BBox ile (FULL AOI region'u DEĞİL, yalnızca o tile'ın kendi
    geometrisi) geemap.ee_export_image ile indirir, sonra rasterio.merge.merge
    ile out_path'e birleştirir. Çözünürlük (scale) DEĞİŞMEZ -- yalnızca AOI
    parçalanır.

    tiles_dir: outputs/experiments/<experiment_id>/data/_tiles/<label>/ --
        çağıran taraf (export_image_direct_or_tiled) hesaplar ve verir.
    """
    import ee
    import geemap
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    tiles_dir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = _bbox_from_region(region)
    tiles = _tile_bboxes(xmin, ymin, xmax, ymax, tile_rows, tile_cols)

    log.info(
        "Tiled export grid: %dx%d (label=%s, %d tile toplam), AOI bbox=[%.6f, %.6f, %.6f, %.6f]",
        tile_rows, tile_cols, label, len(tiles), xmin, ymin, xmax, ymax,
    )

    tile_paths = []
    for tile in tiles:
        r, c = tile["r"], tile["c"]
        tile_bounds = tile["bbox"]  # (xmin, ymin, xmax, ymax)
        tile_path = tiles_dir / f"{out_path.stem}_tile_r{r}_c{c}.tif"
        tile_paths.append(tile_path)

        # DEBUG SAFETY LOG: tile satır/sütun + koordinatlar + hedef dosya --
        # export'tan ÖNCE, her zaman.
        log.info(
            "Exporting tile r%d c%d bounds: [%.6f, %.6f, %.6f, %.6f] -> %s",
            r, c, tile_bounds[0], tile_bounds[1], tile_bounds[2], tile_bounds[3], tile_path,
        )

        if _file_ok(tile_path) and not force:
            log.info("[%s] tile (r%d,c%d) zaten var, atlanıyor: %s", label, r, c, tile_path)
            continue

        tile_geom = ee.Geometry.BBox(*tile_bounds)
        tile_image = image.clip(tile_geom)

        tile_error = None
        try:
            geemap.ee_export_image(
                tile_image, filename=str(tile_path), scale=scale, region=tile_geom,
                crs=crs, file_per_band=False,
            )
        except Exception as exc:  # noqa: BLE001
            tile_error = exc

        if not _file_ok(tile_path):
            raise TiledExportError(
                label, r, c, tile_bounds, tile_path,
                detail=f"geemap hatası: {tile_error}" if tile_error else "geemap sessizce dosya üretmedi.",
            )
        log.info(
            "[%s] tile (r%d,c%d) yazıldı: %s (%d bytes)",
            label, r, c, tile_path, tile_path.stat().st_size,
        )

    log.info(
        "Merging %d tiles -> %s (label=%s)", len(tile_paths), out_path, label,
    )
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        # ALIGNMENT SAFETY: merge'den ONCE, tum tile'larin ORTAK bir grid
        # uzerinde (ayni CRS + ayni piksel boyutu) oldugunu dogrula. Bu,
        # bagimsiz indirilen tile'larin sessizce yanlis hizali birlesmesini
        # (kaydirilmis/duplike dikis pikselleri) ONLER.
        _assert_tile_transforms_compatible(srcs, label)

        merged_array, merged_transform = rasterio_merge(srcs)
        src0 = srcs[0]
        out_profile = src0.profile.copy()
        out_profile.update(
            {
                "height": merged_array.shape[1],
                "width": merged_array.shape[2],
                "transform": merged_transform,
                "compress": "LZW",
                "BIGTIFF": "IF_SAFER",
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # ATOMIK YAZMA: dogrudan out_path'e DEGIL, once bir gecici dosyaya
        # yazilir; ancak TAM ve DOGRULANMIS bir dosya elde edildikten sonra
        # os.replace ile out_path'e ATOMIK olarak tasinir. Boylece merge
        # sirasinda bir kesinti (disk dolmasi, process kill, vb.) YARIM/BOZUK
        # bir dosyayi out_path'te GORUNUR HALE GETIRMEZ -- bir sonraki
        # calistirmada _file_ok(out_path) YANLISLIKLA "basarili" sanmaz.
        tmp_path = out_path.parent / f".{out_path.name}.tmp"
        try:
            with rasterio.open(tmp_path, "w", **out_profile) as dst:
                dst.write(merged_array)
            if not _file_ok(tmp_path):
                raise PredictorRunnerError(
                    f"[{label}] gecici birlestirme dosyasi olusmadi/bos: {tmp_path}"
                )
            _atomic_replace(tmp_path, out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    finally:
        for s in srcs:
            s.close()

    if not _file_ok(out_path):
        raise PredictorRunnerError(f"[{label}] tile birleştirme sonrası dosya oluşmadı/boş: {out_path}")

    log.info("[%s] birleştirilmiş dosya yazıldı: %s (%d tile'dan)", label, out_path, len(tile_paths))
    return out_path


def export_image_direct_or_tiled(
    image,
    out_path: Path,
    region,
    scale: int,
    crs: str,
    label: str,
    force: bool,
    tiles_dir: Path,
    tile_rows: int = 2,
    tile_cols: int = 2,
    cleanup_tiles: bool = False,
    bytes_per_pixel: int = DEFAULT_ESTIMATE_BYTES_PER_PIXEL,
    band_count: int = 2,
    run_alignment_qa: bool = True,
) -> dict:
    """
    Önce, bir ON-FILTRE olarak, tam AOI'nin tahmini indirme boyutunu
    (`_estimate_request_bytes`, GEE'ye dokunmadan) `DIRECT_EXPORT_SAFE_THRESHOLD_BYTES`
    ile karşılaştırır. Tahmin eşiği AŞARSA, doğrudan tiled export'a geçilir
    (başarısız olacağı bilinen bir "direct getPixels" ağ çağrısı hiç
    yapılmaz). AŞMAZSA (veya tahmin herhangi bir nedenle BAŞARISIZ olursa --
    bu durumda eski davranışa dönülür), tam AOI'yi tek seferde
    (geemap.ee_export_image, direct getPixels) export etmeyi dener.

    KRİTİK DÜZELTME (ön-filtreden BAĞIMSIZ, HER ZAMAN aktif): geemap.ee_export_image
    bazen bir istisna FIRLATMADAN (yalnızca konsola "An error occurred while
    downloading" gibi bir mesaj basarak) sessizce başarısız olabilir. Bu
    yüzden direct export'un başarılı sayılması için SADECE "istisna
    fırlatılmadı" yeterli DEĞİLDİR; çıktı dosyasının GERÇEKTEN var olduğu VE
    0 byte'tan büyük olduğu da ayrıca doğrulanır (_file_ok). Direct export şu
    üç durumdan HERHANGİ BİRİNDE "başarısız" sayılır ve tiled fallback'e
    geçilir (hata mesajı içeriğine bakılmaksızın -- yalnızca dosyanın
    var/boyut>0 olup olmadığı yeterli sinyaldir):
        1) geemap.ee_export_image bir istisna fırlatırsa,
        2) istisna fırlatmadan döner ama out_path yoksa,
        3) out_path var ama 0 byte'sa.

    ATOMİKLİK: hem direct hem tiled yol, NİHAİ dosyayı önce bir `.tmp`
    dosyasına yazar, yalnızca TAM ve doğrulanmış bir dosya elde edildikten
    sonra `os.replace` ile `out_path`'e ATOMİK olarak taşır. Bir kesinti
    (network/disk hatası, process kill) YARIM bir dosyayı `out_path`'te asla
    GÖRÜNÜR hale getirmez.

    HİZALAMA QA: (run_alignment_qa=True, varsayılan) taze üretilen HER
    çıktı (direct veya tiled) için CRS/piksel-boyutu/sınırlar/bant-sayısı
    doğrulanır (_validate_export_alignment). `skipped_existing` durumunda
    (dosya zaten vardı) ASLA çalışmaz -- eski/frozen dosyalar retroaktif
    olarak reddedilmez.

    Tiled fallback'te AOI kademeli olarak (2x2 -> 4x4 -> 6x6 -> 8x8) daha
    ince tile'lara bölünür; bir grid'de bir tile bile başarısız
    olursa (TiledExportError) veya boyut-limiti hatası alınırsa bir sonraki
    (daha ince) grid'e geçilir. Boyutla/dosya-eksikliğiyle İLGİSİZ bir hata
    (auth, geometry, vb.) alınırsa döngü durur ve hata olduğu gibi
    fırlatılır.

    Döner: {"path": Path,
            "transport": "direct" | "tiled_direct_fallback" | "tiled_preflight_skip" | "skipped_existing",
            "tile_grid": (rows, cols) | None, "tile_count": int | None,
            "estimated_bytes": int | None, "direct_skipped_preflight": bool,
            "alignment_qa": dict | None}
    """
    import geemap

    if _file_ok(out_path) and not force:
        log.info("[%s] zaten var, atlanıyor: %s", label, out_path)
        return {
            "path": out_path, "transport": "skipped_existing", "tile_grid": None,
            "tile_count": None, "estimated_bytes": None,
            "direct_skipped_preflight": False, "alignment_qa": None,
        }

    # --- ON-FILTRE: tahmini boyut esigi asiyor mu? (GEE'ye dokunur ama
    # yalnizca ucuz bir bounds().getInfo() -- piksel INDIRMEZ). Tahmin
    # herhangi bir nedenle basarisiz olursa (auth yok, region GEE nesnesi
    # degil, vb.) eski davranisa (direct once denenir) SESSIZCE geri
    # DONULUR -- bu ON-filtre asla var olan davranisi KIRAMAZ, yalnizca
    # IYILESTIREBILIR. ---
    estimated_bytes = None
    skip_direct_preflight = False
    try:
        estimated_bytes = _estimate_request_bytes(region, scale, bytes_per_pixel, band_count)
        skip_direct_preflight = estimated_bytes > DIRECT_EXPORT_SAFE_THRESHOLD_BYTES
        log.info(
            "[%s] On-filtre boyut tahmini: ~%d bytes (esik=%d bytes) -> %s",
            label, estimated_bytes, DIRECT_EXPORT_SAFE_THRESHOLD_BYTES,
            "tiled'e dogrudan gecilecek" if skip_direct_preflight else "direct denenecek",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[%s] On-filtre boyut tahmini basarisiz oldu (%s); eski davranisa "
            "(once direct denensin) geri donuluyor.", label, exc,
        )

    direct_ok = False
    direct_error = None
    if skip_direct_preflight:
        log.info(
            "[%s] Direct export ATLANDI (on-filtre: tahmini %d bytes > esik %d bytes); "
            "dogrudan tiled export'a geciliyor.", label, estimated_bytes,
            DIRECT_EXPORT_SAFE_THRESHOLD_BYTES,
        )
    else:
        log.info("Direct export attempt: [%s] tam AOI, tiled olmayan export -> %s", label, out_path)
        tmp_direct_path = out_path.parent / f".{out_path.name}.direct.tmp"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            geemap.ee_export_image(
                image, filename=str(tmp_direct_path), scale=scale, region=region,
                crs=crs, file_per_band=False,
            )
        except Exception as exc:  # noqa: BLE001
            direct_error = exc

        if _file_ok(tmp_direct_path):
            # ATOMIK tasima: yarim bir direct indirme ASLA out_path'te
            # gorunmez -- ya tam dosya ATOMIK olarak yerine gecer ya da hic
            # degismez.
            _atomic_replace(tmp_direct_path, out_path)
            direct_ok = _file_ok(out_path)
        if tmp_direct_path.exists():
            tmp_direct_path.unlink(missing_ok=True)

    if direct_ok:
        log.info("[%s] direkt export başarılı: %s (%d bytes)", label, out_path, out_path.stat().st_size)
        alignment_qa = (
            _validate_export_alignment(out_path, region, scale, crs, expected_band_count=band_count)
            if run_alignment_qa else None
        )
        return {
            "path": out_path, "transport": "direct", "tile_grid": None, "tile_count": None,
            "estimated_bytes": estimated_bytes, "direct_skipped_preflight": False,
            "alignment_qa": alignment_qa,
        }

    if not skip_direct_preflight:
        log.warning(
            "Direct export produced no file; switching to tiled fallback. "
            "[%s] direct export başarısız/dosya üretilmedi. Hata=%s",
            label, direct_error,
        )

    last_exc = None
    for rows, cols in _TILE_GRID_ESCALATION:
        try:
            log.info("Tiled export grid: %dx%d (label=%s)", rows, cols, label)
            result_path = _export_tiled(
                image, out_path, region, scale, crs, label, force,
                tile_rows=rows, tile_cols=cols, tiles_dir=tiles_dir,
            )
            log.info(
                "[%s] Tiled fallback BAŞARILI: grid=%dx%d, tile_count=%d, "
                "final_path=%s", label, rows, cols, rows * cols, result_path,
            )
            # Basari sonrasi gecici tile dosyalarini temizle -- basarisizlikta
            # (asagida raise edilirse) tiles_dir HIC dokunulmadan tanı için
            # kalir.
            if cleanup_tiles:
                _tiles = sorted(tiles_dir.glob(f"{out_path.stem}_tile_*.tif"))
                for t in _tiles:
                    t.unlink()
                log.info("[%s] --cleanup-tiles: %d tile dosyası silindi.", label, len(_tiles))
            alignment_qa = (
                _validate_export_alignment(result_path, region, scale, crs, expected_band_count=band_count)
                if run_alignment_qa else None
            )
            return {
                "path": result_path,
                "transport": "tiled_preflight_skip" if skip_direct_preflight else "tiled_direct_fallback",
                "tile_grid": (rows, cols), "tile_count": rows * cols,
                "estimated_bytes": estimated_bytes, "direct_skipped_preflight": skip_direct_preflight,
                "alignment_qa": alignment_qa,
            }
        except TiledExportError as exc:
            last_exc = exc
            log.warning(
                "[%s] %dx%d grid'de tile (r%d,c%d) başarısız oldu, bir sonraki "
                "(daha ince) grid deneniyor: %s", label, rows, cols, exc.r, exc.c, exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_size_related_error(exc):
                raise
            log.warning(
                "[%s] %dx%d tile grid de boyut limitine takıldı, bir sonraki "
                "grid deneniyor: %s", label, rows, cols, exc,
            )

    raise PredictorRunnerError(
        f"[{label}] Direct export VE tüm tile grid denemeleri ({_TILE_GRID_ESCALATION}) "
        f"başarısız oldu. Son hata: {last_exc}"
    )


# =============================================================================
# Kozan-dışı: doğrudan GEE -> yerel disk export (Drive'a uğramadan)
# =============================================================================
def _export_predictors_direct(ctx: dict, force: bool, cleanup_tiles: bool = False) -> dict:
    """
    Step3'ün zaten parametrik GEE fonksiyonlarını kullanarak current+baseline
    Landsat LST ve NDVI'yı DOĞRUDAN (Drive export/polling/download zincirine
    girmeden) yerel, namespaced diske export eder -- Step6/Step6A'da zaten
    kurulan aynı desen (geemap.ee_export_image), gerekirse otomatik tiled
    fallback ile (bkz. export_image_direct_or_tiled).
    """
    try:
        import ee  # noqa: F401
        import geemap  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise PredictorRunnerError(
            f"ee/geemap import edilemedi: {type(exc).__name__}: {exc}"
        ) from exc

    from core.config import EXPORT_CRS, GEE_PROJECT
    from core.gee_utils import init_gee
    import src.step3_landsat_lst as step3

    try:
        init_gee(GEE_PROJECT)
    except Exception as exc:  # noqa: BLE001
        raise PredictorRunnerError(
            f"GEE init/auth başarısız: {type(exc).__name__}: {exc}. "
            "'earthengine authenticate' çalıştırın."
        ) from exc

    region = get_region(ctx)
    region_name = ctx["region_key"]
    scale = 30

    for d in ("data_root", "baseline_input_dir", "current_period_dir",
              "ndvi_baseline_dir", "ndvi_current_dir"):
        Path(ctx[d]).mkdir(parents=True, exist_ok=True)

    written = {}
    export_transport_log = {}

    def _export(image, out_path: Path, label: str) -> None:
        tiles_dir = ctx["data_root"] / "_tiles" / label
        result = export_image_direct_or_tiled(
            image, out_path, region, scale, EXPORT_CRS, label, force,
            tiles_dir=tiles_dir, cleanup_tiles=cleanup_tiles,
        )
        written[label] = str(result["path"])
        export_transport_log[label] = {
            "export_transport": result["transport"],
            "tile_grid": list(result["tile_grid"]) if result["tile_grid"] else None,
            "tile_count": result["tile_count"],
            "scale": scale,
            "crs": EXPORT_CRS,
            "path": str(result["path"]),
        }

    # --- Current period LST (Celsius + valid count) ---
    current_lst_path = (
        ctx["current_period_dir"]
        / f"landsat_current_period_{ctx['current_period_days']}days.tif"
    )
    current_lst_image, _ = step3.get_current_period_median(
        region, region_name, ctx["current_period_end_date"], ctx["current_period_days"],
    )
    _export(current_lst_image, current_lst_path, "current_lst")

    # --- Current period NDVI (NDVI + valid count) ---
    current_ndvi_path = ctx["ndvi_current_dir"] / "current_ndvi_median.tif"
    current_ndvi_image, _ = step3.get_current_period_ndvi_median(
        region, region_name, ctx["current_period_end_date"], ctx["current_period_days"],
    )
    _export(current_ndvi_image, current_ndvi_path, "current_ndvi")

    # --- Baseline LST, yıl başına tek bant (ST_B10 DN; Step5 dn_to_celsius ile çevirir) ---
    baseline_lst_collection, baseline_lst_meta = step3.get_landsat_baseline_window_median_collection(
        region, region_name, ctx["current_period_end_date"], ctx["current_period_days"],
        baseline_start=ctx["baseline_start_date"], baseline_end=ctx["baseline_end_date"],
    )
    for record in baseline_lst_meta["windows"]:
        year = record["year"]
        end_text = record["window_end"]
        out_path = ctx["baseline_input_dir"] / f"{ctx['landsat_file_prefix']}_baseline_{end_text}.tif"
        year_image = (
            ee.Image(baseline_lst_collection.filter(ee.Filter.eq("baseline_year", year)).first())
            .select("ST_B10")
        )
        _export(year_image, out_path, f"baseline_lst_{year}")

    # --- Baseline NDVI, yıl başına tek bant ---
    baseline_ndvi_collection, baseline_ndvi_meta = step3.get_landsat_baseline_window_ndvi_collection(
        region, region_name, ctx["current_period_end_date"], ctx["current_period_days"],
        baseline_start=ctx["baseline_start_date"], baseline_end=ctx["baseline_end_date"],
    )
    for record in baseline_ndvi_meta["windows"]:
        year = record["year"]
        end_text = record["window_end"]
        out_path = ctx["ndvi_baseline_dir"] / f"ndvi_baseline_{end_text}.tif"
        year_image = (
            ee.Image(baseline_ndvi_collection.filter(ee.Filter.eq("baseline_year", year)).first())
            .select("NDVI")
        )
        _export(year_image, out_path, f"baseline_ndvi_{year}")

    _write_predictor_export_metadata(ctx, export_transport_log)

    return written


def _write_predictor_export_metadata(ctx: dict, export_transport_log: dict) -> Path:
    """
    outputs/experiments/<experiment_id>/predictor_export_metadata.json yazar:
    her export edilen ürün için transport (direct/tiled_direct_fallback),
    tile grid, scale, crs; ayrıca deney bağlamı (region, pencereler, baseline
    yılları).
    """
    metadata = {
        "experiment_id": ctx["experiment_id"],
        "region_key": ctx["region_key"],
        "predictor_start_date": ctx["predictor_start_date"],
        "predictor_end_date": ctx["predictor_end_date"],
        "current_period_end_date": ctx["current_period_end_date"],
        "current_period_days": ctx["current_period_days"],
        "baseline_years": ctx["baseline_years"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exports": export_transport_log,
    }
    out_path = ctx["output_root"] / "predictor_export_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Predictor export metadata yazıldı: %s", out_path)
    return out_path


def _run_kozan_export() -> None:
    """Legacy Kozan Step3->Step4->Step4b->Step5->Step5C zincirini çalıştırır."""
    import src.step3_landsat_lst as step3
    import src.step4_export_geotiff as step4
    import src.step4b_download_drive_export as step4b
    import src.step5_preprocess_timeseries as step5
    import src.step5c_tvdi as step5c

    log.info("[kozan_2023] STEP 3 (Landsat LST GEE hazırlığı)")
    step3_result = step3.main()
    log.info("[kozan_2023] STEP 4 (GEE -> Drive export)")
    step4.main(step3_result=step3_result)
    log.info("[kozan_2023] STEP 4B (Drive -> local indirme + doğrulama)")
    step4b.main()
    log.info("[kozan_2023] STEP 5 (LST anomaly)")
    step5.main()
    log.info("[kozan_2023] STEP 5C (TVDI)")
    step5c.main()


def _run_local_only(ctx: dict) -> dict:
    import src.step5_preprocess_timeseries as step5
    import src.step5c_tvdi as step5c

    if ctx["is_kozan"]:
        log.info("[kozan_2023] STEP 5 (legacy, local-only)")
        step5_result = step5.run_step5(None)
        log.info("[kozan_2023] STEP 5C (legacy, local-only)")
        step5c_result = step5c.run_step5c(None)
    else:
        log.info("[%s] STEP 5 (namespaced, local-only)", ctx["experiment_id"])
        step5_result = step5.run_step5(ctx)
        log.info("[%s] STEP 5C (namespaced, local-only)", ctx["experiment_id"])
        step5c_result = step5c.run_step5c(ctx)

    return {"step5": step5_result, "step5c": step5c_result}


def main(
    experiment_id: str = "kozan_2023",
    dry_run: bool = False,
    export: bool = False,
    local_only: bool = False,
    force: bool = False,
    cleanup_tiles: bool = False,
) -> dict:
    ctx = build_experiment_context(experiment_id)
    log_context_summary(ctx, log)

    if not ctx["is_kozan"]:
        _assert_paths_are_safely_namespaced(ctx)

    if dry_run:
        log.info("[dry-run] Planlanan yollar:")
        _log_planned_paths(ctx)
        log.info("[dry-run] Hiçbir export/işleme ÇALIŞTIRILMADI.")
        return {"experiment_id": experiment_id, "ran": False, "reason": "dry_run"}

    if export and local_only:
        raise PredictorRunnerError("--export ve --local-only birlikte verilemez (çelişkili).")
    if not export and not local_only:
        raise PredictorRunnerError(
            "Ne --export ne --local-only verildi; hangi modda çalışılacağı belirsiz. "
            "Yalnızca önizleme için --dry-run kullanın."
        )

    if export:
        if ctx["is_kozan"]:
            _run_kozan_export()
            return {"experiment_id": experiment_id, "ran": True, "mode": "export_kozan_legacy"}
        else:
            written = _export_predictors_direct(ctx, force=force, cleanup_tiles=cleanup_tiles)
            log.info("Export tamamlandı: %s", written)
            local_result = _run_local_only(ctx)
            return {
                "experiment_id": experiment_id, "ran": True, "mode": "export_namespaced",
                "exported": written, **local_result,
            }

    # local_only
    result = _run_local_only(ctx)
    return {"experiment_id": experiment_id, "ran": True, "mode": "local_only", **result}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0D: deney-farkında (experiment-aware) Step3-Step5/5C "
        "predictor üretim çalıştırıcısı. Step7/Step8'i ÇALIŞTIRMAZ, model "
        "EĞİTMEZ. kozan_2023 legacy davranışını korur; diğer deneyler "
        "(örn. manavgat_2021) tamamen namespaced çalışır. Büyük AOI/ürünler "
        "için otomatik tiled export fallback içerir (çözünürlük değişmez)."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Hiçbir şey çalıştırma; deney özetini + planlanan tüm yolları bas.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="GEE'den current+baseline Landsat LST/NDVI export'unu çalıştırıp "
        "ardından Step5/Step5C'yi çalıştırır.",
    )
    parser.add_argument(
        "--local-only", action="store_true",
        help="GeoTIFF'lerin zaten var olduğunu varsayar; yalnızca Step5/Step5C'yi çalıştırır.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Çıktılar zaten varsa üzerine yaz (export dosyaları, tile'lar ve Step5/5C çıktıları).",
    )
    parser.add_argument(
        "--cleanup-tiles", action="store_true",
        help="Tiled fallback tetiklenirse, birleştirme sonrası ara tile dosyalarını sil "
        "(varsayılan: tile'lar debug için saklanır).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        experiment_id=args.experiment,
        dry_run=args.dry_run,
        export=args.export,
        local_only=args.local_only,
        force=args.force,
        cleanup_tiles=args.cleanup_tiles,
    )