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


def _file_ok(path: Path) -> bool:
    """Dosya gercekten var mi VE 0 byte'tan buyuk mu (bos/yarim dosya da basarisizlik sayilir)."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


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
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(merged_array)
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
) -> dict:
    """
    Önce tam AOI'yi tek seferde (geemap.ee_export_image, direct getPixels)
    export etmeyi dener.

    KRİTİK DÜZELTME: geemap.ee_export_image bazen bir istisna FIRLATMADAN
    (yalnızca konsola "An error occurred while downloading" gibi bir mesaj
    basarak) sessizce başarısız olabilir. Bu yüzden direct export'un
    başarılı sayılması için SADECE "istisna fırlatılmadı" yeterli DEĞİLDİR;
    çıktı dosyasının GERÇEKTEN var olduğu VE 0 byte'tan büyük olduğu da
    ayrıca doğrulanır (_file_ok). Direct export şu üç durumdan HERHANGİ
    BİRİNDE "başarısız" sayılır ve tiled fallback'e geçilir (hata mesajı
    içeriğine bakılmaksızın -- yalnızca dosyanın var/boyut>0 olup olmadığı
    yeterli sinyaldir):
        1) geemap.ee_export_image bir istisna fırlatırsa,
        2) istisna fırlatmadan döner ama out_path yoksa,
        3) out_path var ama 0 byte'sa.

    Tiled fallback'te AOI kademeli olarak (2x2 -> 4x4 -> 6x6 -> 8x8) daha
    ince tile'lara bölünür; bir grid'de bir tile bile başarısız
    olursa (TiledExportError) veya boyut-limiti hatası alınırsa bir sonraki
    (daha ince) grid'e geçilir. Boyutla/dosya-eksikliğiyle İLGİSİZ bir hata
    (auth, geometry, vb.) alınırsa döngü durur ve hata olduğu gibi
    fırlatılır.

    Döner: {"path": Path, "transport": "direct" | "tiled_direct_fallback",
            "tile_grid": (rows, cols) | None, "tile_count": int | None}
    """
    import geemap

    if _file_ok(out_path) and not force:
        log.info("[%s] zaten var, atlanıyor: %s", label, out_path)
        return {"path": out_path, "transport": "skipped_existing", "tile_grid": None, "tile_count": None}

    log.info("Direct export attempt: [%s] tam AOI, tiled olmayan export -> %s", label, out_path)
    direct_error = None
    try:
        geemap.ee_export_image(
            image, filename=str(out_path), scale=scale, region=region,
            crs=crs, file_per_band=False,
        )
    except Exception as exc:  # noqa: BLE001
        direct_error = exc

    direct_ok = _file_ok(out_path)
    if direct_ok:
        log.info("[%s] direkt export başarılı: %s (%d bytes)", label, out_path, out_path.stat().st_size)
        return {"path": out_path, "transport": "direct", "tile_grid": None, "tile_count": None}

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
            if cleanup_tiles:
                _tiles = sorted(tiles_dir.glob(f"{out_path.stem}_tile_*.tif"))
                for t in _tiles:
                    t.unlink()
                log.info("[%s] --cleanup-tiles: %d tile dosyası silindi.", label, len(_tiles))
            return {
                "path": result_path, "transport": "tiled_direct_fallback",
                "tile_grid": (rows, cols), "tile_count": rows * cols,
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
# Manavgat (Kozan-dışı): doğrudan GEE -> yerel disk export (Drive'a uğramadan)
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