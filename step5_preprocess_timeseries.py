"""
step5_preprocess_timeseries.py

Offline raster işleme katmanı.

Bu sürüm, tüm zaman serisini tek seferde belleğe almak yerine GeoTIFF
dosyalarını mekansal pencereler halinde işler. Yaklaşık tepe bellek kullanımı:

    time_count * window_size * window_size * float32

tam raster yığını yerine yalnızca pencere yığını ve birkaç çıktı bloğu
"""

from core.config import *
from core.io_utils import setup_logger

import json
import math
import re
import warnings
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


BASE_DIR = Path(__file__).resolve().parent
BASELINE_INPUT_DIR = BASE_DIR / "data" / "landsat_timeseries"
QA_DIR = BASE_DIR / "data" / "landsat_qa"
CURRENT_PERIOD_DIR = BASE_DIR / "data" / "current_period"
OUTPUT_DIR = BASE_DIR / "outputs" / "step5"

BASELINE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_PERIOD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log, log_file = setup_logger("step5")


def extract_date_from_filename(path: Path) -> np.datetime64:
    """
    Dosya adından tarih bilgisini çıkarır.

    Desteklenen örnekler:
        landsat_lst_dogu_akdeniz_2019-06-01_001.tif -> 2019-06-01
        landsat_lst_dogu_akdeniz_20190601_001.tif   -> 2019-06-01
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", path.name)

    if not match:
        raise ValueError(f"Dosya adında tarih bulunamadı: {path.name}")

    date_text = match.group(1)

    if "-" not in date_text:
        date_text = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"

    return np.datetime64(date_text)


def dn_to_celsius(dn_array: np.ndarray) -> np.ndarray:
    """
    Landsat Collection 2 Level 2 ST_B10 DN değerlerini Celsius'a çevirir.

    Formül:
        Kelvin = DN * LANDSAT_SCALE + LANDSAT_OFFSET
        Celsius = Kelvin - 273.15
    """
    kelvin = dn_array * LANDSAT_SCALE + LANDSAT_OFFSET
    return kelvin - 273.15


def build_cloud_mask_from_qa(qa_array: np.ndarray) -> np.ndarray:
    """
    Landsat QA_PIXEL bandından temiz piksel maskesi üretir.

    True  = temiz piksel
    False = bulut, gölge, kar, dolgu veya sirüs nedeniyle maskelenecek piksel
    """
    fill = 1 << 0
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    snow = 1 << 5

    bad_pixels = fill | dilated_cloud | cirrus | cloud | cloud_shadow | snow
    return (qa_array.astype(np.uint16) & bad_pixels) == 0


def list_baseline_tifs() -> list[Path]:
    """
    Baseline zaman serisini oluşturan GeoTIFF dosyalarını listeler.

    QA dosyaları aynı klasöre yanlışlıkla konmuşsa `_qa` ile bitenleri
    baseline görüntüsü olarak kullanmaz.
    """
    tif_files = sorted(
        path
        for path in BASELINE_INPUT_DIR.glob("*.tif")
        if not path.stem.lower().endswith("_qa")
    )

    if not tif_files:
        raise FileNotFoundError(
            f"Baseline GeoTIFF dosyası bulunamadı: {BASELINE_INPUT_DIR}\n"
            "Step4 baseline zaman serisi export dosyalarını bu klasöre koymalısın."
        )

    return tif_files


def list_current_period_tifs() -> list[Path]:
    """
    Current period median GeoTIFF dosyalarını listeler.

    Birden fazla dosya varsa deterministik olması için sıralı listedeki ilk
    dosya kullanılır ve log'a uyarı yazılır.
    """
    current_files = sorted(CURRENT_PERIOD_DIR.glob("*.tif"))

    if not current_files:
        raise FileNotFoundError(
            f"Current period median dosyası bulunamadı: {CURRENT_PERIOD_DIR}\n"
            "Step4 landsat_current_period_XXdays.tif export dosyasını buraya koymalısın."
        )

    if len(current_files) > 1:
        log.warning(
            "Birden fazla current period dosyası bulundu; ilki kullanılıyor: %s",
            current_files[0].name,
        )

    return current_files


def read_window(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """
    Tek bant rasterdan belirtilen pencereyi float32 olarak okur.

    Rasterio masked array döndürdüğünde maskeli pikseller NaN'a çevrilir.
    """
    array = src.read(1, window=window, masked=True).astype("float32")
    return array.filled(np.nan)


def read_qa_window(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """
    QA rasterından belirtilen pencereyi uint16 olarak okur.

    Maskeli QA pikselleri `fill` biti set edilmiş gibi 1 değeriyle doldurulur;
    böylece güvenli tarafta kalıp temiz piksel sayılmazlar.
    """
    array = src.read(1, window=window, masked=True)
    return array.filled(1).astype("uint16")


def iter_windows(width: int, height: int, window_size: int):
    """
    Raster boyutunu sabit kenarlı pencerelere böler.

    Son satır ve sütundaki pencereler raster sınırına göre daha küçük olabilir.
    """
    if window_size <= 0:
        raise ValueError("STEP5_WINDOW_SIZE pozitif bir tam sayı olmalıdır")

    for row_off in range(0, height, window_size):
        block_height = min(window_size, height - row_off)

        for col_off in range(0, width, window_size):
            block_width = min(window_size, width - col_off)
            yield Window(col_off, row_off, block_width, block_height)


def count_windows(width: int, height: int, window_size: int) -> int:
    """Verilen raster boyutu ve pencere kenarına göre toplam pencere sayısını hesaplar."""
    if window_size <= 0:
        raise ValueError("STEP5_WINDOW_SIZE pozitif bir tam sayı olmalıdır")

    return math.ceil(width / window_size) * math.ceil(height / window_size)


def validate_same_grid(profile: dict, path: Path) -> None:
    """
    Girdi rasterının referans grid ile aynı boyut ve transform'a sahip olduğunu doğrular.

    Baseline, QA ve current period rasterları aynı piksel gridinde değilse pencere
    bazlı hesaplama yanlış pikselleri karşılaştırır; bu yüzden erken hata verir.
    """
    with rasterio.open(path) as src:
        if src.width != profile["width"] or src.height != profile["height"]:
            raise ValueError(
                f"{path.name} için grid boyutu uyuşmuyor: "
                f"{src.width}x{src.height} != "
                f"{profile['width']}x{profile['height']}"
            )

        if src.transform != profile["transform"]:
            raise ValueError(f"{path.name} için transform uyuşmuyor")


def output_profile(profile: dict) -> dict:
    """
    Çıktı GeoTIFF profili üretir.

    Tek bant float32, LZW sıkıştırmalı ve gerektiğinde BigTIFF destekli raster
    yazmak için kaynak profili günceller.
    """
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    return profile


def open_output(path: Path, profile: dict):
    """Çıktı GeoTIFF dosyasını yazma modunda açar."""
    return rasterio.open(path, "w", **output_profile(profile))


def time_offsets_days(times: list[np.datetime64]) -> np.ndarray:
    """
    Tarih listesini ilk tarihten itibaren gün cinsinden sayısal eksene çevirir.

    Bu eksen, düzensiz tarih aralıklarında lineer interpolasyonun gerçek gün
    farklarına göre yapılmasını sağlar.
    """
    start = times[0]
    return np.array(
        [(time - start) / np.timedelta64(1, "D") for time in times],
        dtype="float32",
    )


def estimated_stack_memory_mb(time_count: int, window_size: int) -> float:
    """
    Bir baseline pencere yığınının yaklaşık bellek kullanımını MB cinsinden hesaplar.

    Hesap yalnızca `time_count x window_size x window_size` float32 yığınını kapsar;
    küçük ara diziler ve çıktı blokları buna dahil değildir.
    """
    bytes_per_float32 = np.dtype("float32").itemsize
    return time_count * window_size * window_size * bytes_per_float32 / (1024**2)


def interpolate_stack_along_time(
    stack: np.ndarray,
    time_axis: np.ndarray,
) -> np.ndarray:
    """
    Her pikselin zaman serisindeki iç boşlukları lineer interpolasyonla doldurur.

    İşlem in-place yapılır; yani bellek kullanımını düşük tutmak için yeni bir
    stack kopyası oluşturulmaz. Başta ve sonda kalan NaN değerleri doldurulmaz.
    Bu davranış, xarray.interpolate_na'nın varsayılan ekstrapolasyon yapmayan
    davranışıyla uyumludur.
    """
    if stack.shape[0] < 3:
        return stack

    interpolated = stack
    if len(time_axis) != interpolated.shape[0]:
        raise ValueError("Zaman ekseni uzunluğu stack zaman boyutuyla uyuşmuyor")

    flat = interpolated.reshape(interpolated.shape[0], -1)
    valid_counts = np.sum(np.isfinite(flat), axis=0)

    for pixel_index in np.where(valid_counts > 1)[0]:
        series = flat[:, pixel_index]
        valid = np.isfinite(series)
        first_valid = int(np.argmax(valid))
        last_valid = len(valid) - int(np.argmax(valid[::-1])) - 1
        fill_mask = (
            (~valid)
            & (time_axis >= time_axis[first_valid])
            & (time_axis <= time_axis[last_valid])
        )

        if np.any(fill_mask):
            flat[fill_mask, pixel_index] = np.interp(
                time_axis[fill_mask],
                time_axis[valid],
                series[valid],
            )

    return interpolated


def read_baseline_stack_window(
    datasets: list[rasterio.io.DatasetReader],
    qa_datasets: list[rasterio.io.DatasetReader | None],
    tif_files: list[Path],
    window: Window,
) -> np.ndarray:
    """
    Baseline rasterlarının aynı penceresini zaman boyutunda yığın olarak okur.

    Her sahne için:
        1. DN değerleri okunur.
        2. Celsius'a çevrilir.
        3. Varsa QA maskesi uygulanır.
        4. Fiziksel sıcaklık aralığı dışındaki pikseller NaN yapılır.

    Dönüş şekli: (time, window_height, window_width)
    """
    block_shape = (len(datasets), int(window.height), int(window.width))
    stack = np.empty(block_shape, dtype="float32")

    for scene_index, (src, qa_src, tif_path) in enumerate(
        zip(datasets, qa_datasets, tif_files)
    ):
        dn_array = read_window(src, window)
        lst_celsius = dn_to_celsius(dn_array)

        if qa_src is not None:
            qa_array = read_qa_window(qa_src, window)
            clean_mask = build_cloud_mask_from_qa(qa_array)
            lst_celsius = np.where(clean_mask, lst_celsius, np.nan)

        lst_celsius = np.where(
            (lst_celsius > -30) & (lst_celsius < 80),
            lst_celsius,
            np.nan,
        ).astype("float32")

        stack[scene_index] = lst_celsius

    return stack


def nanmean_float32(stack: np.ndarray) -> np.ndarray:
    """
    NaN değerleri yok sayarak zaman ekseninde ortalama hesaplar.

    Tüm zaman adımları NaN olan piksellerde NumPy uyarısı bastırılır ve sonuç
    NaN olarak kalır.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0).astype("float32")


def nanstd_float32(stack: np.ndarray) -> np.ndarray:
    """
    NaN değerleri yok sayarak zaman ekseninde standart sapma hesaplar.

    Çıktı float32 tutulur; bu hem GeoTIFF çıktılarıyla hem de bellek hedefiyle
    uyumludur.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(stack, axis=0).astype("float32")


class RunningStats:
    """
    Pencere pencere üretilen rasterlar için global özet istatistik tutar.

    Tüm rasterı bellekte toplamadan geçerli piksellerin count, mean, std, min ve
    max değerlerini biriktirmek için kullanılır.
    """

    def __init__(self) -> None:
        """Boş istatistik biriktiricisini başlatır."""
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, array: np.ndarray) -> None:
        """Yeni bir pencere dizisindeki geçerli pikselleri istatistiklere ekler."""
        valid = np.isfinite(array)
        count = int(np.sum(valid))

        if count == 0:
            return

        values = array[valid].astype("float64")
        self.count += count
        self.sum += float(np.sum(values))
        self.sum_sq += float(np.sum(values * values))
        self.min = min(self.min, float(np.min(values)))
        self.max = max(self.max, float(np.max(values)))

    @property
    def mean(self) -> float:
        """Biriktirilen tüm geçerli piksellerin ortalamasını döndürür."""
        return float("nan") if self.count == 0 else self.sum / self.count

    @property
    def std(self) -> float:
        """Biriktirilen tüm geçerli piksellerin standart sapmasını döndürür."""
        if self.count == 0:
            return float("nan")

        variance = max((self.sum_sq / self.count) - (self.mean * self.mean), 0.0)
        return math.sqrt(variance)

    def as_dict(self) -> dict:
        """İstatistikleri JSON'a yazılabilir sözlük formatına çevirir."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": None if self.count == 0 else self.min,
            "max": None if self.count == 0 else self.max,
        }


def process_step5_windowed(
    tif_files: list[Path],
    current_path: Path,
) -> dict:
    """
    Step5 işlemini bellek dostu pencere tabanlı akışla çalıştırır.

    Ana işlem sırası:
        1. Baseline dosyalarını tarihe göre sıralar.
        2. Tüm rasterların aynı gridde olduğunu doğrular.
        3. Rasterı STEP5_WINDOW_SIZE boyutlu pencerelere böler.
        4. Her pencere için baseline zaman yığınını okur ve interpolate eder.
        5. Baseline mean/std, current median ve z-score anomali hesaplar.
        6. Sonuçları aynı anda çıktı GeoTIFF dosyalarına yazar.

    Böylece full raster zaman serisi belleğe alınmaz.
    """
    times = [extract_date_from_filename(path) for path in tif_files]
    sort_order = np.argsort(times)
    tif_files = [tif_files[index] for index in sort_order]
    times = [times[index] for index in sort_order]
    time_axis = time_offsets_days(times)

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile.copy()

    for tif_path in tif_files[1:]:
        validate_same_grid(profile, tif_path)

    validate_same_grid(profile, current_path)
    for tif_path in tif_files:
        qa_path = QA_DIR / tif_path.name.replace(".tif", "_qa.tif")
        if qa_path.exists():
            validate_same_grid(profile, qa_path)

    width = profile["width"]
    height = profile["height"]
    window_count = count_windows(width, height, STEP5_WINDOW_SIZE)
    memory_estimate_mb = estimated_stack_memory_mb(
        len(tif_files),
        STEP5_WINDOW_SIZE,
    )

    log.info("Baseline sahne sayısı: %s", len(tif_files))
    log.info("Raster boyutu: %sx%s", width, height)
    log.info("Pencere boyutu: %s px (%s pencere)", STEP5_WINDOW_SIZE, window_count)
    log.info("Pencere başına yaklaşık baseline stack belleği: %.1f MB", memory_estimate_mb)

    baseline_mean_stats = RunningStats()
    baseline_std_stats = RunningStats()
    current_stats = RunningStats()
    anomaly_stats = RunningStats()

    total_pixels = width * height
    valid_anomaly_pixels = 0

    output_paths = {
        "baseline_mean": OUTPUT_DIR / "baseline_lst_mean_celsius.tif",
        "baseline_std": OUTPUT_DIR / "baseline_lst_std_celsius.tif",
        "current_median": OUTPUT_DIR / "current_period_median_celsius.tif",
        "anomaly_zscore": OUTPUT_DIR / "anomaly_zscore.tif",
    }

    qa_paths = [QA_DIR / path.name.replace(".tif", "_qa.tif") for path in tif_files]

    with ExitStack() as stack_context:
        baseline_datasets = [
            stack_context.enter_context(rasterio.open(path)) for path in tif_files
        ]
        qa_datasets = [
            stack_context.enter_context(rasterio.open(path)) if path.exists() else None
            for path in qa_paths
        ]

        with (
            rasterio.open(current_path) as current_src,
            open_output(output_paths["baseline_mean"], profile) as mean_dst,
            open_output(output_paths["baseline_std"], profile) as std_dst,
            open_output(output_paths["current_median"], profile) as current_dst,
            open_output(output_paths["anomaly_zscore"], profile) as anomaly_dst,
        ):
            for index, window in enumerate(
                iter_windows(width, height, STEP5_WINDOW_SIZE),
                start=1,
            ):
                baseline_stack = read_baseline_stack_window(
                    baseline_datasets,
                    qa_datasets,
                    tif_files,
                    window,
                )
                baseline_stack = interpolate_stack_along_time(
                    baseline_stack,
                    time_axis,
                )

                baseline_mean = nanmean_float32(baseline_stack)
                baseline_std = nanstd_float32(baseline_stack)

                current_median = read_window(current_src, window)
                current_median = np.where(
                    (current_median > -30) & (current_median < 80),
                    current_median,
                    np.nan,
                ).astype("float32")

                with np.errstate(invalid="ignore", divide="ignore"):
                    anomaly_zscore = np.where(
                        baseline_std > STEP5_STD_EPSILON,
                        (current_median - baseline_mean) / baseline_std,
                        np.nan,
                    ).astype("float32")
                anomaly_zscore = np.where(
                    np.isfinite(anomaly_zscore),
                    anomaly_zscore,
                    np.nan,
                ).astype("float32")

                mean_dst.write(baseline_mean, 1, window=window)
                std_dst.write(baseline_std, 1, window=window)
                current_dst.write(current_median, 1, window=window)
                anomaly_dst.write(anomaly_zscore, 1, window=window)

                baseline_mean_stats.update(baseline_mean)
                baseline_std_stats.update(baseline_std)
                current_stats.update(current_median)
                anomaly_stats.update(anomaly_zscore)
                valid_anomaly_pixels += int(np.sum(np.isfinite(anomaly_zscore)))

                if index == 1 or index == window_count or index % 10 == 0:
                    log.info("Pencere işlendi: %s/%s", index, window_count)

    coverage_pct = 100 * valid_anomaly_pixels / total_pixels

    return {
        "profile": profile,
        "times": times,
        "tif_files": tif_files,
        "output_paths": output_paths,
        "baseline_mean_stats": baseline_mean_stats.as_dict(),
        "baseline_std_stats": baseline_std_stats.as_dict(),
        "current_stats": current_stats.as_dict(),
        "anomaly_stats": anomaly_stats.as_dict(),
        "coverage_percent": coverage_pct,
        "window_count": window_count,
        "estimated_stack_memory_mb": memory_estimate_mb,
    }


def write_metadata(
    result: dict,
    tif_files: list[Path],
    current_path: Path,
) -> Path:
    """
    Step5 çıktıları için metadata JSON dosyasını yazar.

    Metadata; girdi klasörlerini, pencere ayarlarını, üretilen dosya adlarını,
    özet istatistikleri ve kullanılan anomali yöntemini içerir.
    """
    times = result["times"]
    tif_files = result["tif_files"]
    output_paths = result["output_paths"]
    baseline_netcdf = None

    if STEP5_WRITE_INTERPOLATED_NETCDF:
        log.warning(
            "STEP5_WRITE_INTERPOLATED_NETCDF=True istendi; ancak bellek dostu "
            "Step5 akışı şu anda NetCDF yazmıyor. Raster çıktıları windowed "
            "işleme ile yazıldı."
        )

    metadata = {
        "step": "step5_preprocess_timeseries",
        "method": "windowed_zscore_anomaly",
        "created_at": datetime.now().isoformat(),
        "input_dirs": {
            "baseline_timeseries": str(BASELINE_INPUT_DIR),
            "qa_masks": str(QA_DIR),
            "current_period": str(CURRENT_PERIOD_DIR),
        },
        "log_file": str(log_file),
        "processing": {
            "mode": "windowed",
            "window_size": STEP5_WINDOW_SIZE,
            "window_count": result["window_count"],
            "std_epsilon": STEP5_STD_EPSILON,
            "estimated_stack_memory_mb": result["estimated_stack_memory_mb"],
            "netcdf_written": baseline_netcdf is not None,
        },
        "baseline": {
            "time_count": len(tif_files),
            "date_range": f"{times[0]} to {times[-1]}",
            "interpolation_method": "linear_internal_gaps_per_window",
            "mean_celsius": result["baseline_mean_stats"]["mean"],
            "std_celsius": result["baseline_std_stats"]["mean"],
            "input_files": [path.name for path in tif_files],
        },
        "current_period": {
            "window_days": CURRENT_PERIOD_DAYS,
            "end_date": CURRENT_PERIOD_END_DATE,
            "input_file": current_path.name,
            "mean_celsius": result["current_stats"]["mean"],
        },
        "anomaly": {
            "method": "z_score",
            "formula": "(current_median - baseline_mean) / baseline_std",
            "coverage_percent": result["coverage_percent"],
            "mean_zscore": result["anomaly_stats"]["mean"],
            "min_zscore": result["anomaly_stats"]["min"],
            "max_zscore": result["anomaly_stats"]["max"],
        },
        "masking": {
            "qa_source": "QA_PIXEL",
            "physical_range": "[-30, 80] Celsius",
        },
        "outputs": {
            "baseline_mean": output_paths["baseline_mean"].name,
            "baseline_std": output_paths["baseline_std"].name,
            "current_median": output_paths["current_median"].name,
            "anomaly_zscore": output_paths["anomaly_zscore"].name,
            "baseline_netcdf": baseline_netcdf,
        },
        "status": "processed",
    }

    metadata_path = OUTPUT_DIR / "step5_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def main() -> None:
    """Komut satırından çalıştırıldığında Step5 pencere bazlı akışı başlatır."""
    log.info("=" * 60)
    log.info("STEP 5 BAŞLIYOR (windowed/chunked işleme)")
    log.info("=" * 60)

    tif_files = list_baseline_tifs()
    current_path = list_current_period_tifs()[0]

    result = process_step5_windowed(tif_files, current_path)
    metadata_path = write_metadata(result, tif_files, current_path)

    log.info("Baseline ortalaması: %.2f C", result["baseline_mean_stats"]["mean"])
    log.info("Baseline standart sapması: %.2f C", result["baseline_std_stats"]["mean"])
    log.info("Current period ortalaması: %.2f C", result["current_stats"]["mean"])
    log.info("Geçerli anomali kapsaması: %.1f%%", result["coverage_percent"])
    log.info("Anomali ortalaması: %.2f sigma", result["anomaly_stats"]["mean"])
    log.info(
        "Anomali aralığı: [%.2f, %.2f] sigma",
        result["anomaly_stats"]["min"],
        result["anomaly_stats"]["max"],
    )
    log.info("Metadata kaydedildi: %s", metadata_path)
    log.info("Çıktı klasörü: %s", OUTPUT_DIR)
    log.info("=" * 60)
    log.info("STEP 5 TAMAMLANDI")


if __name__ == "__main__":
    main()