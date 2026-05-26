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
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


BASE_DIR = Path(__file__).resolve().parent
BASELINE_INPUT_DIR = BASE_DIR / "data" / "landsat_timeseries"
QA_DIR = BASE_DIR / "data" / "landsat_qa"
CURRENT_PERIOD_DIR = BASE_DIR / "data" / "current_period"
MODIS_INPUT_DIR = BASE_DIR / "data" / "modis"
OUTPUT_DIR = BASE_DIR / "outputs" / "step5"
STEP4_METADATA_PATH = BASE_DIR / "outputs" / "step4" / "step4_metadata.json"

BASELINE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_PERIOD_DIR.mkdir(parents=True, exist_ok=True)
MODIS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    qa = qa_array.astype(np.uint16)
    bad_pixels = fill | dilated_cloud | cirrus | cloud | cloud_shadow | snow

    cloud_confidence = (qa >> 8) & 3
    cloud_shadow_confidence = (qa >> 10) & 3
    snow_ice_confidence = (qa >> 12) & 3
    cirrus_confidence = (qa >> 14) & 3

    return (
        ((qa & bad_pixels) == 0)
        & (cloud_confidence < 2)
        & (cloud_shadow_confidence < 2)
        & (snow_ice_confidence < 2)
        & (cirrus_confidence < 2)
    )


def list_baseline_tifs() -> list[Path]:
    """
    Baseline zaman serisini oluşturan GeoTIFF dosyalarını listeler.

    QA dosyaları aynı klasöre yanlışlıkla konmuşsa `_qa` ile bitenleri
    baseline görüntüsü olarak kullanmaz.
    """
    metadata_files = list_baseline_tifs_from_step4_metadata()
    if metadata_files:
        return metadata_files

    log.warning(
        "Step4 metadata'dan baseline dosya listesi okunamadı; klasördeki tüm "
        "uygun Landsat LST tifleri kullanılacak. Eski export kalıntıları varsa "
        "baseline'a karışabilir."
    )

    tif_files = sorted(
        path
        for path in BASELINE_INPUT_DIR.glob("*.tif")
        if not is_qa_tif_name(path.name)
        and path.name.lower().startswith(LANDSAT_EXPORT["file_name_prefix"].lower())
        and BASELINE_START_DATE <= str(extract_date_from_filename(path)) <= BASELINE_END_DATE
    )

    if not tif_files:
        raise FileNotFoundError(
            f"Baseline GeoTIFF dosyası bulunamadı: {BASELINE_INPUT_DIR}\n"
            "Step4 baseline zaman serisi export dosyalarını bu klasöre koymalısın."
        )

    return tif_files


def list_baseline_tifs_from_step4_metadata() -> list[Path]:
    """
    Step4 metadata varsa yalnız son export edilen baseline LST dosyalarını seçer.

    Böylece data/landsat_timeseries içinde önceki denemelerden kalan dosyalar
    yeni Step5 baseline yığınına sessizce karışmaz.
    """
    if not STEP4_METADATA_PATH.exists():
        return []

    try:
        metadata = json.loads(STEP4_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Step4 metadata JSON okunamadı: %s", STEP4_METADATA_PATH)
        return []

    exports = (
        metadata
        .get("exports", {})
        .get("landsat_baseline_timeseries", [])
    )
    selected_files: list[Path] = []

    for export_record in exports:
        lst_prefix = export_record.get("lst", {}).get("file_name_prefix")
        if not lst_prefix:
            continue

        matches = sorted(BASELINE_INPUT_DIR.glob(f"{lst_prefix}*.tif"))
        matches = [path for path in matches if not is_qa_tif_name(path.name)]

        if not matches:
            log.warning("Step4 metadata'daki LST dosyası bulunamadı: %s", lst_prefix)
            continue

        selected_files.extend(matches)

    return sorted(set(selected_files))


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

    expected_prefix = f"landsat_current_period_{CURRENT_PERIOD_DAYS}days"
    matching_files = [
        path
        for path in current_files
        if path.name.lower().startswith(expected_prefix.lower())
    ]

    if matching_files:
        current_files = matching_files
    elif len(current_files) > 1:
        log.warning(
            "CURRENT_PERIOD_DAYS=%s ile eşleşen current raster bulunamadı; "
            "klasördeki ilk dosya kullanılacak.",
            CURRENT_PERIOD_DAYS,
        )

    if len(current_files) > 1:
        log.warning(
            "Birden fazla current period dosyası bulundu; ilki kullanılıyor: %s",
            current_files[0].name,
        )

    return current_files


def list_modis_context_tifs() -> list[Path]:
    """
    Step4b'nin data/modis klasörüne yerleştirdiği MODIS mean/std GeoTIFF'lerini listeler.

    MODIS dosyası Step5 için zorunlu değildir; bulunursa düşük çözünürlüklü bağlam
    z-score ürünü ve Landsat gridine yeniden örneklenmiş mean/std katmanları yazılır.
    """
    if not ENABLE_MODIS_STEP5_CONTEXT:
        return []

    expected_prefix = MODIS_EXPORT["file_name_prefix"].lower()
    modis_files = sorted(
        path
        for path in MODIS_INPUT_DIR.glob("*.tif")
        if path.name.lower().startswith(expected_prefix)
    )

    if not modis_files:
        log.warning(
            "MODIS Step5 bağlamı açık ama MODIS GeoTIFF bulunamadı: %s",
            MODIS_INPUT_DIR,
        )
        return []

    if len(modis_files) > 1:
        log.warning(
            "Birden fazla MODIS GeoTIFF bulundu; ilk dosya kullanılacak: %s",
            modis_files[0].name,
        )

    return modis_files


def is_qa_tif_name(filename: str) -> bool:
    """
    Landsat QA GeoTIFF adını tek parça ve GEE parçalı export adlarında yakalar.

    Örnekler:
        landsat_lst_dogu_akdeniz_2020-06-01_qa.tif
        landsat_lst_dogu_akdeniz_2020-06-01_qa-0000000000-0000000000.tif
    """
    stem = Path(filename).stem.lower()
    return bool(re.search(r"_qa($|[-_])", stem))


def find_qa_path_for_landsat_tif(tif_path: Path) -> Path | None:
    """
    LST GeoTIFF için eşleşen QA dosyasını bulur.

    Tek parça export:
        landsat_lst_..._2020-06-01.tif -> landsat_lst_..._2020-06-01_qa.tif

    Parçalı GEE export:
        landsat_lst_..._2020-06-01-0000000000-0000000000.tif
        landsat_lst_..._2020-06-01_qa-0000000000-0000000000.tif
    """
    exact_path = QA_DIR / f"{tif_path.stem}_qa{tif_path.suffix}"
    if exact_path.exists():
        return exact_path

    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", tif_path.stem)
    if not date_match:
        return None

    base_stem = tif_path.stem[: date_match.end()]
    suffix_after_date = tif_path.stem[date_match.end() :]

    candidates = []
    if suffix_after_date:
        candidates.append(QA_DIR / f"{base_stem}_qa{suffix_after_date}{tif_path.suffix}")
    candidates.extend(sorted(QA_DIR.glob(f"{base_stem}_qa*.tif")))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def read_window(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """
    Tek bant rasterdan belirtilen pencereyi float32 olarak okur.

    Rasterio masked array döndürdüğünde maskeli pikseller NaN'a çevrilir.
    """
    array = src.read(1, window=window, masked=True).astype("float32")
    return array.filled(np.nan)


def read_band_window(
    src: rasterio.io.DatasetReader,
    window: Window,
    band_index: int,
    default: float = np.nan,
) -> np.ndarray:
    """
    İstenen bandı float32 okur; bant yoksa aynı pencere boyutunda default döndürür.

    Bu, eski tek bant current period rasterlarıyla geriye dönük uyumluluk sağlar.
    Yeni current period exportlarında 2. bant QA-temiz gözlem sayısıdır.
    """
    if src.count < band_index:
        return np.full(
            (int(window.height), int(window.width)),
            default,
            dtype="float32",
        )

    array = src.read(band_index, window=window, masked=True).astype("float32")
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


def mask_physical_celsius(array: np.ndarray) -> np.ndarray:
    """Fiziksel LST aralığı dışındaki değerleri NaN yapar."""
    return np.where((array > -30) & (array < 80), array, np.nan).astype("float32")


def open_output(path: Path, profile: dict):
    """Çıktı GeoTIFF dosyasını yazma modunda açar."""
    return rasterio.open(path, "w", **output_profile(profile))


def estimated_stack_memory_mb(time_count: int, window_size: int) -> float:
    """
    Bir baseline pencere yığınının yaklaşık bellek kullanımını MB cinsinden hesaplar.

    Hesap yalnızca `time_count x window_size x window_size` float32 yığınını kapsar;
    küçük ara diziler ve çıktı blokları buna dahil değildir.
    """
    bytes_per_float32 = np.dtype("float32").itemsize
    return time_count * window_size * window_size * bytes_per_float32 / (1024**2)


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

        lst_celsius = mask_physical_celsius(lst_celsius)

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
        self._mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, array: np.ndarray) -> None:
        """Yeni bir pencere dizisindeki geçerli pikselleri istatistiklere ekler."""
        valid = np.isfinite(array)
        count = int(np.sum(valid))

        if count == 0:
            return

        values = array[valid].astype("float64")
        batch_mean = float(np.mean(values))
        batch_m2 = float(np.sum((values - batch_mean) ** 2))

        if self.count == 0:
            self.count = count
            self._mean = batch_mean
            self._m2 = batch_m2
        else:
            old_count = self.count
            new_count = old_count + count
            delta = batch_mean - self._mean
            self._mean += delta * count / new_count
            self._m2 += batch_m2 + delta * delta * old_count * count / new_count
            self.count = new_count

        self.min = min(self.min, float(np.min(values)))
        self.max = max(self.max, float(np.max(values)))

    @property
    def mean(self) -> float:
        """Biriktirilen tüm geçerli piksellerin ortalamasını döndürür."""
        return float("nan") if self.count == 0 else self._mean

    @property
    def std(self) -> float:
        """Biriktirilen tüm geçerli piksellerin standart sapmasını döndürür."""
        if self.count == 0:
            return float("nan")

        variance = max(self._m2 / self.count, 0.0)
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
    modis_path: Path | None = None,
) -> dict:
    """
    Step5 işlemini bellek dostu pencere tabanlı akışla çalıştırır.

    Ana işlem sırası:
        1. Baseline dosyalarını tarihe göre sıralar.
        2. Tüm rasterların aynı gridde olduğunu doğrular.
        3. Rasterı STEP5_WINDOW_SIZE boyutlu pencerelere böler.
        4. Her pencere için baseline zaman yığınını okur.
        5. Baseline mean/std, current median ve z-score anomali hesaplar.
        6. Sonuçları aynı anda çıktı GeoTIFF dosyalarına yazar.

    Böylece full raster zaman serisi belleğe alınmaz.
    """
    times = [extract_date_from_filename(path) for path in tif_files]
    sort_order = np.argsort(times)
    tif_files = [tif_files[index] for index in sort_order]
    times = [times[index] for index in sort_order]

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile.copy()

    for tif_path in tif_files[1:]:
        validate_same_grid(profile, tif_path)

    validate_same_grid(profile, current_path)
    for tif_path in tif_files:
        qa_path = find_qa_path_for_landsat_tif(tif_path)
        if qa_path is not None:
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
    log.info("Minimum geçerli baseline gözlem sayısı: %s", STEP5_MIN_BASELINE_VALID_COUNT)
    log.info("Pencere başına yaklaşık baseline stack belleği: %.1f MB", memory_estimate_mb)

    baseline_mean_stats = RunningStats()
    baseline_std_stats = RunningStats()
    current_stats = RunningStats()
    anomaly_stats = RunningStats()
    modis_mean_stats = RunningStats()
    modis_std_stats = RunningStats()
    modis_context_anomaly_stats = RunningStats()

    total_pixels = width * height
    valid_anomaly_pixels = 0
    valid_modis_context_pixels = 0

    output_paths = {
        "baseline_mean": OUTPUT_DIR / "baseline_lst_mean_celsius.tif",
        "baseline_std": OUTPUT_DIR / "baseline_lst_std_celsius.tif",
        "baseline_valid_count": OUTPUT_DIR / "baseline_valid_count.tif",
        "low_baseline_count_mask": OUTPUT_DIR / "low_baseline_count_mask.tif",
        "low_baseline_std_mask": OUTPUT_DIR / "low_baseline_std_mask.tif",
        "current_median": OUTPUT_DIR / "current_period_median_celsius.tif",
        "current_valid_count": OUTPUT_DIR / "current_period_valid_count.tif",
        "low_current_count_mask": OUTPUT_DIR / "low_current_count_mask.tif",
        "anomaly_zscore": OUTPUT_DIR / "anomaly_zscore.tif",
    }
    if modis_path is not None:
        output_paths.update(
            {
                "modis_mean_resampled": OUTPUT_DIR / "modis_lst_mean_celsius_resampled.tif",
                "modis_std_resampled": OUTPUT_DIR / "modis_lst_std_celsius_resampled.tif",
                "modis_context_zscore": OUTPUT_DIR / "modis_context_zscore.tif",
            }
        )

    qa_paths = [find_qa_path_for_landsat_tif(path) for path in tif_files]

    with ExitStack() as stack_context:
        baseline_datasets = [
            stack_context.enter_context(rasterio.open(path)) for path in tif_files
        ]
        qa_datasets = [
            stack_context.enter_context(rasterio.open(path)) if path is not None else None
            for path in qa_paths
        ]
        modis_vrt = None
        if modis_path is not None:
            modis_src = stack_context.enter_context(rasterio.open(modis_path))
            if modis_src.count < 2:
                raise ValueError(
                    f"MODIS GeoTIFF en az iki bant içermeli (mean/std): {modis_path.name}"
                )

            modis_vrt = stack_context.enter_context(
                WarpedVRT(
                    modis_src,
                    crs=profile["crs"],
                    transform=profile["transform"],
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                )
            )

        with (
            rasterio.open(current_path) as current_src,
            open_output(output_paths["baseline_mean"], profile) as mean_dst,
            open_output(output_paths["baseline_std"], profile) as std_dst,
            open_output(output_paths["baseline_valid_count"], profile) as valid_count_dst,
            open_output(
                output_paths["low_baseline_count_mask"],
                profile,
            ) as low_baseline_count_dst,
            open_output(output_paths["low_baseline_std_mask"], profile) as low_baseline_std_dst,
            open_output(output_paths["current_median"], profile) as current_dst,
            open_output(output_paths["current_valid_count"], profile) as current_count_dst,
            open_output(output_paths["low_current_count_mask"], profile) as low_current_count_dst,
            open_output(output_paths["anomaly_zscore"], profile) as anomaly_dst,
        ):
            modis_mean_dst = None
            modis_std_dst = None
            modis_context_dst = None
            if modis_vrt is not None:
                modis_mean_dst = stack_context.enter_context(
                    open_output(output_paths["modis_mean_resampled"], profile)
                )
                modis_std_dst = stack_context.enter_context(
                    open_output(output_paths["modis_std_resampled"], profile)
                )
                modis_context_dst = stack_context.enter_context(
                    open_output(output_paths["modis_context_zscore"], profile)
                )

            current_has_valid_count = current_src.count >= 2
            if not current_has_valid_count:
                log.warning(
                    "Current period rasterında valid-count bandı yok. "
                    "STEP5_MIN_CURRENT_VALID_COUNT uygulanmayacak; current export'u "
                    "Step3/Step4 ile yeniden üretmek gerekir."
                )

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

                valid_count = np.sum(np.isfinite(baseline_stack), axis=0).astype(
                    "float32"
                )
                baseline_mean = nanmean_float32(baseline_stack)
                baseline_std = nanstd_float32(baseline_stack)
                enough_observations = valid_count >= STEP5_MIN_BASELINE_VALID_COUNT
                baseline_mean = np.where(
                    enough_observations,
                    baseline_mean,
                    np.nan,
                ).astype("float32")
                baseline_std = np.where(
                    enough_observations,
                    baseline_std,
                    np.nan,
                ).astype("float32")

                current_median = read_band_window(current_src, window, band_index=1)
                current_valid_count = read_band_window(
                    current_src,
                    window,
                    band_index=2,
                    default=np.nan,
                )
                if current_src.count < 2:
                    current_valid_count = np.where(
                        np.isfinite(current_median),
                        float(STEP5_MIN_CURRENT_VALID_COUNT),
                        np.nan,
                    ).astype("float32")

                current_median = mask_physical_celsius(current_median)
                enough_current = current_valid_count >= STEP5_MIN_CURRENT_VALID_COUNT
                current_median = np.where(enough_current, current_median, np.nan).astype(
                    "float32"
                )

                low_baseline_count_mask = (~enough_observations).astype("float32")
                low_baseline_std_mask = (
                    enough_observations
                    & np.isfinite(baseline_std)
                    & (baseline_std < STEP5_MIN_BASELINE_STD_CELSIUS)
                ).astype("float32")
                low_current_count_mask = (
                    np.isfinite(current_valid_count)
                    & (current_valid_count < STEP5_MIN_CURRENT_VALID_COUNT)
                ).astype("float32")

                with np.errstate(invalid="ignore", divide="ignore"):
                    anomaly_zscore = np.where(
                        enough_current
                        & (baseline_std >= STEP5_MIN_BASELINE_STD_CELSIUS),
                        (current_median - baseline_mean) / baseline_std,
                        np.nan,
                    ).astype("float32")
                anomaly_zscore = np.where(
                    np.isfinite(anomaly_zscore),
                    anomaly_zscore,
                    np.nan,
                ).astype("float32")

                if modis_vrt is not None:
                    modis_mean = mask_physical_celsius(
                        read_band_window(modis_vrt, window, band_index=1)
                    )
                    modis_std = read_band_window(modis_vrt, window, band_index=2)
                    modis_std = np.where(
                        np.isfinite(modis_std) & (modis_std >= 0),
                        modis_std,
                        np.nan,
                    ).astype("float32")

                    with np.errstate(invalid="ignore", divide="ignore"):
                        modis_context_zscore = np.where(
                            enough_current
                            & (modis_std >= STEP5_MIN_MODIS_STD_CELSIUS),
                            (current_median - modis_mean) / modis_std,
                            np.nan,
                        ).astype("float32")
                    modis_context_zscore = np.where(
                        np.isfinite(modis_context_zscore),
                        modis_context_zscore,
                        np.nan,
                    ).astype("float32")
                else:
                    modis_mean = None
                    modis_std = None
                    modis_context_zscore = None

                mean_dst.write(baseline_mean, 1, window=window)
                std_dst.write(baseline_std, 1, window=window)
                valid_count_dst.write(valid_count, 1, window=window)
                low_baseline_count_dst.write(low_baseline_count_mask, 1, window=window)
                low_baseline_std_dst.write(low_baseline_std_mask, 1, window=window)
                current_dst.write(current_median, 1, window=window)
                current_count_dst.write(current_valid_count, 1, window=window)
                low_current_count_dst.write(low_current_count_mask, 1, window=window)
                anomaly_dst.write(anomaly_zscore, 1, window=window)
                if modis_context_zscore is not None:
                    modis_mean_dst.write(modis_mean, 1, window=window)
                    modis_std_dst.write(modis_std, 1, window=window)
                    modis_context_dst.write(modis_context_zscore, 1, window=window)

                baseline_mean_stats.update(baseline_mean)
                baseline_std_stats.update(baseline_std)
                current_stats.update(current_median)
                anomaly_stats.update(anomaly_zscore)
                valid_anomaly_pixels += int(np.sum(np.isfinite(anomaly_zscore)))
                if modis_context_zscore is not None:
                    modis_mean_stats.update(modis_mean)
                    modis_std_stats.update(modis_std)
                    modis_context_anomaly_stats.update(modis_context_zscore)
                    valid_modis_context_pixels += int(
                        np.sum(np.isfinite(modis_context_zscore))
                    )

                if index == 1 or index == window_count or index % 10 == 0:
                    log.info("Pencere işlendi: %s/%s", index, window_count)

    coverage_pct = 100 * valid_anomaly_pixels / total_pixels
    modis_context_coverage_pct = 100 * valid_modis_context_pixels / total_pixels

    return {
        "profile": profile,
        "times": times,
        "tif_files": tif_files,
        "output_paths": output_paths,
        "baseline_mean_stats": baseline_mean_stats.as_dict(),
        "baseline_std_stats": baseline_std_stats.as_dict(),
        "current_stats": current_stats.as_dict(),
        "anomaly_stats": anomaly_stats.as_dict(),
        "modis_path": modis_path,
        "modis_mean_stats": modis_mean_stats.as_dict(),
        "modis_std_stats": modis_std_stats.as_dict(),
        "modis_context_anomaly_stats": modis_context_anomaly_stats.as_dict(),
        "coverage_percent": coverage_pct,
        "modis_context_coverage_percent": modis_context_coverage_pct,
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
            "modis_context": str(MODIS_INPUT_DIR),
        },
        "log_file": str(log_file),
        "processing": {
            "mode": "windowed",
            "window_size": STEP5_WINDOW_SIZE,
            "window_count": result["window_count"],
            "min_baseline_std_celsius": STEP5_MIN_BASELINE_STD_CELSIUS,
            "min_baseline_valid_count": STEP5_MIN_BASELINE_VALID_COUNT,
            "min_current_valid_count": STEP5_MIN_CURRENT_VALID_COUNT,
            "landsat_baseline_temporal_interpolation": False,
            "temporal_interpolation_used": False,
            "baseline_statistics_source": "observed_qa_clean_landsat_pixels_only",
            "insufficient_observations_policy": "mask_as_nan",
            "modis_context_enabled": ENABLE_MODIS_STEP5_CONTEXT,
            "modis_spatial_resampling": "bilinear_to_landsat_grid_for_context_only",
            "modis_used_as_primary_baseline": False,
            "min_modis_std_celsius": STEP5_MIN_MODIS_STD_CELSIUS,
            "estimated_stack_memory_mb": result["estimated_stack_memory_mb"],
            "netcdf_written": baseline_netcdf is not None,
        },
        "baseline": {
            "time_count": len(tif_files),
            "date_range": f"{times[0]} to {times[-1]}",
            "interpolation_method": "none",
            "temporal_interpolation_used": False,
            "landsat_baseline_temporal_interpolation": False,
            "statistics_source": "observed_qa_clean_landsat_pixels_only",
            "statistics": "np.nanmean/np.nanstd over QA-clean observations only",
            "insufficient_observations_policy": "mask_as_nan",
            "min_valid_count": STEP5_MIN_BASELINE_VALID_COUNT,
            "mean_celsius": result["baseline_mean_stats"]["mean"],
            "std_celsius": result["baseline_std_stats"]["mean"],
            "input_files": [path.name for path in tif_files],
        },
        "current_period": {
            "window_days": CURRENT_PERIOD_DAYS,
            "end_date": CURRENT_PERIOD_END_DATE,
            "input_file": current_path.name,
            "valid_count_band": (
                "Current_Period_Valid_Count if present; otherwise legacy fallback"
            ),
            "min_valid_count": STEP5_MIN_CURRENT_VALID_COUNT,
            "mean_celsius": result["current_stats"]["mean"],
        },
        "modis_context": {
            "enabled": ENABLE_MODIS_STEP5_CONTEXT,
            "input_file": (
                None if result["modis_path"] is None else result["modis_path"].name
            ),
            "role": (
                "coarse_context_only; Landsat z-score remains the primary "
                "high-resolution anomaly product"
            ),
            "resampling": "spatial_resampling_for_modis_context_only",
            "resampling_method": "bilinear_to_landsat_grid",
            "spatial_resampling": "bilinear_to_landsat_grid_for_context_only",
            "used_to_fill_landsat_baseline_gaps": False,
            "used_as_primary_baseline": False,
            "min_std_celsius": STEP5_MIN_MODIS_STD_CELSIUS,
            "mean_celsius": result["modis_mean_stats"]["mean"],
            "std_celsius": result["modis_std_stats"]["mean"],
            "coverage_percent": result["modis_context_coverage_percent"],
            "mean_zscore": result["modis_context_anomaly_stats"]["mean"],
            "min_zscore": result["modis_context_anomaly_stats"]["min"],
            "max_zscore": result["modis_context_anomaly_stats"]["max"],
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
            "baseline_valid_count": output_paths["baseline_valid_count"].name,
            "low_baseline_count_mask": output_paths["low_baseline_count_mask"].name,
            "low_baseline_std_mask": output_paths["low_baseline_std_mask"].name,
            "current_median": output_paths["current_median"].name,
            "current_valid_count": output_paths["current_valid_count"].name,
            "low_current_count_mask": output_paths["low_current_count_mask"].name,
            "anomaly_zscore": output_paths["anomaly_zscore"].name,
            "modis_mean_resampled": (
                output_paths.get("modis_mean_resampled").name
                if "modis_mean_resampled" in output_paths
                else None
            ),
            "modis_std_resampled": (
                output_paths.get("modis_std_resampled").name
                if "modis_std_resampled" in output_paths
                else None
            ),
            "modis_context_zscore": (
                output_paths.get("modis_context_zscore").name
                if "modis_context_zscore" in output_paths
                else None
            ),
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
    modis_files = list_modis_context_tifs()
    modis_path = modis_files[0] if modis_files else None

    result = process_step5_windowed(tif_files, current_path, modis_path=modis_path)
    metadata_path = write_metadata(result, tif_files, current_path)

    log.info("Baseline ortalaması: %.2f C", result["baseline_mean_stats"]["mean"])
    log.info("Baseline standart sapması: %.2f C", result["baseline_std_stats"]["mean"])
    log.info("Current period ortalaması: %.2f C", result["current_stats"]["mean"])
    log.info("Geçerli anomali kapsaması: %.1f%%", result["coverage_percent"])
    if modis_path is not None:
        log.info("MODIS bağlam dosyası: %s", modis_path.name)
        log.info(
            "MODIS bağlam z-score kapsaması: %.1f%%",
            result["modis_context_coverage_percent"],
        )
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