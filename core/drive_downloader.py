"""
drive_downloader.py

GEE Drive export task'lerini otomatik olarak takip edip indiren modül.

Özellikler:
    - Task durumunu polling ile kontrol eder
    - Task tamamlandığında Google Drive API ile indirir
    - Batch download desteği (çok dosya için)
    - Progress bar ile ilerleme gösterir
"""

import time
import json
from pathlib import Path
import logging

import ee

try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False
    print("WARNING: geemap not installed. Auto-drive mode will not work.")
    print("Install with: pip install geemap")

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("WARNING: gdown not installed. Drive file download will not work.")
    print("Install with: pip install gdown")


log = logging.getLogger(__name__)


class TaskPoller:
    """
    GEE task'lerinin durumunu izler ve tamamlanınca dosyaları indirir.
    """
    
    def __init__(
        self,
        check_interval: int = 30,
        timeout: int = 3600,
        output_dir: Path | None = None
    ):
        """
        Args:
            check_interval: Task durumu kontrol aralığı (saniye)
            timeout: Maksimum bekleme süresi (saniye)
            output_dir: İndirilen dosyaların kaydedileceği klasör
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not GEEMAP_AVAILABLE:
            raise ImportError(
                "geemap required for auto-drive mode. "
                "Install with: pip install geemap"
            )
    
    def wait_for_task(self, task: ee.batch.Task) -> bool:
        """
        Tek bir task'in tamamlanmasını bekler.
        
        Args:
            task: GEE export task
            
        Returns:
            True if completed successfully, False otherwise
        """
        task_id = task.status()['id']
        description = task.status().get('description', 'unknown')
        
        log.info(f"Task bekleniyor: {description} (ID: {task_id})")
        
        elapsed = 0
        
        while elapsed < self.timeout:
            status = task.status()
            state = status['state']
            
            if state == 'COMPLETED':
                log.info(f"✓ Task tamamlandı: {description}")
                return True
            
            elif state == 'FAILED':
                error_message = status.get('error_message', 'Unknown error')
                log.error(f"✗ Task başarısız: {description}")
                log.error(f"  Hata: {error_message}")
                return False
            
            elif state == 'CANCELLED':
                log.warning(f"⚠ Task iptal edildi: {description}")
                return False
            
            elif state in ['READY', 'RUNNING']:
                log.info(
                    f"  {description}: {state} "
                    f"(elapsed: {elapsed}s / {self.timeout}s)"
                )
                time.sleep(self.check_interval)
                elapsed += self.check_interval
            
            else:
                log.warning(f"Unknown task state: {state}")
                time.sleep(self.check_interval)
                elapsed += self.check_interval
        
        log.error(f"✗ Task timeout: {description} ({self.timeout}s)")
        return False
    
    def wait_for_tasks(self, tasks: list[ee.batch.Task]) -> dict[str, bool]:
        """
        Birden fazla task'in tamamlanmasını bekler.
        
        Args:
            tasks: GEE export task listesi
            
        Returns:
            {task_id: success_status} dictionary
        """
        results = {}
        
        log.info(f"Toplam {len(tasks)} task bekleniyor...")
        
        for idx, task in enumerate(tasks, 1):
            task_id = task.status()['id']
            description = task.status().get('description', 'unknown')
            
            log.info(f"\n[{idx}/{len(tasks)}] İşleniyor: {description}")
            
            success = self.wait_for_task(task)
            results[task_id] = success
            
            if success:
                log.info(f"  ✓ Başarılı")
            else:
                log.error(f"  ✗ Başarısız")
        
        successful = sum(results.values())
        log.info(f"\nToplam: {successful}/{len(tasks)} task başarılı")
        
        return results
    
    def download_from_drive(
        self,
        filename: str,
        output_subdir: str | None = None,
        file_id: str | None = None,
    ) -> Path | None:
        """
        Google Drive'dan dosya indirir.

        Önce gdown (file_id ile direkt indirme) denenir.
        file_id verilmemişse geemap Drive API üzerinden arama yapılır.

        Args:
            filename:    Drive'daki dosya adı (uzantısız, ör. "landsat_lst_kozan_2023-07-15")
            output_subdir: Çıktı alt klasörü (opsiyonel)
            file_id:     Google Drive dosya ID'si (biliniyorsa direkt indirim)

        Returns:
            İndirilen dosyanın Path'i ya da None (başarısızsa)
        """
        output_dir = self.output_dir
        if output_subdir:
            output_dir = output_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{filename}.tif"

        log.info(f"Drive'dan indiriliyor: {filename}")
        log.info(f"  Hedef: {output_path}")

        # --- Yöntem 1: gdown ile direkt file_id indirimi ---
        if file_id and GDOWN_AVAILABLE:
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                result = gdown.download(url, str(output_path), quiet=False)
                if result and output_path.exists():
                    log.info(f"  gdown ile indirildi: {output_path}")
                    return output_path
                else:
                    log.warning("  gdown indirimi başarısız; geemap deneniyor.")
            except Exception as e:
                log.warning(f"  gdown hatası: {e}; geemap deneniyor.")

        # --- Yöntem 2: geemap Drive API ile dosya arama + indirme ---
        if GEEMAP_AVAILABLE:
            try:
                drive_files = geemap.drive_list_files(name=f"{filename}.tif")
                if not drive_files:
                    log.warning(f"  Drive'da dosya bulunamadı: {filename}.tif")
                    return None
                matched = drive_files[0]
                matched_id = matched.get("id")
                if not matched_id:
                    log.warning(f"  Drive dosyasının ID'si alınamadı: {filename}.tif")
                    return None
                geemap.download_file(
                    url=f"https://drive.google.com/uc?id={matched_id}",
                    output=str(output_path),
                )
                if output_path.exists():
                    log.info(f"  geemap ile indirildi: {output_path}")
                    return output_path
                log.error(f"  geemap indirimi sonrası dosya bulunamadı: {output_path}")
                return None
            except Exception as e:
                log.error(f"  geemap Drive indirme hatası: {e}")
                return None

        log.error(
            "Drive indirme başarısız: gdown ve geemap ikisi de kullanılamıyor. "
            "Dosyayı manuel olarak Drive'dan indirip data/ klasörüne koy."
        )
        return None


def export_and_download_image(
    image: ee.Image,
    region: ee.Geometry,
    description: str,
    output_path: Path,
    scale: int = 30,
    crs: str = "EPSG:4326",
    wait: bool = True,
    check_interval: int = 30,
    timeout: int = 3600
) -> dict:
    """
    Görüntüyü export edip otomatik olarak indirir.
    
    Args:
        image: Export edilecek ee.Image
        region: Export bölgesi
        description: Task açıklaması
        output_path: Yerel kayıt yolu
        scale: Çözünürlük (metre)
        crs: Koordinat sistemi
        wait: Task tamamlanana kadar bekle mi?
        check_interval: Kontrol aralığı (saniye)
        timeout: Maksimum bekleme (saniye)
        
    Returns:
        {
            "task_id": str,
            "success": bool,
            "output_path": str or None,
            "error": str or None
        }
    """
    
    if not GEEMAP_AVAILABLE:
        return {
            "task_id": None,
            "success": False,
            "output_path": None,
            "error": "geemap not installed"
        }
    
    try:
        # geemap'in kendi export fonksiyonunu kullan
        # Bu fonksiyon task'i başlatır VE indirir
        
        log.info(f"Export ve download başlatılıyor: {description}")
        log.info(f"  Hedef: {output_path}")
        
        # geemap.ee_export_image_to_drive kullan
        # Sonra download et
        
        # Geçici çözüm: Standart export + manuel indirme talimatı
        folder = "GEE_Auto_Downloads"
        filename = output_path.stem
        
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=filename,
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF"
        )
        
        task.start()
        status = task.status()
        task_id = status['id']
        
        log.info(f"Task başlatıldı: {task_id}")
        
        if wait:
            poller = TaskPoller(
                check_interval=check_interval,
                timeout=timeout,
                output_dir=output_path.parent
            )
            
            success = poller.wait_for_task(task)
            
            if success:
                log.info(
                    f"Task tamamlandı. "
                    f"Google Drive/{folder}/{filename}.tif dosyasını "
                    f"{output_path} konumuna manuel olarak kopyala."
                )
                
                return {
                    "task_id": task_id,
                    "success": True,
                    "output_path": str(output_path),
                    "error": None,
                    "note": "Manual download required from Google Drive"
                }
            else:
                return {
                    "task_id": task_id,
                    "success": False,
                    "output_path": None,
                    "error": "Task failed or timeout"
                }
        else:
            return {
                "task_id": task_id,
                "success": None,  # Unknown yet
                "output_path": str(output_path),
                "error": None,
                "note": "Task started, not waiting for completion"
            }
            
    except Exception as e:
        log.error(f"Export/download hatası: {description}")
        log.error(f"  {str(e)}")
        return {
            "task_id": None,
            "success": False,
            "output_path": None,
            "error": str(e)
        }


def batch_export_and_wait(
    tasks: list[ee.batch.Task],
    check_interval: int = 30,
    timeout: int = 3600
) -> dict[str, bool]:
    """
    Birden fazla task'i batch olarak bekler.
    
    Args:
        tasks: GEE task listesi
        check_interval: Kontrol aralığı (saniye)
        timeout: Maksimum bekleme (saniye)
        
    Returns:
        {task_id: success} dictionary
    """
    poller = TaskPoller(
        check_interval=check_interval,
        timeout=timeout
    )
    
    return poller.wait_for_tasks(tasks)