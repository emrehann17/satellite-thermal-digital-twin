# B7 – Uydu LST Tabanlı 3B Termal Dijital İkiz ve Mobil Arayüz

Bu proje, Google Earth Engine (GEE) üzerinden alınan uydu (MODIS ve Landsat) Yüzey Sıcaklığı (LST) verilerini kullanarak Doğu Akdeniz (Adana/Mersin/Hatay) bölgesi için 3 boyutlu bir termal dijital ikiz oluşturmayı ve Jetson Orin Nano üzerinde YOLOv7 ile orman yangını riski tahmini yapmayı amaçlamaktadır.

## 🚀 Proje Mimarisi (Veri Boru Hattı)
Sistem, bulut tabanlı büyük veri işleme ve yerel (edge) yapay zeka çıkarımı olmak üzere iki temel ayaktan oluşmaktadır:
1. **Google Earth Engine (GEE) API:** Uydu verilerinin (MODIS/Landsat) filtrelenmesi, sıcaklık dönüşümleri (°C) ve zaman serisi ortalamalarının alınması.
2. **Yerel Veri Ön İşleme (Local ETL):** İndirilen GeoTIFF dosyalarının `rasterio` ve `xarray` ile temizlenmesi ve interpolasyonu.
3. **Jetson & YOLOv7:** Hazırlanan termal harita altlıkları üzerinde yangın riski modellemesi.

## 📂 Klasör Yapısı
Proje, veri güvenliği ve modülerlik ilkelerine göre yapılandırılmıştır:
- `data/`: GEE üzerinden indirilen ham ve işlenmiş GeoTIFF dosyaları (Git'te izlenmez).
- `logs/`: Her bir modülün çalışma zamanı ve durum logları (Git'te izlenmez).
- `step1_kurulum.py`: GEE bağlantı testi ve örnek MODIS veri çekimi.
- `step2_modis_5year.py`: 5 Yıllık (Haziran-Eylül) MODIS verisinin filtrelenmesi, °C dönüşümü ve Google Drive'a toplu aktarımı (Batch Export).

## 📊 Geliştirme Durumu (Adım Adım)
- [x] **Adım 1:** GEE Python API kurulumu ve temel sorgular.
- [ ] **Adım 2:** MODIS MOD11A1 LST: Doğu Akdeniz son 5 yıl yaz ortalaması sorgusu.
- [ ] **Adım 3:** Landsat Collection 2 (30m) yüksek çözünürlüklü LST sorgusu.
- [ ] **Adım 4:** GEE'den offline indirme (GeoTIFF batch export to Drive).
- [ ] **Adım 5:** Python (rasterio) ile yerel veri ön işleme (Bulut maskeleme, DN→°C).
- [ ] **Adım 6-10:** 3B Dijital İkiz, YOLOv7 entegrasyonu ve FastAPI sunucusu.