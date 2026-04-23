# Uydu LST Tabanlı Termal Çevre Temsili Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan MODIS uydu verilerini kullanarak Doğu Akdeniz (Adana/Mersin/Hatay) bölgesi için Yüzey Sıcaklığı (LST) boru hattı oluşturan ve sıcaklık anomalilerini görselleştiren bir veri mühendisliği prototipidir.

## 🚀 Proje Kapsamı (Mevcut Durum)
Sistem şu an temel bir veri işleme boru hattı (pipeline) olarak çalışmaktadır:
1. **Veri Kaynağı:** MODIS/061/MOD11A1 (1km çözünürlük).
2. **Sorgu ve Filtreleme:** Doğu Akdeniz bölgesi için son 5 yılın (2019-2023) sadece yaz ayları (Haziran-Eylül) filtrelenmektedir.
3. **Dönüşüm ve İndirgeme:** GEE sunucularında DN -> °C dönüşümü yapılmakta ve 5 yıllık ortalama alınarak GeoTIFF formatında Google Drive'a aktarılmaktadır (Batch Export).
4. **Yerel Ön İşleme:** İndirilen GeoTIFF dosyaları `rasterio` ile okunup, `matplotlib` ile 2B termal harita olarak görselleştirilmektedir.

## 🛠️ Kurulum ve Kullanım
**1. Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt