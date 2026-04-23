# Uydu Tabanlı 2B Termal Çevre Temsili Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS LST** verilerini kullanarak Doğu Akdeniz bölgesi için **2B termal çevre temsili** oluşturan ve **sıcaklık anomalilerini görselleştiren** bir prototip sistemdir.

## Amaç

Uydu tabanlı yüzey sıcaklığı verilerini işleyerek belirli bir bölgedeki sıcaklık dağılımını incelemek ve ileride geliştirilebilecek dijital ikiz / risk analizi çalışmaları için temel bir yapı kurmak.

## Kapsam

Projede şu adımlar yer alır:

- bölge seçimi
- MODIS LST verisinin çekilmesi
- sıcaklık dönüşümü
- GeoTIFF üretimi
- 2B görselleştirme
- anomali haritası oluşturma

## Mevcut Durum

Tamamlanan kısımlar:

- GEE bağlantısı
- bölge tanımlama
- veri çekme
- sıcaklık rasteri oluşturma
- GeoTIFF indirme / üretme
- temel görselleştirme

## Planlanan Çalışmalar

Henüz tamamlanmamış kısımlar:

- 3B görselleştirme katmanı
- gelişmiş risk / anomali analizi
- karar destek yapısı
- Jetson / YOLO entegrasyonu

## Veri Kaynağı

- **MODIS/061/MOD11A1**
- günlük kara yüzey sıcaklığı verisi
- yaklaşık 1 km çözünürlük

## Kurulum

```bash
pip install -r requirements.txt