# Uydu Tabanlı Termal Dijital İkiz Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS** ve **Landsat** yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesi için **termal çevre temsili** oluşturan modüler bir prototip sistemdir. Proje; veri sorgulama, sıcaklık işleme, GeoTIFF export, Python ile ön işleme, zaman serisi hazırlama ve ileride eklenecek 3B görselleştirme / yangın riski analizi adımlarını içeren bir yapı hedeflemektedir.

## Amaç

Uydu tabanlı yüzey sıcaklığı verilerini işleyerek belirli bir bölgedeki sıcaklık dağılımını incelemek, düşük ve yüksek çözünürlüklü termal katmanlar üretmek ve ileride geliştirilecek dijital ikiz / risk analizi çalışmaları için temel oluşturmak.

## Kapsam

Projede şu adımlar yer almaktadır:

- bölge seçimi
- MODIS LST verisinin çekilmesi
- sıcaklık dönüşümü
- GeoTIFF üretimi
- 2B görselleştirme
- anomali haritası oluşturma

## Mevcut Durum

Şu anda geliştirilen / tamamlanan kısımlar:

- GEE bağlantısı
- bölge tanımlama
- veri çekme
- sıcaklık rasteri oluşturma
- GeoTIFF indirme / üretme
- temel görselleştirme

## Planlanan Çalışmalar

Henüz tamamlanmamış veya sonraki aşamalarda geliştirilecek kısımlar:

- 3B görselleştirme katmanı
- gelişmiş risk / anomali analizi
- karar destek yapısı
- Jetson / YOLO entegrasyonu

## Veri Kaynağı

- **MODIS/061/MOD11A1**
- günlük kara yüzey sıcaklığı verisi
- yaklaşık 1 km çözünürlük

## Kurulum

Projeyi klonlayın:

```bash
git clone https://github.com/emrehann17/satellite-thermal-digital-twin.git
cd satellite-thermal-digital-twin
```

Sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt