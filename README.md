# Uydu Tabanlı Termal Dijital İkiz Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS** ve **Landsat** yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesi için **termal çevre temsili** oluşturan bir prototip sistemdir. Proje; veri çekme, sıcaklık işleme, GeoTIFF export, Python ile ön işleme ve ileride eklenecek 3B görselleştirme / yangın riski analizi adımlarını içeren modüler bir yapı hedeflemektedir.

## Amaç

Uydu tabanlı yüzey sıcaklığı verilerini işleyerek belirli bir bölgedeki sıcaklık dağılımını incelemek, düşük ve yüksek çözünürlüklü termal katmanlar üretmek ve ileride geliştirilecek dijital ikiz / risk analizi çalışmaları için temel oluşturmak.

## Kapsam

Projede şu adımlar yer almaktadır:

* çalışma bölgesinin seçilmesi
* MODIS LST verisinin sorgulanması
* 5 yıllık yaz dönemi sıcaklık ortalamasının oluşturulması
* Landsat yüksek çözünürlüklü LST üretimi
* GeoTIFF export
* Python ile ön işleme
* zaman serisi hazırlığı
* 2B ve ileride 3B görselleştirme

## Mevcut Durum

Şu anda geliştirilen / tamamlanan kısımlar:

* GEE bağlantısı ve doğrulama
* çalışma bölgelerinin tanımlanması
* MODIS MOD11A1 veri sorgulama
* MODIS için 5 yıllık yaz ortalaması üretimi
* Landsat Collection 2 Surface Temperature işleme
* MODIS ve Landsat görüntülerinin GeoTIFF export akışı
* modüler proje yapısı (`core/` + step dosyaları)

## Planlanan Çalışmalar

Henüz tamamlanmamış veya geliştirilmeye devam eden kısımlar:

* Python tarafında ön işleme adımlarının genişletilmesi
* bulut maskeleme / kalite filtreleri
* zaman serisi ve interpolasyon
* 3B arazi + termal harita görselleştirmesi
* FIRMS yangın kayıtları ile etiketli veri seti oluşturma
* Jetson / YOLOv7-Tiny entegrasyonu
* mobil erişim ve API katmanı

## Veri Kaynakları

* **MODIS/061/MOD11A1**

  * günlük kara yüzey sıcaklığı verisi
  * yaklaşık 1 km çözünürlük

* **LANDSAT/LC08/C02/T1_L2**

  * Surface Temperature verisi
  * yaklaşık 30 m çözünürlük

## Proje Yapısı

```text
core/
├── config.py
├── gee_utils.py
├── regions.py
└── io_utils.py

step1_fetch_modis.py
step2_modis_5year_mean.py
step3_landsat_lst.py
step4_export_geotiff.py
step5_preprocess_timeseries.py

old_codes/
data/
outputs/
logs/
```

## Step Açıklamaları

### Step 1

GEE bağlantısını başlatır, çalışma bölgelerini tanımlar ve temel MODIS LST sorgusunu gerçekleştirir.

### Step 2

Doğu Akdeniz bölgesi için MODIS verisinden 5 yıllık yaz dönemi ortalama sıcaklık görüntüsü üretir.

### Step 3

Aynı bölge için Landsat tabanlı yüksek çözünürlüklü yüzey sıcaklığı görüntüsü üretir.

### Step 4

Step2 ve Step3 çıktılarının GeoTIFF olarak export edilmesini sağlar.

### Step 5

Python tarafında ön işleme, kalite kontrol, raster okuma ve zaman serisi hazırlığı için ayrılmış adımdır. Bu kısım henüz geliştirme aşamasındadır.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma Sırası

```bash
python step1_fetch_modis.py
python step2_modis_5year_mean.py
python step3_landsat_lst.py
python step4_export_geotiff.py
python step5_preprocess_timeseries.py
```

## Not

Bu repo şu anda tamamlanmış bir **3B termal dijital ikiz sistemi** değildir. Mevcut haliyle, uydu tabanlı termal veri işleme pipeline’ının modüler biçimde geliştirildiği bir prototip çalışmadır. Projenin sonraki aşamalarında 3B görselleştirme, risk analizi, veri birleştirme ve Jetson tabanlı model entegrasyonu eklenecektir.
