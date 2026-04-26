# Uydu Tabanlı Termal Dijital İkiz Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS** ve **Landsat** yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesi için **termal çevre temsili** oluşturan modüler bir prototip sistemdir. Proje; veri sorgulama, sıcaklık işleme, GeoTIFF export, Python ile ön işleme, zaman serisi hazırlama ve ileride eklenecek 3B görselleştirme / yangın riski analizi adımlarını içeren bir yapı hedeflemektedir.

## Amaç

Uydu tabanlı yüzey sıcaklığı verilerini işleyerek belirli bir bölgedeki sıcaklık dağılımını incelemek, düşük ve yüksek çözünürlüklü termal katmanlar üretmek ve ileride geliştirilecek dijital ikiz / risk analizi çalışmaları için temel oluşturmak.

## Kapsam

Projede şu adımlar yer almaktadır:

- çalışma bölgesinin seçilmesi
- MODIS LST verisinin sorgulanması
- 5 yıllık yaz dönemi sıcaklık ortalamasının oluşturulması
- Landsat yüksek çözünürlüklü LST üretimi
- GeoTIFF export
- Landsat zaman serisi export
- Python ile ön işleme
- QA tabanlı bulut maskeleme
- zaman serisi ve anomali üretimi
- 2B ve ileride 3B görselleştirme

## Mevcut Durum

Şu anda geliştirilen / tamamlanan kısımlar:

- GEE bağlantısı ve doğrulama
- çalışma bölgelerinin tanımlanması
- MODIS MOD11A1 veri sorgulama
- MODIS için 5 yıllık yaz ortalaması üretimi
- Landsat Collection 2 Surface Temperature işleme
- Landsat zaman serisi koleksiyonunun hazırlanması
- MODIS GeoTIFF export akışı
- Landsat LST ve QA GeoTIFF export akışı
- modüler proje yapısı (`core/` + step dosyaları)
- Step5 için raster okuma, QA tabanlı maskeleme, zaman serisi ve anomali üretimi mantığı kodlanmış olup doğrulama ve test süreci devam etmektedir

## Geliştirme Aşamasında Olan Kısımlar

Henüz test / iyileştirme aşamasında olan bölümler:

- Step4 export sonrası GeoTIFF dosyalarının şu an manuel olarak indirilip uygun klasörlere yerleştirilmesi
- aynı tarihte birden fazla Landsat sahnesi geldiğinde bunların birleştirilmesi / yönetimi
- Step5 zaman serisi ön işleme akışının kararlı hale getirilmesi
- çoklu Landsat sahnelerinde aynı tarihli görüntülerin yönetimi
- QA dosyası eşleştirme ve maskeleme akışının güçlendirilmesi
- interpolasyon ve eksik veri yönetiminin doğrulanması
- örnek çıktıların ve görselleştirmelerin düzenlenmesi

## Planlanan Çalışmalar

Henüz tamamlanmamış veya sonraki aşamalarda geliştirilecek kısımlar:

- Python tarafında daha gelişmiş ön işleme adımları
- daha güçlü zaman serisi analizi
- 2B çıktıların görsel kalite iyileştirmeleri
- 3B arazi + termal harita görselleştirmesi
- FIRMS yangın kayıtları ile etiketli veri seti oluşturma
- Jetson / YOLOv7-Tiny entegrasyonu
- mobil erişim ve API katmanı

## Veri Kaynakları

### MODIS
- Veri kümesi: `MODIS/061/MOD11A1`
- İçerik: günlük kara yüzey sıcaklığı verisi
- Çözünürlük: yaklaşık 1 km
- Kullanım amacı: geniş alanlı referans termal katman üretimi

### Landsat 8
- Veri kümesi: `LANDSAT/LC08/C02/T1_L2`
- İçerik: Surface Temperature (`ST_B10`) ve kalite bandı (`QA_PIXEL`)
- Çözünürlük: yaklaşık 30 m
- Kullanım amacı: yüksek çözünürlüklü termal katman ve zaman serisi üretimi

## Metodoloji

Projede genel iş akışı şu şekildedir:

1. Google Earth Engine üzerinden çalışma bölgesi tanımlanır.
2. MODIS verisi ile Doğu Akdeniz bölgesi için 5 yıllık yaz dönemi ortalama sıcaklık görüntüsü üretilir.
3. Landsat verisi ile aynı bölge için yüksek çözünürlüklü yüzey sıcaklığı görüntüleri hazırlanır.
4. MODIS referans çıktıları ve Landsat zaman serisi LST / QA rasterları GeoTIFF olarak export edilir.
5. Python tarafında raster veriler okunur, sıcaklık dönüşümü ve QA tabanlı maskeleme uygulanacak şekilde yapı kurulmuştur. Zaman serisi ve anomali üretimi aşaması geliştirilmekte ve test edilmektedir.
6. Ortalama LST ve anomali rasterları üretilir.
7. Sonraki aşamalarda 3B görselleştirme, yangın riski analizi ve karar destek katmanları eklenecektir.

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

data/
outputs/
logs/
````

## Step Açıklamaları

### Step 1

GEE bağlantısını başlatır, çalışma bölgelerini tanımlar ve temel MODIS LST sorgusunu gerçekleştirir.

### Step 2

Doğu Akdeniz bölgesi için MODIS verisinden 5 yıllık yaz dönemi ortalama sıcaklık görüntüsü üretir.

### Step 3

Aynı bölge için Landsat tabanlı yüksek çözünürlüklü yüzey sıcaklığı görüntülerini ve zaman serisi koleksiyonunu hazırlar.

### Step 4

MODIS referans çıktılarının ve Landsat zaman serisi LST / QA rasterlarının GeoTIFF olarak export edilmesini sağlar.

### Step 5

Python tarafında raster okuma, sıcaklık dönüşümü, QA tabanlı bulut maskeleme, zaman serisi oluşturma, interpolasyon ve anomali üretimi için ayrılmış adımdır.
Bu adım kod seviyesinde oluşturulmuş olup, GEE export süresi ve veri hazırlama süreçleri nedeniyle hâlen test ve doğrulama aşamasındadır.

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
```

Google Earth Engine erişimi için kimlik doğrulaması yapın:

```bash
earthengine authenticate
```

```bash
Step4 sonrasında export edilen GeoTIFF dosyaları şu an Google Drive üzerinden manuel olarak indirilmekte ve Step5 için ilgili veri klasörlerine yerleştirilmektedir.
```

## Çalıştırma Sırası

```bash
python step1_fetch_modis.py
python step2_modis_5year_mean.py
python step3_landsat_lst.py
python step4_export_geotiff.py
python step5_preprocess_timeseries.py
```

## Örnek Çıktılar

Bu bölüm henüz geliştirme aşamasındadır. İlerleyen güncellemelerde MODIS ortalama sıcaklık çıktısı, Landsat zaman serisi örnekleri ve Step5 ön işleme çıktıları testler tamamlandıktan sonra görsel olarak eklenecektir.

## Not

Bu repo şu anda tamamlanmış bir **3B termal dijital ikiz sistemi** değildir. Mevcut haliyle, uydu tabanlı termal veri işleme pipeline’ının modüler biçimde geliştirildiği bir prototip çalışmadır. Özellikle Step5 ve sonraki aşamalar hâlen aktif geliştirme altındadır. Projenin sonraki aşamalarında 3B görselleştirme, risk analizi, veri birleştirme ve Jetson tabanlı model entegrasyonu eklenecektir.
