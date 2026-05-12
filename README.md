# Uydu Tabanlı Termal Veri İşleme Prototipi

Bu proje, Google Earth Engine (GEE) üzerinden alınan **MODIS** ve **Landsat** yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesi için **termal çevre temsili** oluşturan modüler bir prototip sistemdir. Proje; veri sorgulama, sıcaklık işleme, GeoTIFF export ve Python ile yerel raster ön işleme adımlarını içeren bir altyapı kurmayı hedeflemektedir.

Bu repo şu anda tamamlanmış bir **3B termal dijital ikiz** veya **yangın riski tahmin sistemi** değildir. Mevcut haliyle, bu hedeflere ilerleyen bir **termal veri işleme ve ön analiz prototipi** olarak değerlendirilmelidir.

---

# Amaç

Projede amaç, uydu tabanlı yüzey sıcaklığı verilerini kullanarak Doğu Akdeniz bölgesine ait:

* geniş alanlı termal referans katmanlar üretmek,
* yüksek çözünürlüklü Landsat sıcaklık verilerini işlemek,
* zaman serisi tabanlı ön işleme akışı kurmak,
* anomali üretimine temel oluşturacak raster çıktılar elde etmek

ve daha sonraki aşamalarda bunları 3B görselleştirme, risk analizi ve karar destek katmanlarına bağlayabilecek bir temel hazırlamaktır.

---

# Proje Kapsamı

Projede şu adımlar yer almaktadır:

* çalışma bölgesinin tanımlanması
* MODIS LST verisinin sorgulanması
* MODIS için 5 yıllık yaz dönemi baseline üretimi
* baseline için ortalama ve standart sapma hesaplanması
* Landsat yüksek çözünürlüklü LST verisinin hazırlanması
* Landsat günlük composite zaman serisi oluşturulması
* GeoTIFF export
* QA tabanlı bulut maskeleme
* Python ile zaman serisi ön işleme
* baseline ve anomaly raster üretimi
* ileride 2B/3B görselleştirme ve risk analizi

---

# Mevcut Durum

Şu anda tamamlanan veya büyük ölçüde kurulan kısımlar şunlardır:

* GEE bağlantısı ve doğrulama
* çalışma bölgelerinin tanımlanması
* proje yapısının modüler hale getirilmesi (`core/` yapısı)
* ortak ayarların `config.py` altında toplanması
* MODIS veri sorgulama
* MODIS için 5 yıllık yaz dönemi **ortalama** üretimi
* MODIS için 5 yıllık yaz dönemi **standart sapma** üretimi
* Landsat veri sorgulama
* Landsat zaman serisi koleksiyonunun hazırlanması
* aynı tarihe ait çoklu Landsat sahnelerinin **daily median composite** mantığıyla birleştirilmesi
* Step4’te MODIS ve Landsat exportlarının aç/kapa mantığıyla kontrol edilmesi
* Step4’e kadar çalışan bir `main.py` yapısının kurulması
* Step5’in küçük örnek veri ile çalıştırılması
* baseline mean raster, baseline std raster, current period median raster ve z-score anomaly raster çıktılarının üretilmesi
* NetCDF zaman serisi çıktılarının oluşturulması
* anomaly üretiminde tek sahne yaklaşımından temporal window + median composite yaklaşımına geçilmesi

---

# Şu Anda Gözlenen Sınırlılıklar

Proje çalışıyor olsa da şu anda bazı önemli teknik sınırlılıklar bulunmaktadır:

* Step4 sonrası veri akışı hâlâ tam otomatik değildir; Drive export sonrası dosyalar manuel olarak indirilip Step5 klasörlerine yerleştirilmektedir.
* `ee.Image.getDownloadURL()` ile doğrudan indirme denenmiştir ancak büyük boyutlu rasterlarda yeterli olmadığı görülmüştür.
* Google Drive export sürecinin otomatikleştirilmesi henüz tamamlanmamıştır.
* Step5 şu ana kadar sınırlı sayıda sahne ile test edilmiştir.
* Bu nedenle anomaly ve diğer raster çıktılarında veri bulunmayan beyaz alanlar oluşabilmektedir.
* Anomaly üretimi artık tek sahne yaklaşımı yerine belirli bir zaman penceresi içerisindeki Landsat sahnelerinin median composite çıktısı üzerinden yapılmaktadır.
* Ancak küçük test pencerelerinde veya sınırlı sahne sayısında hâlâ veri boşlukları oluşabilmektedir.
* Bu boşlukların temel nedeni yetersiz zamansal kapsama ve Landsat sahnelerinin doğal mekânsal kapsama farklılıklarıdır.
* MODIS (~1 km) ve Landsat (~30 m) çözünürlük farkı nedeniyle anomaly üretimi hâlâ geliştirme aşamasındadır.
* Step5 çıktıları henüz büyük ölçekli veri ile tam doğrulanmış değildir.
* Büyük raster işlemlerinde bellek kullanımı yüksek olabilmektedir ve WSL ortamında ek optimizasyon gerektirmektedir.

---

# Geliştirme Aşamasında Olan Kısımlar

Şu anda aktif olarak geliştirilen / iyileştirilen alanlar:

* anomaly üretiminde temporal window yaklaşımının iyileştirilmesi
* daha geniş zaman pencereleri ile veri boşluklarının azaltılması
* current period composite üretiminin daha kararlı hale getirilmesi
* Step5 çıktılarının daha büyük veri kümeleri ile doğrulanması
* çıktıların README’ye eklenebilecek düzeyde iyileştirilmesi
* Step4 sonrası veri geçiş sürecinin daha kontrollü hale getirilmesi
* Google Drive export sürecinin geemap task polling benzeri yöntemlerle otomatikleştirilmesi
* config bağımlılığının zamanla azaltılması
* step fonksiyonlarına parametre geçişinin artırılması
* raster işleme sırasında bellek kullanımının optimize edilmesi
* chunked/windowed raster processing yaklaşımına geçilmesi

---

# Planlanan Çalışmalar

Henüz tamamlanmamış veya ileride geliştirilecek başlıklar:

* 2B çıktıların daha güçlü görselleştirilmesi
* 3B görselleştirme katmanı
* gelişmiş anomali / risk analizi
* karar destek yapısı
* yangın kayıtları ile veri birleştirme
* Jetson / YOLO entegrasyonu
* daha kararlı online/offline pipeline ayrımı
* tam otomatik export + download akışı
* Step5 sonrası aşamaların yapılandırılması

---

# Veri Kaynakları

## MODIS

* Veri kümesi: `MODIS/061/MOD11A1`
* İçerik: günlük kara yüzey sıcaklığı verisi
* Çözünürlük: yaklaşık 1 km
* Kullanım amacı: geniş alanlı termal baseline üretimi

## Landsat 8

* Veri kümesi: `LANDSAT/LC08/C02/T1_L2`
* İçerik: Surface Temperature (`ST_B10`) ve kalite bandı (`QA_PIXEL`)
* Çözünürlük: yaklaşık 30 m
* Kullanım amacı: yüksek çözünürlüklü termal katman ve zaman serisi işleme

---

# MODIS Baseline Seçiminin Gerekçesi

Projede MODIS verisi, Doğu Akdeniz için düşük çözünürlüklü ama zamansal olarak daha düzenli bir referans termal davranış üretmek amacıyla kullanılmaktadır.

Bu aşamada MODIS tarafında tam zaman serisi yerine 5 yıllık yaz dönemi ortalaması ve standart sapması alınmıştır. Bunun başlıca gerekçeleri şunlardır:

* MODIS, Landsat’a göre daha düzenli gözlem sıklığı sağlar.
* 5 yıllık pencere, tek bir yıla göre daha kararlı bir baseline üretir.
* Yaz aylarına odaklanılması, yüksek sıcaklık davranışını inceleme amacıyla uyumludur.
* Ortalama tek başına yeterli olmadığı için, değişkenliği tanımlamak üzere standart sapma da eklenmiştir.
* Bu yapı, ileride z-score veya normalize anomali hesapları için temel sağlayacaktır.

Bu nedenle MODIS çıktısı, Landsat’ın yerine geçen bir zaman serisi değil; daha çok Landsat analizini destekleyen referans termal katman olarak kullanılmaktadır.

---

# Metodoloji

Projede genel akış şu şekildedir:

1. Çalışma bölgesi GEE üzerinde tanımlanır.
2. MODIS verisi kullanılarak 5 yıllık yaz dönemi baseline katmanları hazırlanır.
3. Bu baseline için ortalama ve standart sapma hesaplanır.
4. Landsat verisi aynı bölge için filtrelenir.
5. Aynı tarihe ait çoklu Landsat sahneleri daily median composite ile tek çıktıya indirilir.
6. Current period için belirli bir temporal window tanımlanır.
7. Bu pencereye düşen Landsat sahneleri median composite ile tek current state rasterına dönüştürülür.
8. MODIS ve Landsat rasterları GeoTIFF olarak export edilir.
9. QA verisi kullanılarak bulut maskeleme yapılır.
10. Python tarafında zaman serisi kurulup baseline mean/std rasterları, current period median rasterı ve z-score anomaly rasterı üretilir.
11. Çıktılar görsel ve sayısal olarak kontrol edilir.

---

# Proje Yapısı

```text
core/
├── config.py
├── gee_utils.py
├── io_utils.py
└── regions.py

step1_fetch_modis.py
step2_modis_5year_mean.py
step3_landsat_lst.py
step4_export_geotiff.py
step5_preprocess_timeseries.py
main.py
```

---

# Step Açıklamaları

## Step 1

GEE bağlantısını başlatır, çalışma bölgelerini tanımlar ve temel MODIS sorgusunu gerçekleştirir.

## Step 2

MODIS verisini kullanarak 5 yıllık yaz dönemi baseline katmanlarını üretir. Bu aşamada ortalama ve standart sapma hesaplanır.

## Step 3

Landsat zaman serisi koleksiyonunu hazırlar. Aynı tarihe ait çoklu görüntüler daily median composite ile tek çıktıya indirgenir. Ayrıca current period için temporal window tabanlı median composite oluşturulur.

## Step 4

Online export katmanıdır. MODIS ve Landsat rasterlarını GeoTIFF olarak export eder. Gerekirse MODIS ve Landsat exportları ayrı ayrı açılıp kapatılabilir.

Şu anda export sonrası veri indirme süreci tam otomatik değildir ve manuel Drive indirme adımı gerektirmektedir.

## Step 5

Offline raster işleme katmanıdır. İndirilen GeoTIFF dosyaları okunur, QA tabanlı maskeleme uygulanır ve aşağıdaki çıktılar üretilir:

* baseline mean raster
* baseline standard deviation raster
* current period median raster
* z-score anomaly raster
* interpolated NetCDF zaman serisi çıktısı

## main.py

Şu anda Step4’e kadar olan akışı organize biçimde çalıştırmak için kullanılmaktadır. Step5 ve sonrası manuel olarak yürütülmektedir.

---

# Step5 Çıktıları

Step5 sonunda aşağıdaki raster ve zaman serisi çıktıları üretilmektedir:

* `baseline_lst_mean_celsius.tif`
* `baseline_lst_std_celsius.tif`
* `current_period_median_celsius.tif`
* `anomaly_zscore.tif`
* `baseline_timeseries_interpolated.nc`

---

# Kurulum

```bash
git clone https://github.com/emrehann17/satellite-thermal-digital-twin.git
cd satellite-thermal-digital-twin
```

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

```bash
pip install -r requirements.txt
```

```bash
earthengine authenticate
```

---

# Çalıştırma Sırası

## Online kısım

```bash
python main.py
```

veya:

```bash
python step1_fetch_modis.py
python step2_modis_5year_mean.py
python step3_landsat_lst.py
python step4_export_geotiff.py
```

## Manuel geçiş

Step4 sonrası export edilen GeoTIFF dosyaları şu anda Google Drive üzerinden manuel olarak indirilmektedir.

İndirilen rasterlar daha sonra Step5 veri klasörlerine yerleştirilerek offline işleme aşamasına aktarılır.

## Offline kısım

```bash
python step5_preprocess_timeseries.py
```

---

# Örnek Çıktılar

## Baseline Mean Raster

Bu raster, baseline dönemi boyunca hesaplanan ortalama yüzey sıcaklığını temsil etmektedir.

![Baseline Mean](docs/images/baseline_lst_mean_celsius.png)

## Z-Score Anomaly Raster

Bu raster, current period median sıcaklık değerlerinin baseline mean ve standard deviation kullanılarak hesaplanan z-score anomaly çıktısını göstermektedir.

![Anomaly](docs/images/anomaly_zscore.png)

---

# Çıktılar Hakkında Önemli Notlar

Bu görseller, projenin mevcut geliştirme aşamasını temsil eden erken prototip çıktılarıdır ve aşağıdaki sınırlamaları içermektedir:

* Çıktılar, işlem süresini kısaltmak amacıyla sınırlı sayıda Landsat sahnesi kullanılarak üretilmiştir.
* Bu nedenle anomaly raster üzerinde veri bulunmayan alanlar oluşabilmektedir.
* Bu boşluklar sistemsel bir hatadan değil, yetersiz zamansal kapsama probleminden kaynaklanmaktadır.
* Current thermal state, belirli bir zaman penceresine düşen Landsat sahnelerinin median composite çıktısı olarak tanımlanmaktadır.
* Bu yaklaşım, tek sahne kullanımına göre daha kararlı ve daha geniş kapsamalı anomaly üretimi sağlamayı hedeflemektedir.
* Daha geniş temporal window ve daha fazla sahne kullanımı ile anomaly kapsamasının iyileştirilmesi hedeflenmektedir.

---

# Planlanan İyileştirmeler

* Daha fazla Landsat sahnesi kullanılarak veri boşluklarının azaltılması
* Temporal window ve composite stratejisinin daha güçlü anomaly kapsaması sağlayacak şekilde geliştirilmesi
* MODIS baseline ile Landsat anomaly ilişkisinin daha sağlam kurulması
* Büyük rasterlar için daha verimli bellek yönetimi
* Export sonrası veri indirme sürecinin otomatikleştirilmesi
* Nihai, doğrulanmış ve daha temiz görsel çıktılarla bu bölümün güncellenmesi

---

# Not

Bu repo şu anda tamamlanmış bir 3B termal dijital ikiz sistemi değildir. Mevcut haliyle, MODIS ve Landsat tabanlı termal veri işleme ve ön analiz akışını kurmaya odaklanan bir prototip çalışmadır.

Şu anki en önemli açık konular:

* anomaly’nin tüm bölgeyi temsil edecek şekilde güçlendirilmesi
* daha fazla Landsat sahnesi ile Step5’in yeniden test edilmesi
* veri boşluklarının azaltılması
* Step4 sonrası veri akışının iyileştirilmesi
* export + download sürecinin daha otomatik hale getirilmesi

Bu eksikler giderildikçe proje daha güçlü 2B/3B termal temsil ve risk analizi katmanlarına doğru genişletilecektir.
