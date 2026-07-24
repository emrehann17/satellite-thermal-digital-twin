# Ortam ve Kimlik Doğrulama Kurulumu

Çoğu kullanıcı için gereken tek şey Python bağımlılıkları ve Earth Engine kimlik doğrulamasıdır. Google Drive kurulumu isteğe bağlıdır ve yalnızca legacy Kozan reprodüksiyon iş akışı için geçerlidir. Hızlı yönlendirme ve temel kurulum komutları için [README.md](README.md) dosyasına bakın; bu belge ayrıntılı ortam, kimlik doğrulama ve isteğe bağlı legacy yapılandırmanın sahibidir.

## 1. Python ortamı

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows: WSL veya bir POSIX kabuk (Git Bash) önerilir; komutlar yukarıdakiyle aynıdır. Yerel PowerShell kullanılıyorsa yalnızca sanal ortam etkinleştirme farklıdır (`.venv\Scripts\Activate.ps1`).

`requirements-lock.txt`, isteğe bağlı bir tekrarüretilebilirlik (reproducibility) anlık görüntüsüdür; birebir aynı sürümleri kurmak isterseniz `requirements.txt` yerine bunu kullanabilirsiniz.

`xgboost` yalnızca `--model xgboost` açıkça seçildiğinde gereklidir; varsayılan `random_forest` / `hist_gradient_boosting` akışları onu gerektirmez (bkz. `requirements.txt`).

## 2. Earth Engine kimlik doğrulaması

Export/GEE işlemleri için bir kez kimlik doğrulaması yapın:

```bash
earthengine authenticate
```

Kurulumu doğrulamak için (kendi Earth Engine etkin Cloud projenizle):

```bash
python -c "import ee; ee.Initialize(project='<your-ee-project>'); print('EE ok')"
```

Kimlik doğrulaması yerel olarak başarılı olsa bile, ilgili Earth Engine erişimi ve Cloud proje izinleri ayrıca gerekebilir. Depo varsayılan proje adı `core/config.py` içindeki `GEE_PROJECT` ile ayarlanır.

## 3. Varsayılan deney-farkında akış

Normal (deney-farkında) kullanım için:

- Google Drive klasör yapılandırması **gerekmez**.
- Hiçbir Drive batch export/indirme aşaması kullanılmaz.
- Predictor export'u doğrudan/tiled yerel Earth Engine indirmesi ile yapılır (`scripts/run_predictors_only.py`, `export_image_direct_or_tiled`).
- Çıktılar `outputs/experiments/<experiment_id>/` altına yazılır.

Bu akış için 1. ve 2. bölümler yeterlidir; bu bölümde `.env` Drive değişkenleri kullanılmaz.

## 4. İsteğe bağlı legacy Kozan Drive yapılandırması

> **Yalnızca legacy Kozan reprodüksiyon iş akışı için gereklidir.** Yeni deneyler için önerilen yol değildir.

Legacy akış, tarihsel Google Drive tabanlı zinciri korur ve şu doğrulanmış komutla açıkça çağrılır:

```bash
python scripts/main.py legacy --experiment kozan_2023 --force
```

Bu akış iki tarihsel aşama içerir:

- **Step4:** GEE'den Google Drive'a batch export (`ee.batch.Export.image.toDrive`).
- **Step4B:** Drive klasöründen yerel indirme, doğrulama ve Step5 veri klasörlerine yerleştirme.

Yapılandırma. Depoda `.env.example` bulunmadığından, proje kök dizininde `.env` dosyasını elle oluşturun ve yalnızca kodun gerçekten okuduğu değişkenleri ekleyin (`core/config.py`). Aşağıdaki ikisinden **biri** yeterlidir; Step4B önce ID'yi dener, yoksa URL'ye düşer:

```env
# Yer tutucu değerler -- gerçek klasör kimliğinizle değiştirin.
GOOGLE_DRIVE_EXPORT_FOLDER_ID=YOUR_FOLDER_ID
GOOGLE_DRIVE_EXPORT_FOLDER_URL=https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

Gerçek kimlik/klasör tanımlayıcılarını depoya koymayın; örneklerde yer tutucu kullanın.

## 5. Doğrulama ve sorun giderme

```bash
# aktif Python yorumlayıcısı
which python

# temel bağımlılık import'u
python -c "import ee, geemap, rasterio, numpy, pandas, sklearn; print('deps ok')"

# Earth Engine kimlik doğrulaması
python -c "import ee; ee.Initialize(project='<your-ee-project>'); print('EE ok')"

# (isteğe bağlı) legacy Drive yapılandırması ayarlı mı
python -c "from core.config import GOOGLE_DRIVE_EXPORT_FOLDER_ID as f; print('drive set' if f else 'drive empty (normal akış için sorun değil)')"
```

Son kontrol boşsa (`drive empty`), bu normal deney-farkında akış için beklenen durumdur; yalnızca legacy Kozan akışını çalıştıracaksanız 4. bölümü uygulayın.

## 6. Güvenlik

- `.env` dosyasını **asla** commit etmeyin.
- Kimlik/credential dosyalarını **asla** commit etmeyin.
- `.env`, `.gitignore` tarafından yok sayılmaya devam etmelidir.
- Örneklerde her zaman yer tutucu değerler kullanın; gerçek klasör kimlikleri veya anahtarlar paylaşılmamalıdır.
