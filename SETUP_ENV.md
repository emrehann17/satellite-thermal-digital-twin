# Google Drive Configuration Setup

## 🔒 Güvenlik Güncellemesi

Google Drive kimlik bilgileri artık `.env` dosyasında güvenli bir şekilde saklanıyor.

## 📋 Kurulum Adımları

### 1. `.env` Dosyası Oluşturma

Proje kök dizininde `.env` dosyası oluşturun:

```bash
cp .env.example .env
```

### 2. Kimlik Bilgilerini Ekleme

`.env` dosyasını açın ve kendi Google Drive bilgilerinizi ekleyin:

```env
GOOGLE_DRIVE_EXPORT_FOLDER_URL=https://drive.google.com/drive/u/0/folders/YOUR_FOLDER_ID
GOOGLE_DRIVE_EXPORT_FOLDER_ID=YOUR_FOLDER_ID
```

### 3. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

## ⚠️ Önemli Notlar

- `.env` dosyası **asla** Git'e commit edilmemelidir
- `.gitignore` dosyasında `.env` zaten eklidir
- `.env.example` dosyası template olarak kullanılabilir (hassas bilgi içermez)

## 🔍 Doğrulama

Yapılandırmanızı test etmek için Python'da:

```python
from core.config import GOOGLE_DRIVE_EXPORT_FOLDER_ID
print(GOOGLE_DRIVE_EXPORT_FOLDER_ID)
```

Eğer `None` dönüyorsa, `.env` dosyanızın proje kök dizininde olduğundan emin olun.
