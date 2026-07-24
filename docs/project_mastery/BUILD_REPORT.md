# BUILD_REPORT.md — PDF Derleme ve Doğrulama Raporu

- **Repository commit:** `47452308c37a3e0e4915a9ab88be8bc8a2d5bf80`
- **Üretim zamanı:** 2026-07-23 (UTC)
- **Ana çıktı:** `docs/PROJECT_MASTERY_GUIDE.pdf` (147 sayfa, ~2.7 MB)
- **Kanonik kaynak:** `docs/project_mastery/PROJECT_MASTERY_GUIDE.md` (4444 satır)

## 1. Araç tespiti (ortamda mevcut olanlar)

Bu ortamda sistem-seviyesi PDF renderer'ları **yoktu**: WeasyPrint, wkhtmltopdf, pandoc, LaTeX, Chromium ve poppler (pdftoppm) **mevcut değil**. Mevcut olanlar:

- `matplotlib` (figürler için) — sistem + venv
- `pygments`, `jinja2` — venv
- DejaVu TTF fontları (`/usr/share/fonts/truetype/dejavu/`) — tam Türkçe glyph desteği

Saf-Python, sistem-bağımlılığı gerektirmeyen bir zincir kurmak için venv'e (sistem geneli DEĞİL, `requirements.txt` DEĞİŞTİRİLMEDEN) iki paket kuruldu:

- `fpdf2==2.8.7` — Markdown→PDF derleme (Unicode TTF, tablo, bookmark/TOC, sayfa no, görsel).
- `pymupdf==1.28.0` (fitz) — doğrulama için PDF→PNG render + metin çıkarımı (poppler gerektirmez).

> Not: Bu iki paket proje-yerel `venv/`'e kuruldu; `requirements.txt`/`requirements-lock.txt` **değiştirilmedi**, sistem geneli hiçbir paket kurulmadı.

## 2. Derleme yaklaşımı

1. **Markdown kanonik kaynaktır** (`PROJECT_MASTERY_GUIDE.md`), elle bakımı yapılır.
2. Özel bir Markdown-alt-kümesi ayrıştırıcısı (`build_project_mastery_pdf.py`) Markdown'ı doğrudan fpdf2 çizim çağrılarına dönüştürür: başlıklar (bookmark + TOC), paragraflar (`**kalın**` + `` `kod` `` inline, state-machine tokenizer), madde/numaralı listeler, pipe tabloları (başlık gölgeli + alternatif satır dolgusu), fenced code blokları (monospace, gri kutu, uzun-satır sarma), blockquote callout kutuları (renkli aksан çubuğu), görseller (ortalı + başlık), yatay çizgiler.
3. Kapak sayfası: başlık + zihinsel-model diyagramı + commit hash + zaman damgası.
4. Tıklanabilir içindekiler (`insert_toc_placeholder`) + PDF outline bookmark'ları (`start_section`), nokta-lider ve sayfa numaralarıyla.
5. Her sayfada footer: commit + zaman damgası + sayfa numarası.
6. Fontlar: DejaVu Sans (regular/bold; italic upright'a eşlenir — ortamda oblique dosyası yok), DejaVu Sans Mono (kod).

## 3. Derleme komutu

```bash
cd ~/satellite-thermal-digital-twin
source venv/bin/activate
# (yalnız ilk kez) pip install fpdf2 pymupdf      # venv-yerel; requirements DEĞİŞMEZ

# Figürleri (yeniden) üret — salt-okunur; yalnız outputs/ okur
python docs/project_mastery/figures/generate_figures.py

# PDF'i derle (Markdown -> docs/PROJECT_MASTERY_GUIDE.pdf)
python docs/project_mastery/build_project_mastery_pdf.py
```

## 4. Doğrulama (gerçekleştirilen kontroller)

| # | Kontrol | Sonuç |
|---|---|---|
| 1 | PDF üretildi ve açılıyor | ✓ 147 sayfa, PyMuPDF açtı |
| 2 | Kapak: başlık + diyagram + commit + zaman damgası | ✓ görsel doğrulandı (p1) |
| 3 | İçindekiler: hiyerarşi + nokta-lider + sayfa no + tıklanabilir link | ✓ görsel doğrulandı (p6-7) |
| 4 | Mimari diyagram sayfası | ✓ Şekil 0 (kapak) + 15 figür gömülü, okunur çözünürlük |
| 5 | Büyük tablo sayfası | ✓ p56 (Bölüm 13.4, 5-sütun transfer tablosu) taşma yok |
| 6 | Kod-yoğun sayfa | ✓ p140 (runbook), kod sarma çalışıyor |
| 7 | Orta bölüm | ✓ Bölüm 10-13 render doğrulandı |
| 8 | Appendix sayfası | ✓ p72 (Bölüm 19, fonksiyon imzaları) |
| 9 | Son sayfa | ✓ p147 (Bölüm 25 cheat sheets) |
| 10 | Türkçe glyph'ler (ş ğ ı İ ç ö ü Ş Ğ Ç) | ✓ hepsi PDF metninde mevcut ve doğru render |
| 11 | Metin kırpılması / öğe örtüşmesi | ✓ gözlemlenmedi |
| 12 | Tablo taşması (yatay) | ✓ sütun genişlikleri içeriğe göre ölçekli; taşma yok |
| 13 | Kod sarma | ✓ uzun satırlar kutu içinde sarılıyor |
| 14 | Görsel çözünürlüğü | ✓ figürler 200 DPI PNG |
| 15 | Boş sayfa | ✓ 0 boş sayfa (PyMuPDF taraması) |
| 16 | İç bağlantılar (TOC links) | ✓ `add_link(page=...)` ile sayfa hedefleri |
| 17 | Metin çıkarımı: tüm zorunlu bölüm başlıkları | ✓ Bölüm 0-25 hepsi PDF metninde |
| 18 | Kaynak başlık sayısı ↔ PDF varlığı | ✓ 27 H1 + 208 H2 = 235 başlık; PDF bookmark level0=27, level1=208; metinde 0 eksik |
| 19 | PDF outline bookmark'ları | ✓ 365 bookmark girdisi |
| 20 | Bilimsel çıktı/kod değiştirilmedi mi | ✓ `git status`: yalnız `docs/` eklendi; pipeline/config/outputs dokunulmadı |

**Bulunan ve düzeltilen kusur:** Kod-span içeren kalın metin (`**Formal \`all_valid\`**`) ilk sürümde literal `**` render ediyordu. Inline ayrıştırıcı state-machine'e çevrildi; yeniden derlenip görsel doğrulandı (p56).

## 5. Yeniden üretilebilirlik notu

- Figürler `outputs/` altındaki canonical JSON'lardan salt-okunur üretilir; hiçbir bilimsel dosya değiştirilmez.
- Markdown'ı düzenleyip `build_project_mastery_pdf.py`'yi yeniden çalıştırmak PDF'i deterministik olarak yeniden üretir (aynı commit'te aynı içerik).
- PDF metadata'sında başlık/yazar/creator gömülüdür.

## 6. Deliverable manifesti

```
docs/
├── PROJECT_MASTERY_GUIDE.pdf              # ana kullanıcı çıktısı (147 sayfa)
└── project_mastery/
    ├── PROJECT_MASTERY_GUIDE.md           # kanonik editable kaynak (4444 satır)
    ├── COVERAGE_REPORT.md                 # 130 modül + CLI + deney + feature kapsamı
    ├── SOURCE_INDEX.json                  # 24 bölüm→kaynak→çıktı izlenebilirlik girdisi
    ├── BUILD_REPORT.md                    # bu dosya
    ├── build_project_mastery_pdf.py       # Markdown→PDF derleyici (fpdf2)
    └── figures/
        ├── generate_figures.py            # figür üreteci (matplotlib, salt-okunur)
        └── fig00..fig15 (16 PNG)          # 12 zorunlu şema + 4 veri-güdümlü grafik
```
