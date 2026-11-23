# VIDEO DNA — Viral Video Tersine Mühendislik Sistemi

Amaç: Çok izlenen videoların **neden** işe yaradığını tahminle değil ölçümle bulmak ve bunu
kendi videolarında uygulanabilir, sayısal kurallara çevirmek.

## Strateji: 3 katman

| Katman | Ne yapar | Araç | Güvenilirlik |
|---|---|---|---|
| 1. Ölçüm | Sayısal metrikler | `measure.py` (FFmpeg, PySceneDetect, OpenCV, Whisper, Librosa, Tesseract) | Yüksek. Aynı videoda hep aynı sonucu verir |
| 2. Yorum | Ölçülemeyen psikoloji ve hikâye | `ai_rubrik.md` + `consensus.py` | Orta. Kapalı kategori, 3 bağımsız çalıştırma ve çoğunluk oyuyla tutarlı hâle getirilir |
| 3. Karşılaştırma | Gerçek kuralları ayıklar | `compare.py` (başarılı ve kontrol grubu, Cliff δ etki büyüklüğü) | Veri sayısı arttıkça yükselir |

**Temel ilke:** Bir özellik ancak başarılı videolarda kontrol grubundan belirgin farklıysa "kural" sayılır.
Herkesin yaptığı şey "niş standardı" olarak ayrılır. Dağınık olan şeyler elenir.

## Hangi madde nasıl çözüldü

### A. Doğrudan ölçülebilen (olcum)
| Madde | Metrik |
|---|---|
| Süre, format, en-boy | `temel.*` |
| Sahne sayısı ve süreleri, dakikada kesme | `sahne.*` |
| Sert ve yumuşak geçiş oranı | `sahne.sert_kesme_orani` (fade/dissolve ayrı tespit edilir) |
| Renk paleti (HEX), parlaklık, doygunluk, kontrast | `renk.*` |
| LUFS, LRA, true peak | `ses.*` (ffmpeg ebur128) |
| BPM, vuruşlar, sessizlikler, ani ses kesilmeleri | `ses.*` |
| WPM, hece/sn, duraksamalar, hız eğrisi, cümle uzunluğu | `konusma.*` (Whisper, kelime düzeyi zaman damgası) |
| Tetikleyici kelimeler, soru, doğrudan hitap, sayı kullanımı | `konusma.*` |

### B. Dolaylı ölçülen (yaklasik, sayısal ama vekil metrik)
| Madde | Nasıl ölçüldü |
|---|---|
| Kamera hareketi (zoom/pan/tilt), hız, ease-in/out | Optik akış (Farneback). Bilinen %24/sn zoom ve %22/sn pan ile doğrulandı |
| Sahne içi görsel olaylar (grafik/nesne girişi) | Kareler arası fark tepeleri |
| Altyazı: süre, font boyu, ekran alanı, konum, güvenli alan, konuşma gecikmesi | OCR (Tesseract) + Whisper eşleştirme |
| Ses-görüntü senkronu | Kesme–vuruş, kesme–ses olayı, kesme–kelime eşleşmesi. **Rastlantı beklentisi de raporlanır** |
| Pattern interrupt | Görsel olay ile güçlü ses olayının 0,2 sn içinde çakışması |
| Hook gücü | İlk 1/3/5 sn olay sayısı, ilk kesme, ilk kelime, ilk yazı, hareket ve ses yoğunluğu oranı |
| Loop | İlk ve son kare benzerliği, ilk ve son cümle ortaklığı, son ses seviyesi |
| Ses tonu ifadesi | Perde değişkenliği (yarım ton) |

### C. Ölçülemeyen maddeler ve çözümleri
| Madde | Neden ölçülemez | Çözüm |
|---|---|---|
| **Retention (izleyici tutma)** | Videodan hesaplanamaz, platform verisi gerekir | 1) Vekil olarak uyaran yoğunluğu ve 3 sn'den uzun monoton bölümler ölçülür. 2) Kendi videolarının Analytics verisi `meta.csv` → `ort_izlenme_yuzdesi` sütununa girilir, `compare.py` vekil metriklerin gerçek retention ile korelasyonunu hesaplar. Böylece vekil doğrulanmış olur |
| **"Neden izlendi" / başarı** | İzlenme algoritma, zamanlama ve kanal büyüklüğüne bağlı | Kontrol grubu + normalize performans (izlenme ÷ kanalın ortalama izlenmesi). Sadece ayırt eden özellik kural olur |
| **Hook türü, merak boşluğu, vaat** | Anlamsal, yoruma dayalı | Kapalı kategoriler + zorunlu zaman kodu kanıtı + 3 bağımsız AI çalıştırması + çoğunluk oyu. Uyum 2/3'ün altındaysa "BELIRSIZ" sayılır ve kurala girmez |
| **Hikâye iskeleti, merak döngüleri** | Anlamsal | Sabit blok etiketleri (hook/problem/bilgi/gerilim/dönüş/ödül/cta/kapanış) ve zaman aralıkları |
| **Duygusal tetikleyiciler** | Öznel | 8 duygu için 0-3 sabit ölçek, medyan ile birleştirme |
| **Font ailesi** | OCR font adı vermez | AI "tahmin" etiketiyle yazar. Boyut ve kalınlık zaten ölçülüyor |
| **Tek seferlik efekt mi, stil kuralı mı?** | Tek videoda ayırt edilemez | Çoklu video karşılaştırması: başarılı videoların çoğunda tekrarlanan şey kuraldır |
| **Kopyalama / telif** | Teknik değil | Sistem görüntü ve ses kopyalamaz, yalnızca parametre (süre, oran, hız) çıkarır |

## Kurulum
```bash
pip install -r requirements.txt
# + ffmpeg ve tesseract (Türkçe dil paketiyle: tesseract-ocr-tur)
#   Windows: winget install ffmpeg ; UB-Mannheim tesseract kurulumu (Turkish seçili)
#   macOS:   brew install ffmpeg tesseract tesseract-lang
```

## Kullanım (adım adım)
```bash
# 1) Video listesi: linkler veya dosyalar, her satıra bir tane
#    (başarılı ve kontrol videolarını birlikte koy)
python measure.py linkler.txt -o cikti           # veya: python measure.py video.mp4 -o cikti
#    -> cikti/<video>/measure.json, contact_sheet.jpg, transcript.txt

# 2) Her video için ai_rubrik.md'deki promptu 3 kez bağımsız çalıştır
#    (measure.json + contact_sheet.jpg + transcript.txt ekleyerek),
#    çıktıları cikti/<video>/ai_1.json, ai_2.json, ai_3.json olarak kaydet
python consensus.py cikti/<video>

# 3) meta.csv doldur (meta_ornek.csv'ye bak) ve karşılaştır
python compare.py cikti --meta meta.csv
#    -> cikti/rapor.md, stil_profili.json, uretim_promptu.md
```

## Veri önerisi
- En az **10 başarılı + 10 kontrol** videosu. Aynı niş, benzer süre, mümkünse aynı kanallardan.
- Kontrol grubu = aynı kanalların **düşük performanslı** videoları. Bu, kanal büyüklüğü etkisini ortadan kaldırır.
- Kendi videoların için Analytics'ten ortalama izlenme yüzdesini mutlaka ekle.

## Bilinen sınırlar
- OCR: stilize, hareketli ya da gölgeli yazılarda kaçırma olabilir (`yaklasik`).
- BPM: konuşma yoğun videolarda yanılabilir.
- Optik akış: tam ekran hareketli grafikleri kamera hareketi sanabilir. Bu yüzden ayrı bir `nesne_hareketi` etiketi var.
- Link indirme platform engeline takılabilir. O durumda videoyu indirip dosya yolu ver.
