# Offline Canlı Çeviri Altyazısı

Bilgisayarda o an ne çalıyorsa (YouTube, Netflix, oyun, video görüşmesi...) sistem
sesini dinler, konuşmayı gerçek zamanlıya yakın şekilde metne çevirir (Whisper),
hedef dile çevirir (varsayılan: NLLB-200, GPU üzerinde CTranslate2 ile) ve
ekranın altına, her zaman üstte duran yarı saydam bir altyazı çubuğu olarak
yansıtır. Konuşma bittikten 1–3 saniye sonra altyazı ekranda olur.

Tamamen offline çalışır: internet yalnızca ilk kurulumda modelleri indirmek için
gerekir, ses hiçbir sunucuya gönderilmez.

## Gereksinimler

- Windows 10 / 11
- Python 3.13
- NVIDIA GPU (CUDA). Geliştirme RTX 4060 (8 GB VRAM) üzerinde yapıldı. GPU yoksa
  CPU'da da çalışır ama belirgin şekilde yavaştır (bkz. Kurulum, 3. adım).
- Yaklaşık 9 GB boş disk (sanal ortam + modeller + Hugging Face önbelleği)

## Kurulum

**1. Projeyi indir ve sanal ortamı kur**

```
git clone https://github.com/alisarfurkan/-Offline-canli-ceviri-altyazi-uygulamasi.git
cd -Offline-canli-ceviri-altyazi-uygulamasi
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. NLLB-200 çeviri modelini hazırla (tek seferlik, ~2.4 GB indirme)**

```
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir models\nllb-ct2 --quantization float16
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M').save_pretrained('models/nllb-ct2/tokenizer')"
```

Bu adımı atlarsan uygulama açılır ama çeviri yapamaz; o durumda Ayarlar'dan
çeviri motorunu "Argos Translate (CPU, basit)" olarak değiştir — kurulum
gerektirmez, kalitesi daha düşüktür.

**3. GPU'nun görüldüğünü kontrol et**

```
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`1` (veya daha fazla) dönmeli. `0` dönüyorsa GPU kullanılamıyor demektir:
`transcriber.py` içindeki `Transcriber(device="cuda", compute_type="float16")`
varsayılanlarını `device="cpu", compute_type="int8"` yap ve Ayarlar'dan çeviri
motorunu Argos'a al.

Whisper modeli (`large-v3-turbo`, ~1.6 GB) uygulama ilk açıldığında otomatik
indirilir; bu ilk açılış birkaç dakika sürebilir.

## Çalıştırma

### Masaüstü kısayolu (önerilen)

Proje klasöründe PowerShell açıp şunu bir kez çalıştır:

```powershell
$proj = (Get-Location).Path
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Canlı Altyazı.lnk")
$lnk.TargetPath = "$proj\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "`"$proj\altyazi.pyw`""
$lnk.WorkingDirectory = $proj
$lnk.IconLocation = "$proj\altyazi.ico,0"
$lnk.Save()
```

Masaüstünde **Canlı Altyazı** kısayolu oluşur. Çift tıklayınca uygulama konsol
penceresi açılmadan başlar ve kendiliğinden dinlemeye geçer.

### Komut satırından

```
.venv\Scripts\Activate.ps1
python main.py
```

### Kullanım

Açılışta altyazı çubuğunda önce "Modeller yükleniyor...", ardından
"Dinleniyor..." yazar; video konuştukça orijinal metin (üstte, küçük) ve çevirisi
(altta, büyük) görünür.

Sistem tepsisindeki mavi **CC** simgesine (saatin yanında, gizli oklar altında
olabilir) sağ tıklayarak:

- **Durdur / Başlat**: dinlemeyi kapat/aç.
- **Konumu Kilitle**: altyazı çubuğunu tıklamaya şeffaf yapar, alttaki pencereye
  tıklamalar geçer. Kilitli değilken çubuğu sürükleyerek taşıyabilirsin.
- **Ayarlar...**: kaynak dil (veya "Otomatik algıla"), hedef dil, dinlenecek ses
  çıkış cihazı, Whisper model boyutu ve çeviri motoru (NLLB-200 / Argos).
  Ayarları kaydettikten sonra dinlemeyi yeniden başlatmak için **Başlat**'a bas.
- **Çıkış**: uygulamayı kapatır.

### Sorun giderme

Kısayoldan açıldığında konsol olmadığı için hatalar proje klasöründeki
`altyazi.log` dosyasına yazılır. Altyazı çubuğunda "Model yükleme hatası" gibi
bir mesaj görürsen ayrıntı oradadır.

Altyazı gelmiyorsa önce videonun gerçekten ses çıkardığını ve Ayarlar'daki ses
cihazının, sesin çıktığı cihazla aynı olduğunu kontrol et.

## Bilinen sınırlar

- **DirectX exclusive fullscreen oyunlar**: altyazı çubuğu görünmez (Windows'un
  compositor'ı devre dışı kalır). Discord/OBS gibi araçlarda da aynı kısıt var;
  oyunu "sınırsız pencere" (borderless windowed) moduna al.
- **Çeviri kalitesi**: NLLB-200 (600M) bulut servisleri (DeepL/GPT) kadar akıcı
  olmayabilir ama Argos'a göre belirgin şekilde daha doğaldır. `translator.py`'deki
  `Translator` arayüzü sayesinde ileride yerel bir LLM (Ollama) tabanlı bir motor
  da eklenebilir.
- **NLLB dil kapsamı**: `translator.py` içindeki `ISO_TO_FLORES` sözlüğünde
  olmayan bir dil kodu hata verir; yeni bir dil eklemek için o dilin FLORES-200
  kodunu sözlüğe eklemek yeterli.
- **Whisper model boyutu**: `large-v3-turbo` 8 GB VRAM için hız/doğruluk dengesi
  en iyi seçenektir; VRAM yetmezse Ayarlar'dan `medium` veya `small` seç.

## Proje yapısı

| Dosya | Görev |
|---|---|
| `altyazi.pyw` | Konsolsuz başlatıcı (masaüstü kısayolu bunu çalıştırır) |
| `altyazi.ico` | Kısayol simgesi |
| `main.py` | Uygulamanın giriş noktası |
| `audio_capture.py` | WASAPI loopback ile sistem sesi yakalama |
| `vad_segmenter.py` | Silero VAD ile akan sesi cümlelere bölme |
| `transcriber.py` | faster-whisper (GPU) ile konuşma → metin |
| `translator.py` | Offline çeviri: NLLB-200 (CTranslate2/GPU, varsayılan) + Argos Translate (CPU) |
| `gpu_env.py` | ctranslate2'nin CUDA DLL'lerini (cuBLAS/cuDNN) bulması için ortam ayarı |
| `pipeline.py` | Yukarıdakileri thread/queue zinciriyle bağlar |
| `overlay_ui.py` | Her zaman üstte, yarı saydam altyazı çubuğu (PySide6) |
| `tray_app.py` / `settings_dialog.py` | Sistem tepsisi ve ayarlar penceresi |
| `config.py` | Ayarların `config.yaml`'a kaydedilmesi |
