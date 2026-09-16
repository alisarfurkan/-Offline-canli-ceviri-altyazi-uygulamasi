# Offline Canlı Çeviri Altyazısı

Bilgisayarda o an ne çalıyorsa (YouTube, Netflix, oyun, video görüşmesi...) sistem
sesini dinler, konuşmayı gerçek zamanlıya yakın şekilde metne çevirir (Whisper),
hedef dile çevirir (varsayılan: NLLB-200, GPU üzerinde CTranslate2 ile) ve
ekranın altına, her zaman üstte duran yarı saydam bir altyazı çubuğu olarak
yansıtır. Tamamen offline çalışır (ilk çalıştırmada model indirmeleri hariç),
gizlilik ve ücret derdi yoktur.

## Kurulum

```
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU'nun (NVIDIA CUDA) görüldüğünü doğrulamak için:

```
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

`1` (veya daha fazla) dönmeli. `0` dönerse veya hata alırsanız `transcriber.py`
içindeki `Transcriber(device="cuda", ...)` çağrısını `device="cpu",
compute_type="int8"` yaparak CPU'ya düşürebilirsiniz (daha yavaş ama çalışır).

İlk çalıştırmada:
- Whisper modeli (`large-v3-turbo`, varsayılan) internetten otomatik indirilir.
- Argos Translate backend'i seçiliyse, kaynak/hedef dil çifti için gerekli
  paketi otomatik indirir (doğrudan çift yoksa İngilizce üzerinden pivot yapar).

Sonrasında internet gerekmez.

### NLLB-200 çeviri modelini kurma (varsayılan backend)

Varsayılan çeviri motoru NLLB-200-distilled-600M'dir (Argos'a göre belirgin
şekilde daha akıcı/doğal çeviri, GPU'da <0.3s gecikme). Bu modeli bir kez
CTranslate2 formatına dönüştürmek gerekir (~2.4GB indirme, tek seferlik):

```
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir models\nllb-ct2 --quantization float16
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M').save_pretrained('models/nllb-ct2/tokenizer')"
```

Bu adımdan sonra `models/nllb-ct2/` tamamen offline çalışır. Bu adımı atlarsanız
veya `models/nllb-ct2` klasörü yoksa, Ayarlar'dan çeviri motorunu "Argos
Translate (CPU, basit)" olarak değiştirin — kurulum gerektirmeden hemen çalışır.

## Çalıştırma

```
python main.py
```

Uygulama sistem tepsisine (saat yanına, gizli oklar altında olabilir) bir simge
ekler. Sağ tık menüsünden:
- **Başlat / Durdur**: dinlemeyi aç/kapat.
- **Konumu Kilitle**: altyazı çubuğunu tıklamaya şeffaf yapar (kilitliyken
  altındaki pencereye tıklamalar geçer); kilitli değilken sürükleyerek
  taşıyabilirsiniz.
- **Ayarlar...**: kaynak dil (veya "Otomatik algıla"), hedef dil, dinlenecek
  ses çıkış cihazı, Whisper model boyutu ve çeviri motoru (NLLB-200 / Argos).

## Bilinen sınırlar

- **DirectX exclusive fullscreen oyunlar**: altyazı çubuğu görünmez (Windows'un
  compositor'ı devre dışı kalır). Discord/OBS gibi araçlarda da aynı kısıt
  vardır; oyunu "sınırsız pencere" (borderless windowed) moduna alın.
- **Çeviri kalitesi**: NLLB-200 (600M) bulut servisleri (DeepL/GPT) kadar
  akıcı olmayabilir ama Argos'a göre belirgin şekilde daha doğal. Daha da
  yükseltmek isterseniz `translator.py`'deki `Translator` arayüzü sayesinde
  ileride yerel bir LLM (Ollama) tabanlı bir backend de eklenebilir.
- **NLLB dil kapsamı**: `translator.py` içindeki `ISO_TO_FLORES` sözlüğünde
  olmayan bir dil kodu kullanılırsa hata verir; yeni bir dil eklemek için o
  dilin FLORES-200 kodunu sözlüğe eklemek yeterli.
- **Whisper model boyutu**: `large-v3-turbo` RTX 4060 (8GB) için hız/doğruluk
  dengesi en iyi seçenektir; VRAM sıkışırsa `medium`/`small`'a düşün.

## Proje yapısı

| Dosya | Görev |
|---|---|
| `audio_capture.py` | WASAPI loopback ile sistem sesi yakalama |
| `vad_segmenter.py` | Silero VAD ile akan sesi cümlelere bölme |
| `transcriber.py` | faster-whisper (GPU) ile konuşma → metin |
| `translator.py` | Offline çeviri: NLLB-200 (CTranslate2/GPU, varsayılan) + Argos Translate (CPU) |
| `gpu_env.py` | ctranslate2'nin CUDA DLL'lerini (cuBLAS/cuDNN) bulması için ortam ayarı |
| `pipeline.py` | Yukarıdakileri thread/queue zinciriyle bağlar |
| `overlay_ui.py` | Her zaman üstte, yarı saydam altyazı çubuğu (PySide6) |
| `tray_app.py` / `settings_dialog.py` | Sistem tepsisi ve ayarlar penceresi |
| `config.py` | Ayarların `config.yaml`'a kaydedilmesi |
| `main.py` | Uygulamanın giriş noktası |
