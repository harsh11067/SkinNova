# SkinNova — Resources (links, commands, file locations, versions)

Legend: ✅ checked Oct 2026 · ⚠️ verify before relying · ❌ doesn't fit

## 1. Models
| What | Where | Notes |
|---|---|---|
| Gemma 4 E2B instruct (train base) | ✅ https://huggingface.co/google/gemma-4-E2B-it | Apache-2.0 |
| Unsloth mirror (use for training) | ✅ `unsloth/gemma-4-E2B-it` | Unsloth fixed the E2B/E4B `use_cache=False` garbage-logits bug; E2B LoRA fits 8–10 GB VRAM |
| Stock phone model (Phase 0, arm A) | ✅ https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm → `gemma-4-E2B-it.litertlm` (2,583 MB) | mixed 2/4/8-bit; vision + audio encoders loaded on demand; re-download if fetched before 5 May 2026 (speculative decoding) |
| skintaglabs SigLIP | ✅ https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier (MIT; SigLIP Apache-2.0) · code https://github.com/skintaglabs/skintaglabs.github.io | 878M params, ~3.5 GB; lesion-focused 10 classes; desktop baseline/teacher only |
| CV backbone | timm `efficientnet_b0.ra_in1k`, alt `mobilenetv3_large_100` | ImageNet-pretrained |

**No `.gguf`.** LiteRT-LM loads `.litertlm` only. Chain: safetensors → LoRA → merged safetensors → `litert-torch export_hf` → `.litertlm`.
**Opus + Hugging Face:** downloads directly (`hf download`). Training runs in Kaggle notebooks that pull the model themselves; results come back via your private HF repos.

## 2. Runtimes, libraries, tools
| Tool | Coordinate / install | Docs |
|---|---|---|
| LiteRT-LM Android | `com.google.ai.edge.litertlm:litertlm-android:<pin>` (versions: https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm-android) | ✅ https://ai.google.dev/edge/litert-lm/android |
| LiteRT-LM CLI / Python | `uv tool install litert-lm` · `pip install litert-lm-api` | ✅ https://ai.google.dev/edge/litert-lm/cli |
| Converter | `uv tool install litert-torch-nightly` | ✅ https://ai.google.dev/edge/litert-lm/models/gemma-4 |
| Fine-tune → device tutorial | — | ✅ https://ai.google.dev/edge/litert-lm/tutorials/convert-and-run |
| LiteRT (CV on Android) | `com.google.ai.edge.litert:litert` (+ `litert-gpu`) | https://ai.google.dev/edge/litert/android |
| PyTorch → .tflite | `pip install litert-torch` | https://github.com/google-ai-edge/litert-torch |
| Unsloth | `pip install unsloth` (in notebook) | ✅ https://unsloth.ai/docs/models/gemma-4/train |
| Unsloth E2B Vision notebook (template) | ✅ https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb | |
| Unsloth E2B Audio notebook (reference) | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Audio.ipynb | not fine-tuning audio in v1 |
| AI Edge eval | `uv tool install ai-edge-eval` | https://github.com/google-ai-edge/eval |
| Reference apps | AI Edge Gallery https://github.com/google-ai-edge/gallery · Captain Gemma https://github.com/erintwalsh/on-device-gemma-tutorial | engine lifecycle patterns |
| OpenCV Android | Maven `org.opencv:opencv:<pin>` (4.9+ publishes to Maven Central ⚠️ confirm) | https://opencv.org/android/ |
| CameraX | `androidx.camera:camera-{core,camera2,lifecycle,view}` | https://developer.android.com/media/camera/camerax |
| Room / WorkManager / Navigation-Compose | AndroidX | developer.android.com |
| PdfDocument | `android.graphics.pdf.PdfDocument` (platform) | https://developer.android.com/reference/android/graphics/pdf/PdfDocument |
| TextToSpeech | platform | https://developer.android.com/reference/android/speech/tts/TextToSpeech |
| SpeechRecognizer (fallback) | platform, `EXTRA_PREFER_OFFLINE` | https://developer.android.com/reference/android/speech/SpeechRecognizer |
| Kaggle CLI | `pip install kaggle` (≥ 1.8: `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`; legacy `kaggle.json` still works) | ✅ https://github.com/Kaggle/kaggle-api |

### Known-good commands
```bash
# Run stock model on desktop
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm --prompt="What is the capital of France?"

# Export safetensors → .litertlm (stock or fine-tuned)
litert-torch export_hf \
  --model=$HF_USER/skinnova-e2b-merged \
  --output_dir=models/litertlm/skinnova \
  --externalize_embedder \
  --jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm
sha256sum models/litertlm/skinnova/model.litertlm

# Push to phone (USB)
adb.exe shell mkdir -p /sdcard/Android/data/com.skinnova.app/files/models/
adb.exe push models/litertlm/skinnova/model.litertlm /sdcard/Android/data/com.skinnova.app/files/models/skinnova-e2b-v1.litertlm
```
⚠️ Unverified until Phase 0 / L6: that `export_hf` on a fine-tuned checkpoint keeps the vision **and audio** encoders working.

### Kotlin API cheat-sheet (official docs, Sep 2026)
```kotlin
val engine = Engine(EngineConfig(modelPath = path, backend = Backend.GPU(),
    visionBackend = Backend.GPU(), audioBackend = Backend.CPU(), cacheDir = ctx.cacheDir.path))
engine.initialize()                                  // seconds; background thread
val conv = engine.createConversation(ConversationConfig(
    systemInstruction = Contents.of(system),
    samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.2)))
conv.sendMessageAsync(Contents.of(Content.ImageFile(img), Content.Text(user))).collect { /* stream */ }
conv.sendMessage(Contents.of(Content.AudioBytes(wavBytes), Content.Text("Transcribe…")))
```
Manifest `<application>`: `<uses-native-library android:name="libvndksupport.so" android:required="false"/>` and `libOpenCL.so` (GPU). Errors: `LiteRtLmJniException`, `IllegalStateException`. Speculative decoding: `ExperimentalFlags.enableSpeculativeDecoding = true` (GPU recommended). Thinking: `ThinkingConfig(enableThinking = false)`.

### Reference performance (Google, Samsung S26 Ultra, E2B)
CPU: prefill 557 tok/s, decode ~47 tok/s, TTFT 1.8 s, ~1.7 GB RAM · GPU: prefill 3,808 tok/s, decode ~52 tok/s, ~0.7 GB RAM. Mid-range phones will be several times slower; measure in Phase 0.

## 3. Datasets (fill license + ✓ before training)
| Dataset | Link | Use | License | ✓ |
|---|---|---|---|---|
| mgmitesh Skin Disease Detection | https://www.kaggle.com/datasets/mgmitesh/skin-disease-detection-dataset | conditional | ⚠️ read Kaggle page | ☐ |
| PAD-UFES-20 | https://data.mendeley.com/datasets/zr7vgbcyr2/1 | train/val/test + real questionnaire metadata | CC BY 4.0 ⚠️ confirm | ☐ |
| SkinDisNet | https://data.mendeley.com/datasets/yj3md44hxg/2 | inflammatory classes, patient split | ⚠️ | ☐ |
| SkinDiseaseBD | https://data.mendeley.com/datasets/9ggd3shdr7 | external test (raw images only) | ⚠️ | ☐ |
| HAM10000 | https://doi.org/10.7910/DVN/DBW86T | optional lesions | CC BY-NC | ☐ |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k | skin-tone eval | ⚠️ | ☐ |
| DDI | https://ddi-dataset.github.io | diverse tones eval | access request | ☐ |
| Team voice / timeline sets | internal, consented | USP eval | team consent | ☐ |

## 4. Medical content sources (summarize, cite, never copy)
AAD https://www.aad.org/public/diseases · NHS https://www.nhs.uk/conditions/ · DermNet https://dermnetnz.org/topics · WHO scabies https://www.who.int/health-topics/scabies

## 5. File locations
```
C:\dev\skinnova  (= /mnt/c/dev/skinnova)              git repo (code, docs, design, small assets)
~/skinnova-data/                                      WSL native, symlinked as data/ and models/
├── data/raw/<source>/                                untouched downloads (+ raw/voice/, raw/timeline/)
├── data/processed/images/<sha1>.jpg                  normalized, deduped
├── data/splits/{train,val,test,external_test}.csv    path,label,source,group_id,phash,real_q
├── data/llm/{sft_train,sft_val,llm_test}.jsonl
├── models/litertlm/stock/gemma-4-E2B-it.litertlm     arm A
├── models/litertlm/skinnova/model.litertlm           → distributed as skinnova-e2b-v1.litertlm
├── models/tflite/skin_cls.tflite                     → copied to android/app/src/main/assets/cv/
└── models/cv/ckpt/
HF (private): $HF_USER/skinnova-sft-data · skinnova-e2b-lora-v1 · skinnova-e2b-merged
Phone: /sdcard/Android/data/com.skinnova.app/files/{models,bench,voice,timeline}/   (dev, adb-reachable)
       /data/data/com.skinnova.app/files/models/                                 (after in-app import)
Keys:  C:\keys\skinnova-release.jks  (never in repo)
```

## 6. Accounts
Hugging Face (write token, Gemma terms), Kaggle (phone-verified, API token, `HF_TOKEN` secret), optional Gemini/other judge API key for **evaluation only** (never in the app).

## 7. Versions (pinned 2026-10-05; each row verified by a build/test/run)
| Item | Version | Verified on |
|---|---|---|
| litertlm-android | 0.17.1 (Kotlin 2.4 metadata; minSdk 24; arm64 + x86_64 libs) | `./gradlew assembleOfflineDebug` OK; API checked with javap |
| litert (CV on Android) | com.google.ai.edge.litert:litert 2.2.0 (needs `android.uniquePackageNames=false`) | assemble OK |
| litert-torch (desktop export) | 0.9.4 in `.venv-export` (CPU torch 2.11.0, timm 1.0.30) | B0 NHWC export: max logit diff 5.6e-11 vs PyTorch |
| ai-edge-litert (desktop interpreter) | 2.2.0 | parity run |
| litert-lm CLI / litert-torch-nightly | not installed yet (needed for `.litertlm` export, Kaggle notebook) | — |
| unsloth / unsloth_zoo / transformers / trl / peft | 2026.9.14 / 2026.9.9 / 5.5.0 (notebook) / latest / 0.21.2 (local) | Kaggle smoke run (pending) |
| torch / timm / opencv-python | 2.11.0+cu130 (system, RTX 3050 6 GB) / 1.0.30 / 4.x headless | CV training, dedupe |
| OpenCV Android | org.opencv:opencv 4.12.0 (22 MB arm64 .so) | assemble OK |
| AGP / Kotlin / Gradle / Compose BOM | 9.0.1 / 2.4.20 / 9.2.1 / 2026.02.00 | `testOfflineDebugUnitTest` green |
| compileSdk / targetSdk / minSdk | 36 / 36 / 26 | assemble OK |
| JDK (WSL build) | Temurin 17 (`~/android/jdk`) | gradle |
| Kaggle CLI | 2.2.4 as uv tool (Python 3.12) — `--accelerator NvidiaTeslaT4` | `kaggle quota` |
| Reference phone (model, SoC, RAM, Android) | — (Harsh: d2y.md) | |
| Second phone | — | |
