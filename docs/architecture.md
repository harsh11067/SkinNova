# SkinNova — Architecture

Audience: Opus (implementer) and reviewers. Data shapes live in `contracts.md`; this file defines components, responsibilities, flows, threading, storage and failure handling.

---

## 1. System context

```
            ┌──────────────────────── Development (WSL + Kaggle) ─────────────────────────┐
            │ ml/data ─► ml/cv (Kaggle GPU) ─► skin_cls.tflite ───────────┐               │
            │ ml/llm  ─► Kaggle LoRA ─► HF merged ─► litert-torch export ─┼─► .litertlm   │
            │ ml/eval ─► reports/*.json (every number in the report)      │               │
            └─────────────────────────────────────────────────────────────┼───────────────┘
                                                                          │ adb / file share
┌──────────── Phone ──────────────────────────────────────────────────────▼───────────────┐
│ SkinNova APK (Kotlin, Compose)  +  filesDir/models/skinnova-e2b-v1.litertlm             │
│ No server. No account. `offline` flavor declares no INTERNET permission.               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Android module & package layout

Single Gradle module `:app` (multi-module adds build complexity with little benefit at this size). Packages:

```
com.skinnova.app
├── SkinNovaApp.kt                 Application; builds AppContainer (manual DI, no Hilt needed)
├── di/AppContainer.kt             singletons: ModelManager, EngineHolder, CvClassifier, repos
├── ui/
│   ├── theme/                     Color.kt Type.kt Shape.kt Spacing.kt (from design/tokens.json)
│   ├── nav/NavGraph.kt            routes: onboarding, setup, capture, questionnaire, analyzing,
│   │                              result/{id}, history, timeline/{spotId}, recapture/{spotId}, settings, about
│   ├── onboarding/  setup/  capture/  questionnaire/  analyzing/  result/
│   ├── history/  timeline/  settings/  about/
│   └── components/                TriageBanner, CategoryRow, UncertaintyMeter, GhostOverlay,
│                                  MicButton, EvidenceChip, MetricSparkline, DisclaimerFooter
├── ml/
│   ├── EngineHolder.kt            owns LiteRT-LM Engine lifecycle (one per process)
│   ├── GemmaEngine.kt             analyze(), transcribe(), extractIntake(), narrateChange(), localize()
│   ├── CvClassifier.kt            LiteRT .tflite, preprocessing per preprocess.json, temperature
│   ├── QualityGate.kt             blur (Laplacian var), exposure, skin-pixel ratio, size checks
│   ├── PromptBuilder.kt           fills templates from assets/prompts/*.txt
│   ├── OutputParser.kt            JSON extraction, schema validation, repair-retry policy
│   └── AudioCapture.kt            AudioRecord 16 kHz mono PCM16 → WAV bytes, 30 s cap, VAD trim
├── timeline/
│   ├── Aligner.kt                 ORB + BFMatcher + RANSAC homography (OpenCV)
│   ├── LesionSegmenter.kt         GrabCut seeded by tap + ellipse prior; mask cleanup
│   ├── CoinDetector.kt            HoughCircles + radius sanity; px-per-mm
│   ├── ChangeMetrics.kt           area_ratio, contrast_delta (ΔE2000), border, cv_shift
│   └── NoiseFloor.kt              per-spot calibration from repeat captures
├── safety/
│   ├── RedFlagRules.kt            R1–R8, T1–T3 (contracts §4); pure functions
│   ├── TierResolver.kt            max(LLM, rules, class floors, timeline) — monotone
│   └── ContentGuards.kt           dose regex, prescription-term list, single-diagnosis phrasing check
├── i18n/
│   ├── Localizer.kt               deterministic strings + LLM translation of free text
│   └── Tts.kt                     TextToSpeech wrapper, language availability check
├── model/                         data classes from contracts.md (kotlinx.serialization)
├── data/
│   ├── db/                        Room: AnalysisEntity, SpotEntity, CaptureEntity, MetricsEntity
│   ├── ImageStore.kt              AES-GCM encrypted files, key in Android Keystore
│   └── Repositories.kt
├── report/DoctorSummaryPdf.kt     android.graphics.pdf.PdfDocument
├── reminders/ReminderWorker.kt    WorkManager, local notification only
└── setup/ModelManager.kt          locate / import (SAF) / verify sha256 / delete / space checks
```

Build config:
- `productFlavors { offline {}; online {} }`. INTERNET permission only in `src/online/AndroidManifest.xml`. ModelDownloader is in `src/online/` only, so it can't be called from offline.
- `abiFilters "arm64-v8a"` (smaller APK; nearly all phones from the last ~6 years).
- `aaptOptions.noCompress += ["tflite"]` so the CV model is memory-mapped from the APK.
- `minSdk`: use what LiteRT-LM requires (check AAR manifest; record in resources §7). `targetSdk`/`compileSdk`: latest stable.

---

## 3. Core analysis pipeline (sequence)

```
User taps Analyze
 │
 ├─[Main]   AnalyzingViewModel.start(captureId, answers)
 ├─[Default] QualityGate.check(bitmap)              ~20 ms   → fail ⇒ RetakeSuggested (user may force → R8)
 ├─[Default] CvClassifier.classify(bitmap)           50–300 ms → probs[11], calibrated by T
 ├─[Default] RedFlagRules.pre(answers, cv, timeline?) <1 ms   → ruleTier, messages
 ├─[Default] PromptBuilder.analysis(...)             <5 ms
 ├─[LLM]    EngineHolder.await()                    0 s (warm) / 5–15 s (cold)
 ├─[LLM]    GemmaEngine.analyze(imagePath, prompt)  streams tokens → UI preview
 ├─[Default] OutputParser.parse(text)
 │            invalid → GemmaEngine.repair(text, errors) once → invalid → Fallback
 ├─[Default] RedFlagRules.post + TierResolver.resolve
 ├─[LLM]    Localizer.translateFreeText (only if lang ≠ en)
 ├─[IO]     Repository.save(analysis)              (only if history enabled)
 └─[Main]   navigate Result(id)
```
`[LLM]` = a single-thread dispatcher `Dispatchers.Default.limitedParallelism(1)` guarded by a `Mutex` in EngineHolder. The engine never runs two conversations at once.

### Analysis state machine
```kotlin
sealed interface AnalysisState {
  data object Idle : AnalysisState
  data class CheckingPhoto(val progress: Float) : AnalysisState
  data class Classifying(val done: Boolean) : AnalysisState
  data object LoadingModel : AnalysisState
  data class Generating(val partialText: String, val tokens: Int) : AnalysisState
  data object Validating : AnalysisState
  data object Translating : AnalysisState
  data class Done(val result: FinalResult) : AnalysisState
  data class Fallback(val result: FinalResult, val reason: FallbackReason) : AnalysisState
  data class Failed(val error: UiError, val retryable: Boolean) : AnalysisState
}
```
Cancellation: user Cancel → cancel the coroutine job → `conversation.close()` → the engine stays loaded.

### Fallback renderer
If the LLM fails (init error, OOM, double invalid JSON, timeout 180 s): build `FinalResult` from CV top-3 + condition-card summaries + rule messages + tier floors. The UI labels it "Basic mode". This path must also be fully tested.

---

## 4. Engine lifecycle & memory

```kotlin
class EngineHolder(private val ctx: Context, private val models: ModelManager) {
  private val mutex = Mutex()
  private var engine: Engine? = null
  suspend fun <T> use(block: suspend (Engine) -> T): T = mutex.withLock {
      val e = engine ?: create().also { engine = it }
      block(e)
  }
  private fun create(): Engine {
      val path = models.activeModelPath() ?: throw ModelMissing()
      return try { build(path, Backend.GPU()) } catch (t: Throwable) { build(path, Backend.CPU()) }
  }
  private fun build(path: String, b: Backend) = Engine(EngineConfig(
      modelPath = path, backend = b, visionBackend = b, audioBackend = Backend.CPU(),
      cacheDir = ctx.cacheDir.path)).apply { initialize() }
  fun release() { engine?.close(); engine = null }   // onTrimMemory(RUNNING_CRITICAL / UI_HIDDEN+bg)
}
```
- Warm-up: start `use{}` in the background when the user reaches the Questionnaire, so the model loads while they answer.
- Backend choice is persisted after the first success. A GPU crash at init flips to CPU next time.
- Conversations are short-lived: one per task (analyze, transcribe, extract, narrate, translate), each with its own `systemInstruction`, closed after use. This keeps the KV cache small and avoids context leaking between tasks.
- Sampler: analysis/extraction `temperature=0.2, topK=40, topP=0.95`; narration/translation `0.4`. `maxOutputToken`: analyze 700, extract 300, narrate 200, translate 600. Thinking disabled (`ThinkingConfig(enableThinking=false)`).

**Memory budget (8 GB phone, GPU backend), to be confirmed in Phase 0:**
| Item | Est. |
|---|---|
| Gemma E2B weights resident (GPU) + mmapped embeddings | ~0.7–1.0 GB RAM (+ GPU mem) |
| Vision/audio encoders (loaded on demand) | + few hundred MB |
| KV cache @ ~2k ctx | ~100–200 MB |
| CV model + OpenCV + bitmaps (downscaled to ≤ 1024 px) | ~100–150 MB |
| App/UI | ~150 MB |
Bitmaps: decode with `inSampleSize` to ≤ 1024 px long side; the Gemma input is saved as a 768 px JPEG q90 temp file for `Content.ImageFile`.

---

## 5. Prompt & token budget
- System prompt ≤ 350 tokens; 3 condition cards ≤ 3×160; answers JSON ≤ 120; CV/rule block ≤ 80; image tokens: Gemma 4 vision token count depends on the resolution setting (measure in Phase 0); output ≤ 700. Target total ≤ 2,048 so the context stays short and prefill fast.
- Templates in `assets/prompts/`: `analyze_system.txt`, `analyze_user.txt`, `repair.txt`, `transcribe.txt`, `extract_system.txt`, `narrate_system.txt`, `translate_system.txt`. Each has a version header `# v1`; the version is stored with every saved result. **Training and app load the same files** (`ml/llm/prompts/` is a symlink/copy checked by a test).

---

## 6. Safety layer
```
answers, cv, timeline ──► RedFlagRules.pre ──► ruleTier + messages ──► injected in prompt
LLM JSON ──► OutputParser (schema) ──► ContentGuards (dose, Rx terms, "you have X") ──► reject → repair/fallback
         ──► RedFlagRules.post ──► TierResolver.resolve = max(llmTier, ruleTier, classFloors(top3 p≥0.15), timelineTier)
```
Properties enforced by tests: monotone (no input can make the final tier lower than ruleTier); red-flag messages always rendered above results; disclaimer on every result view and in the PDF. Safety strings are never LLM-translated; they come from `strings.xml`.

---

## 7. USP-1 architecture — SkinTimeline

```
Recapture screen: CameraX Preview + GhostOverlay(baseline, alpha .35) + live ORB align score (every 500 ms on 320 px frames)
 └─ Capture ─► Aligner.align(new, baseline) → H, inlierRatio
     ├─ inlierRatio < 0.25 → "Couldn't match the spot, retake" (no metrics)
     └─ warp new → baseline frame
         ├─ LesionSegmenter(baseline seed point stored in SpotEntity) → masks both
         ├─ CoinDetector (both) → pxPerMm? 
         ├─ ChangeMetrics → area_ratio, contrast_delta, border_delta, cv_shift, confidence
         ├─ NoiseFloor.thresholds(spot)
         ├─ RedFlagRules.timeline (T1–T3) → timelineTier
         └─ GemmaEngine.narrateChange(metrics, cvBefore, cvAfter) → 2–4 sentences (validated length + guards)
```
Data model: `SpotEntity(id, name, bodySite, seedX, seedY, coinDiameterMm?, reminderDays, createdAt)` 1—n `CaptureEntity(id, spotId, imageRef, takenAt, cvProbs, analysisId?)` 1—1 `MetricsEntity(captureId, baselineCaptureId, ...metrics, confidence)`.
Baseline = first capture with align quality ok; the user can re-baseline.
Reminders: `WorkManager` periodic → local notification "Time to re-check 'left forearm spot'".
PDF: A4, page 1 summary (spot, dates, tier history, disclaimer), page 2+ photo grid with date + metrics, last page the answers and CV top-3 per capture. Shared via `FileProvider` + `ACTION_SEND` (user-initiated; no network by the app itself).

## 8. USP-2 architecture — Bhasha Voice Intake
```
MicButton (RECORD_AUDIO runtime permission) ─► AudioCapture: 16 kHz mono PCM16, max 30 s, energy VAD trims silence ─► WAV bytes
 └─ GemmaEngine.transcribe(wav, langHint)    prompt: "Transcribe exactly in the spoken language/script. Output only the transcript."
     └─ TranscriptConfirm UI (editable)
         └─ GemmaEngine.extractIntake(transcript) → IntakeExtraction JSON (contracts §9)
             └─ IntakeValidator: schema; every non-null field has evidence; evidence is a substring
                (normalized, fuzzy ≥ 0.8) of transcript; enums valid → drop failing fields (never guess)
                 └─ Questionnaire prefilled; EvidenceChip under each prefilled field; user confirms all
```
Audio input via `Content.AudioBytes(wav)` (format accepted to be verified in Phase 0 spike 4). Output language: `Localizer` uses `values-hi/strings.xml` etc. for all fixed text; free explanation → `GemmaEngine.localize(text, lang)` → ContentGuards rerun on the translated text (dose regex works on digits). TTS: `TextToSpeech.isLanguageAvailable(Locale("hi","IN"))`; if missing, hide 🔊 and show "Install Hindi voice in system settings".

## 9. Storage & privacy
- Room DB in app-private storage. Images saved as AES-256-GCM files; key generated in Android Keystore (`KeyGenParameterSpec`, non-exportable). (Jetpack `security-crypto` is deprecated, so Keystore + `javax.crypto` is used directly.)
- History **off by default**; timeline requires opt-in at "Track this spot".
- `android:allowBackup="false"`, `dataExtractionRules` exclude all. No analytics SDK. No crash reporter that uploads.
- "Delete everything" in Settings: DB, images, cached temp files (model file kept unless also selected).

## 10. Error handling matrix
| Failure | Detection | User sees | Recovery |
|---|---|---|---|
| Model file missing/corrupt | ModelManager sha256 / size | Setup screen | re-import |
| Not enough storage for import (< size + 500 MB) | StatFs | clear message + needed GB | free space |
| GPU init fails | exception in build() | nothing (silent) | CPU fallback, persisted |
| OOM / process killed during load | next launch flag `lastLoadCrashed` | "Device low on memory" + Basic mode option | CPU + smaller maxTokens |
| Invalid JSON twice | OutputParser | "Basic mode" result | — |
| Generation timeout 180 s | coroutine timeout | Basic mode | — |
| Audio permission denied | runtime permission | tap-only questionnaire | settings link |
| Transcript empty / garbage | length / script check | "Couldn't hear clearly, try again or tap answers" | — |
| Timeline align fail | inlierRatio | "Couldn't match the spot" + tips | retake |
| Camera unavailable | CameraX error | gallery picker | — |

## 11. Training-side architecture (`ml/`)
```
ml/
├── common/      paths.py (reads .env), seed.py, io.py, schema.py (pydantic mirrors of contracts)
├── data/        download, normalize, dedupe, build_label_map, make_splits, data_card
├── cv/          dataset.py, train.py, calibrate.py, export_tflite.py, parity.py, notebooks/cv_train.ipynb
├── llm/
│   ├── prompts/              (same files as android assets/prompts — test enforces equality)
│   ├── cards/condition_cards.json
│   ├── build_sft_dataset.py  tasks T1–T9 → JSONL + images → HF dataset (private)
│   ├── validate.py           schema + guards (mirror of OutputParser/ContentGuards)
│   ├── notebooks/gemma4_e2b_lora.ipynb
│   ├── merge_and_push.py     (runs in notebook)
│   ├── export_litertlm.sh    litert-torch export_hf … + sha256 → model_manifest.json
│   └── run_litertlm.py       desktop inference via litert-lm Python API (used by eval)
├── timeline/    synth_changes.py (ground-truth transforms), eval_timeline.py (Python port of ChangeMetrics)
├── voice/       manifest of recordings, eval_voice.py
└── eval/        eval_cv.py, eval_llm.py, eval_combined.py, parity.py, redflags.py, bootstrap.py, judge.py
```
Python ↔ Kotlin parity for deterministic logic (red flags, tier resolve, validator, change metrics) is enforced by shared JSON fixtures in `tests/fixtures/` run by both pytest and JUnit.

## 12. Dependencies (pin in resources §7)
Kotlin, Compose BOM, Navigation-Compose, Lifecycle/ViewModel, CameraX (core, camera2, lifecycle, view), LiteRT-LM android, LiteRT (+gpu), OpenCV Android, Room (+ksp), WorkManager, kotlinx.serialization, kotlinx.coroutines, Coil (thumbnails). Test: JUnit4/5, kotlinx-coroutines-test, Turbine, Robolectric (pure JVM rules), Compose UI test, Espresso/UIAutomator (permission dialogs).
