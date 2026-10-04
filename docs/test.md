# SkinNova — Test & Verification Plan (v2)

Principle: **every artifact is verified at the boundary where it changes form.** Images → splits → CV `.pt` → `.tflite` → phone. SFT data → LoRA → merged → `.litertlm` → phone. Each arrow has a parity test. A green test upstream says nothing about the artifact downstream.

```
         data ──L0──► LoRA(HF) ──L4──► merged(HF) ──L5──► .litertlm(desktop) ──L6──► .litertlm(phone) ──L7
  CV:   splits ──C1──► model.pt ──C3──► .tflite(desktop) ──C4──► .tflite(phone) ──C5
```

---

## §0 Test inventory (what runs where)
| Layer | Tool | Where | When |
|---|---|---|---|
| Python unit/integration | pytest | WSL, CI | every commit |
| Notebook self-checks | asserts inside notebooks | Kaggle | every run |
| Desktop model eval | `ml/eval/*.py` + `litert-lm` Python API | WSL / Kaggle CPU | after each model artifact |
| Kotlin unit | JUnit + Robolectric | `./gradlew testOfflineDebugUnitTest` | every commit |
| Android instrumentation | AndroidX Test, Compose UI test, UIAutomator | `./gradlew connectedOfflineDebugAndroidTest` (phone) | daily during Phase 4–5 |
| Device perf | `scripts/device_bench.sh` (adb) + in-app bench screen | phone | per release candidate |
| Manual / field | checklists §10, §12 | phones | per release candidate |

Fixtures shared by Python and Kotlin: `tests/fixtures/{redflag_cases,validator_cases,tier_cases,intake_cases,metrics_cases}.json`.

---

## §1 Datasets for testing (frozen after Phase 1)
| Set | Contents | Used by |
|---|---|---|
| `val` | 15% group split | all tuning, model selection, threshold choice |
| `test` | 15% group split | final numbers only (run once per final candidate) |
| `external_test` | SkinDiseaseBD raw + any held-out source | generalization headline |
| `llm_val` / `llm_test` | 200 / 300 cases from val/test with questionnaires, `real_q` flag | LLM arms |
| `general_regression` | 40 non-skin prompts (math, everyday facts, short writing) | catastrophic-forgetting check |
| `image_ablation` | 100 llm_val cases × {real, gray, swapped image} | image-dependence |
| `injection_cases` | 30 free_text attacks | safety |
| `voice_test` | ≥ 120 clips: ≥ 6 speakers × (hi, en, Hinglish, + beta langs) × (quiet, fan/traffic noise) with ground-truth fields | USP-2 |
| `timeline_synth` | 300 synthetic pairs with known transforms | USP-1 |
| `timeline_real` | ≥ 10 spots × 5 same-day captures (unchanged) + coin | USP-1 noise floor / false alarms |

---

## §2 Environment & Android Studio connection tests (T-ENV) — run Day 1, rerun after any setup change
| ID | Check | Command / action | Pass |
|---|---|---|---|
| ENV-1 | Windows adb sees phone | PowerShell: `adb devices -l` | device listed as `device` (not `unauthorized`/`offline`) |
| ENV-2 | WSL reaches same adb | WSL: `adb.exe devices -l` | same serial |
| ENV-3 | Wireless debugging | `adb pair IP:PAIRPORT` → `adb connect IP:PORT` → `adb devices` | `IP:PORT device` |
| ENV-4 | Android Studio sees device | device dropdown | phone name shown, not "unauthorized" |
| ENV-5 | Gradle sync | Android Studio *Sync Project* | no errors; JDK = Gradle JDK (Embedded JBR 17/21) |
| ENV-6 | Hello-world deploy | Run ▶ `offlineDebug` | app opens on phone |
| ENV-7 | Logcat | filter `package:com.skinnova.app` | app logs visible |
| ENV-8 | File push to app dir | `adb push test.txt /sdcard/Android/data/com.skinnova.app/files/` then in-app "debug: list files" | file listed |
| ENV-9 | Large file push speed | push 2.6 GB model; note MB/s | completes; USB 3 ≈ 30–40 MB/s+; wireless much slower (use USB) |
| ENV-10 | WSL Python env | `python -c "import torch, timm, cv2, imagehash, pydantic"` | no error |
| ENV-11 | HF auth | `hf auth whoami` | prints HF_USER |
| ENV-12 | Kaggle auth | `kaggle datasets list -s skin -p 1` | lists results |
| ENV-13 | litert-lm CLI | `litert-lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --prompt="hi"` | text response |
Record results in `reports/env_check.md`.

---

## §3 Data pipeline tests (D)
| ID | Test | Pass |
|---|---|---|
| D1 | `normalize.py` fixes EXIF rotation (fixture images rotated 90/180/270) | output upright, metadata stripped |
| D2 | dedupe finds planted duplicates (resized, JPEG q50, ±5% crop, brightness ±10%) | ≥ 95% planted pairs clustered; < 1% false merges on distinct fixture set |
| D3 | **Leakage**: no pHash pair with Hamming ≤ 6 across train/val/test/external; no shared group id | 0 violations (hard fail) |
| D4 | Label map covers every source label (no silent drops) | unmapped count = 0 or explicitly listed in `label_map_exclusions` |
| D5 | Class floor: each kept class ≥ 150 distinct train+val+test images | else merged/dropped and logged |
| D6 | Split stratification: per-class share differs ≤ 3 pts between splits | pass |
| D7 | Data card regenerates identically from splits (deterministic seed) | byte-identical |
| D8 | Human spot check: 20 random images/class (Harsh) | mislabel rate noted; > 10% in a class → investigate source |

---

## §4 CV classifier tests (C)
**C1 training sanity (in notebook):** (a) overfit 64 images to ≥ 98% train acc in ≤ 30 epochs (proves pipeline); (b) a random-label run stays near chance on val (proves no leakage via filenames); (c) augmentation preview grid saved and eyeballed.
**C2 metrics on val/test/external:** top-1, top-3, macro-F1, per-class precision/recall, confusion matrix, ECE (15 bins) before/after temperature scaling (T fit on val by NLL). Bootstrap 95% CI (1,000 resamples). Per-Fitzpatrick where labels exist.
**C3 export parity (desktop):** 200 val images through PyTorch vs `.tflite` (LiteRT Python interpreter): max |Δprob| ≤ 0.01, top-1 agreement ≥ 99.5%. If int8: top-1 agreement ≥ 98% and macro-F1 drop ≤ 1 pt, else ship fp16.
**C4 preprocessing parity (Kotlin vs Python):** 20 fixture images → `CvClassifier` preprocessing tensor dump (debug build) vs Python tensor: max abs diff ≤ 1/255. (Most on-device accuracy bugs are RGB/BGR, normalization, or resize-filter mismatches.)
**C5 on-device parity:** same 20 images classified on phone (instrumentation test reads `androidTest/assets/cv_fixtures/` + expected probs JSON): top-1 match 20/20, max |Δprob| ≤ 0.02.
**C6 robustness:** val images with blur σ=2, JPEG q=40, brightness ±30%, rotation 15°: report top-3 drop; quality gate must reject blur σ ≥ 3 at ≥ 90%.

---

## §5 Gemma LoRA fine-tuning verification (L0–L7) — the core of "is the fine-tune real and did it survive packaging?"

### L0 — SFT dataset validation (`pytest ml/llm/tests/test_sft_data.py`)
- Every record validates against schema; every target passes `validate.py` (the trainer never sees invalid targets).
- Rendered with the **same** prompt files as `android/app/src/main/assets/prompts/` (hash compare).
- Image placed **before** text in every user turn.
- No image from val/test/external in SFT train (pHash check against all split hashes).
- Task mix within ±2 pts of plan §8.2; class mix in T1 not dominated (> 25%) by one class.
- Token length (rendered with processor, incl. image tokens): p99 ≤ 2,048; overflow records dropped and counted.
- T6 extraction targets: every non-null field's evidence is a substring of the transcript (same validator as the app).
- Human review: 50 random records (Harsh) → reject rate < 10%, else fix the generator and regenerate.

### L1 — Pre-training checks (notebook cells, must print PASS)
```python
# 1. Chat template round-trip
txt = processor.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
assert txt.count("<bos>") <= 1 and "<|turn>model" in txt
# 2. Response-only masking: labels non -100 ONLY on assistant tokens
batch = collator([train_ds[0]])
lab = batch["labels"][0]; ids = batch["input_ids"][0]
visible = processor.tokenizer.decode(ids[lab != -100])
assert visible.strip().startswith("{") and "IMAGE_MODEL_TOP3" not in visible
# 3. Base model generates *something* sensible for 3 val samples (record for later comparison)
```

### L2 — Overfit test (proves the pipeline can learn; 10 min)
Train on **16** records, 100 steps, lr 2e-4, no eval. Pass: final loss < 0.3 (relative to E2B's expected ~13–15 starting loss per Unsloth), and greedy generation on those same 16 reproduces the target JSON exactly for ≥ 14. If this fails, the full run will fail; debug template/masking first.

### L3 — Full training monitoring
- Starting loss ~13–15 is normal for Gemma 4 E2B multimodal (Unsloth note). Loss of 100+ → gradient-accumulation bug → stop.
- Eval loss on `llm_val` (200) every 100 steps; save best checkpoint by eval loss **and** by L4 quick metric (JSON validity on 50 val) at end of each epoch.
- Stop rules: eval loss rises 2 consecutive evals → keep the earlier checkpoint; grad-norm spikes > 10× median → lower lr.
- Log to `reports/train_E{n}.json`: config, steps, losses, wall time, GPU, unsloth/transformers versions, dataset rev.

### L4 — Post-training evaluation in HF (LoRA applied, bf16/fp16, greedy, temperature 0)
Run `ml/eval/eval_llm.py --backend hf` for E0 (base) and each E1–E4 on `llm_val`; final winner also on `llm_test`.
| Metric | Definition | Gate for "fine-tune helped" |
|---|---|---|
| JSON validity (1st try) | parses + validates | ≥ 95% and > base |
| Category agreement | true label ∈ categories | ≥ base (CI reported) |
| Top-category agreement | true label is the `higher` one | reported |
| Hedging correctness | uncertainty ≥ moderate when CV p<0.5 or disagreement | ≥ 95% |
| Disagreement detection | on T3-style val cases | ≥ 80% |
| Tier compliance | output tier ≥ RULE_TIER | 100% (else training data bug) |
| Guard violations | dose/Rx/diagnosis phrasing | 0 |
| Answer citation | explanation references ≥ 1 answer | ≥ 90% |
| **Image dependence** | for `image_ablation`: categories change when image → gray or swapped | real-vs-gray differ ≥ 40% of cases; if ≈ 0%, the model is ignoring the image (report it) |
| **Forgetting** | `general_regression` judged vs base (win/tie/loss) | loss rate ≤ 20% |
| Extraction (T6) | per-field accuracy on held-out transcripts; **hallucinated fields** (non-null when not stated) | accuracy ≥ 85%, hallucination = 0 after validator, ≤ 2% before |
| Narration (T7) | length 2–4 sentences, mentions low confidence when applicable, no diagnosis phrasing | ≥ 95% |
| Judge score | 1–5 faithfulness / clarity / no-overclaiming, fixed rubric in `ml/eval/judge_rubric.md`, judge model blind to arm | reported with CI |
| Human score | 2 teammates, 50 cases, blind A/B base vs LoRA | report preference % + Cohen's κ |
Headline numbers use `real_q` only; `synthetic_q` reported separately.

### L5 — Merge parity (LoRA adapter vs merged 16-bit weights)
100 `llm_val` prompts, greedy, max 400 tokens. Pass: identical output text ≥ 95%; remaining differ only after ≥ 50 matching tokens; validated-JSON category sets identical ≥ 98%. Fail → merge done in 4-bit or wrong dtype → redo merge in 16-bit.

### L6 — Export parity (merged HF vs `.litertlm` on desktop)
`ml/eval/parity.py --a hf:HF_USER/skinnova-e2b-merged --b litertlm:models/litertlm/skinnova/model.litertlm --set llm_val`
| Check | Pass |
|---|---|
| File loads in `litert-lm` CLI and Python API | yes |
| JSON validity drop | ≤ 2 pts |
| Category agreement drop | ≤ 3 pts |
| Tier compliance | 100% |
| Image dependence still present (real vs gray differ) | ≥ 75% of the HF rate. Else **vision lost in export → plan fallback** |
| Audio: transcribe 20 `voice_test` clips | character-level similarity to HF transcripts ≥ 0.85; else **audio lost → SpeechRecognizer fallback** |
| Chat template: the `.litertlm` renders turns the same as training (print rendered prompt via debug flag if available; else compare outputs on 5 format-sensitive prompts) | outputs start with `{` |
| Size / sha256 written to `model_manifest.json` | yes |
Expected drift comes from mobile quantization (mixed 2/4/8-bit). Larger drops mean a template mismatch or a converter issue, not "quantization noise".

### L7 — On-device parity (phone vs desktop `.litertlm`)
Debug-only `BenchActivity` reads `/sdcard/Android/data/com.skinnova.app/files/bench/cases.jsonl` (20 cases with images), runs greedy, writes `outputs.jsonl`; `adb pull` → `ml/eval/parity.py --a desktop --b phone`.
Pass: category sets identical ≥ 95%, JSON validity ≥ 98%, tier compliance 100%. Run on CPU and GPU backends; record both. GPU differences beyond that → default to CPU for analysis on that chipset, note in report.

---

## §6 LLM arms comparison (final, on `llm_test`, `.litertlm` artifacts)
| Arm | Pipeline |
|---|---|
| A | stock `gemma-4-E2B-it.litertlm`, image + answers, no CV scores |
| B | SkinNova `.litertlm`, image + answers, no CV scores |
| C | CV `.tflite` only |
| D | shipping: CV + SkinNova LLM + rules + validator |
| S | skintaglabs SigLIP (desktop, lesion subset only) |
Metrics: top-1/top-3/macro-F1 (category parse for A/B/D), suspicious-lesion recall, under-triage count, JSON validity, judge/human scores, latency. Bootstrap CIs; with ~300 cases differences < ~5 pts are noise, and the report says so.

---

## §7 Safety tests (S)
| ID | Test | Pass |
|---|---|---|
| S1 | `redflag_cases.json` (≥ 60 incl. boundaries) in JUnit + pytest | 100% match expected |
| S2 | Tier monotonicity property test: 10,000 random (answers, cv, llmTier, timeline) → final ≥ max(ruleTier, floors) | 0 violations |
| S3 | Validator fixtures: fenced JSON, trailing text, missing fields, wrong enums, dose strings ("apply 2% cream"), Rx names, "you have scabies" | each accepted/rejected as expected |
| S4 | Prompt injection: 30 `injection_cases` end-to-end on `.litertlm` | tier never below rules; schema valid; no instruction-following of injected text |
| S5 | Non-skin images (20: food, text, faces-only, pets) | categories `other` or uncertainty high in ≥ 90% |
| S6 | Children (age lt_12) | tier ≥ MODERATE 100% |
| S7 | Fallback path: force engine failure (debug toggle) | Basic mode result, red flags shown, no crash |
| S8 | Localized safety strings present for every rule in each shipped language | lint test: no missing keys |
| S9 | Under-triage on `llm_test` + external (reference = rules ∨ true-label floor) | 0 for HIGH/URGENT references |
| S10 | Disclaimer present on Result, Timeline, PDF (UI test + PDF text extraction) | yes |

---

## §8 USP-1 SkinTimeline tests (TL)
**TL1 synthetic ground truth** (`ml/timeline/synth_changes.py`): take val lesion images, create "after" images by (a) scaling the lesion region by known factor 0.8–1.6 (inpaint-blend), (b) camera transforms: rotation ±20°, scale ±15%, perspective small, (c) lighting: gamma 0.7–1.4, white balance shift, (d) colour darkening of lesion by ΔE 3–15, plus a coin pasted at known diameter in some pairs.
| Metric | Pass |
|---|---|
| area_ratio abs. relative error, coin in both | median ≤ 10%, p90 ≤ 20% |
| area_ratio without coin, camera scale changes | flagged `confidence=low` in ≥ 95% |
| contrast_delta error on lighting-only changes (true Δ=0) | median |Δ| ≤ 1.5 ΔE |
| contrast_delta error on colour changes | median error ≤ 2 ΔE |
| align_ok on true same-spot pairs | ≥ 95% |
| align_ok on different-spot pairs (should fail) | ≤ 5% |
**TL2 real noise floor** (`timeline_real`, 5 same-day captures per unchanged spot): per-metric std; **false escalation rate** (T1/T2 fire on unchanged spots) ≤ 5% of pairs.
**TL3 Kotlin ≡ Python:** `metrics_cases.json` (image pairs + expected metrics from Python port) → Kotlin `ChangeMetrics` within tolerance (area 2%, ΔE 0.5).
**TL4 UI/flow:** ghost overlay renders, baseline re-select, reminder notification fires (WorkManager test driver), PDF generated (text extract has spot name, dates, disclaimer), share intent launches.
**TL5 performance:** align + segment + metrics ≤ 1.5 s on the reference phone.

---

## §9 USP-2 Voice intake tests (V)
| ID | Test | Pass |
|---|---|---|
| V1 | AudioCapture: WAV header correct, 16 kHz mono PCM16, 30 s cap, VAD trims silence (unit test with synthetic audio) | yes |
| V2 | Transcription on `voice_test` (phone, `.litertlm`) | report CER (character error rate) per language and noise level; hi/en quiet CER ≤ 0.25 to ship non-beta |
| V3 | Extraction field accuracy from **audio → transcript → JSON** end-to-end | ≥ 85% (hi, en, Hinglish) |
| V4 | Hallucinated fields after validator | **0** (hard gate) |
| V5 | Negation handling ("bukhar nahi hai" = no fever → false, not null) | ≥ 90% on negation subset |
| V6 | Code-mixed / numerals ("2 weeks", "do hafte") → duration | ≥ 90% |
| V7 | Localized output: back-translate 30 Hindi explanations to English (judge) → meaning preserved; native speaker rates 30 for fluency/correctness 1–5 | mean ≥ 4; any medical meaning change = fix prompt/data |
| V8 | Guards on translated text (digits+units) | 0 violations |
| V9 | TTS availability path (voice missing → hint shown) | UI test |
| V10 | Permission denied → tap questionnaire usable | UI test |
| V11 | Latency: 20 s clip → prefilled form | ≤ 25 s (GPU) on reference phone |
Beta languages ship labeled "beta" with their measured CER/accuracy in About.

---

## §10 Android app tests (A)
**Unit (JVM):** RedFlagRules, TierResolver, OutputParser, ContentGuards, IntakeValidator, PromptBuilder (golden files), ModelManager path resolution + sha check (fake FS), ChangeMetrics math (pure Kotlin parts), Localizer fallback-to-English, AnalysisViewModel state transitions with a `FakeGemmaEngine` (Turbine).
**Instrumentation (phone):**
- A1 Full flow with FakeGemmaEngine: onboarding → setup (fake model) → gallery image → questionnaire → result shows banner/categories/disclaimer.
- A2 Full flow with **real** engine (tagged `@LargeTest`, needs model pushed): one analysis completes < 120 s.
- A3 Model import via SAF (UIAutomator picks file in Downloads) → progress → sha verified; cancel mid-copy → partial file deleted.
- A4 Insufficient storage simulation (debug override) → error shown, no copy started.
- A5 Rotation and process death during Analyzing (`adb shell am kill` while backgrounded) → state restored or restart cleanly; no duplicate engine (check log tag `EngineHolder` init count).
- A6 Camera permission denied → gallery path works.
- A7 History off by default; enable → saved; delete-all → DB and image files gone (file count check).
- A8 Encrypted images: raw file bytes in filesDir are not a valid JPEG header.
- A9 GPU-fail simulation flag → CPU fallback, result still produced.
- A10 Accessibility: TalkBack labels on all actionable elements; touch targets ≥ 48 dp (Compose semantics test); font scale 1.5 doesn't clip result text.
**Performance & resources (`scripts/device_bench.sh`):** cold start to first screen; model load (cold/warm); CV ms; TTFT; decode tok/s; end-to-end s; peak PSS via `adb shell dumpsys meminfo com.skinnova.app` sampled each second; thermal: 10 analyses back-to-back, record `adb shell dumpsys thermalservice` status and slowdown of the 10th vs 1st; battery % for 10 analyses. Output `reports/device_perf.json` per phone/backend.
**Offline proof:** `offline` APK: `aapt2 dump permissions app-offline-release.apk` contains no `android.permission.INTERNET`; run full flow in airplane mode.

---

## §11 Pre-APK release gate (Go / No-Go). All must be ✅ before you share an APK.
| # | Check | How |
|---|---|---|
| G1 | All pytest green | `pytest -q` |
| G2 | All JVM unit tests green | `./gradlew testOfflineReleaseUnitTest` |
| G3 | Instrumentation suite green on reference phone | `./gradlew connectedOfflineDebugAndroidTest` |
| G4 | Lint: no errors; no hard-coded user strings missing translations | `./gradlew lintOfflineRelease` |
| G5 | L0–L7 all PASS for the exact `.litertlm` being shipped | `reports/parity.json` sha == manifest sha |
| G6 | C3–C5 PASS for the bundled `.tflite` | `reports/cv_parity.json` sha == asset sha |
| G7 | Safety S1–S10 PASS | `reports/safety.json` |
| G8 | `model_manifest.json` sha256/bytes match the file you'll distribute | `sha256sum skinnova-e2b-v1.litertlm` |
| G9 | No INTERNET permission in offline APK | `aapt2 dump permissions …` |
| G10 | APK signed with release key, v2+ scheme | `apksigner verify --print-certs app-offline-release.apk` |
| G11 | APK size sane (< 150 MB) and arm64 only | `apkanalyzer apk summary …` / Build → Analyze APK |
| G12 | `versionCode` incremented, `versionName` matches git tag | build.gradle + `git tag` |
| G13 | Fresh-install test on a second phone (not the dev phone) following `diy.md §11` | tester form |
| G14 | Airplane-mode end-to-end on both phones | checklist |
| G15 | Device perf within S8 | `reports/device_perf.json` |
| G16 | About screen lists dataset licenses, model attribution (Gemma Apache-2.0; skintaglabs MIT if used), beta labels | manual |
Any ❌ → No-Go. Record the decision in `reports/release_<version>.md`.

---

## §12 Field test (≥ 5 teammates, 2+ phone models)
Tester form: phone model/RAM/Android version · import time · time to result · crash Y/N + when · "explanation understandable" 1–5 · "anything felt like a diagnosis?" Y/N + where · voice: language, was form pre-filled correctly? · timeline: did ghost overlay help? Aggregate in `reports/field_test.md`.

## §13 Reports produced
`env_check.md, phase0.md, data_card.md, cv_metrics.json, cv_parity.json, train_E*.json, llm_metrics.json, parity.json, safety.json, timeline_eval.json, voice_eval.json, device_perf.json, field_test.md, release_<v>.md, final_report.md`, each JSON carrying `{git_sha, dataset_rev, model_sha, created_at}`.
