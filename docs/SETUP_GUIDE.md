# SkinNova — setup guide (phone + developer)

Two ways in: **A** just use the app on an Android phone (no computer), **B** work on the code.

---

## A. Use the app on your phone (no computer needed)

Needs: Android 8+ (arm64), ~6 GB free storage, 6–8 GB RAM recommended.

1. **Install the app.** On the phone open **https://github.com/harsh11067/SkinNova/releases/latest**, download
   `SkinNova-<version>.apk`, open it, allow "install unknown apps" for your browser when Android asks, Install.
2. **Get the AI model file (3.9 GB, once).** The app runs a fine-tuned Gemma 4 E2B on the phone; it is not inside the
   APK. In the phone's browser:
   - **Fine-tuned SkinNova model (best):** https://huggingface.co/kumarharsh11067/skinnova-gemma4-e2b-litertlm →
     *Files* → `skinnova-e2b-v2.litertlm` → download. The repository is private: sign in to Hugging Face and ask the
     owner (Harsh) for access first.
   - **No access yet?** The app also accepts Google's public **stock** Gemma 4 E2B (works fully offline, less tuned for
     skin): https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm → `gemma-4-E2B-it.litertlm` (2.6 GB).
3. **Import it.** Open SkinNova → *Get Started* → on *Set up the AI model* tap **Import model file** → pick the file in
   *Downloads*. The app copies it, checks its SHA-256 against the list it trusts, and says *Verified*. (Later:
   Profile → On-device model.)
4. **Allow it to keep running** (Xiaomi, vivo, Oppo, Realme…): when the analysis screen shows *Keep SkinNova running*,
   tap **Allow**; on vivo also Settings → Battery → Background power consumption management → SkinNova → Allow.
5. Done. Everything now works in airplane mode. First analysis after installing/updating: ~2 min extra while the phone
   builds the model's GPU cache; afterwards ~1–1.5 min per full analysis on a mid-range phone (the *Early look* shows
   the image model's result and advice level within a second).

| Expected file | Size | SHA-256 |
|---|---|---|
| `skinnova-e2b-v2.litertlm` (shipped) | 3,862,170,848 B | `046020629f9dadf1f4d02d23c72810d6c08d877a7f3d6405e0f341e3bc2d22ed` |
| `skinnova-e2b-v1.litertlm` (older) | 3,862,170,848 B | `30064f2c81cfd0395894be0e7074b8c0046a158e7764652b381e67011eab886a` |
| `gemma-4-E2B-it.litertlm` (stock fallback) | 2,588,147,712 B | `181938105e0e…` (full value in `android/app/src/main/assets/model_manifest.json`) |

---

## B. Work on the code

### B1. Get the code and the tools
```bash
git clone git@github.com:harsh11067/SkinNova.git && cd SkinNova      # or https://github.com/harsh11067/SkinNova.git
cp .env.example .env            # put your own HF_TOKEN / KAGGLE_* here — never commit .env
```
- **Android:** Android Studio (Ladybug or newer) → *Open* the `android/` folder. JDK 17, Android SDK 36, NDK not needed.
- **Python (ML, evaluation):** Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/):
  ```bash
  uv venv .venv && uv pip install --python .venv -r ml/requirements.txt                 # training · data · evaluation
  uv venv .venv-export && uv pip install --python .venv-export -r ml/requirements-export.txt   # TFLite / LiteRT-LM export + eval
  .venv/bin/python -m pytest -q ml/tests                                               # 154 tests, no data needed
  ```

### B2. Build, test, install
```bash
cd android
./gradlew :app:testOfflineDebugUnitTest          # 34 JVM tests (rules, validators, app lock, care notes, chat safety…)
./gradlew :app:assembleOfflineRelease            # → app/build/outputs/apk/offline/release/app-offline-release.apk
./gradlew :app:assembleOfflineDebug :app:assembleOfflineDebugAndroidTest   # for on-device tests
adb install -r -g app/build/outputs/apk/offline/release/app-offline-release.apk
```
- `offline` flavor = no INTERNET permission at all (the privacy claim). `online` adds a one-time model download.
- Without your own keystore the release APK is signed with the debug key (fine for testing and sharing). For a store
  release set `SKINNOVA_KEYSTORE_PATH / _PASSWORD / SKINNOVA_KEY_ALIAS / SKINNOVA_KEY_PASSWORD` in the environment.
- Put the model on a USB-connected phone: `scripts/push_model.sh path/to/skinnova-e2b-v2.litertlm`, or download it with
  `hf download kumarharsh11067/skinnova-gemma4-e2b-litertlm skinnova-e2b-v2.litertlm --local-dir models/litertlm/skinnova`.
- On-device tests: `scripts/device_tests.sh` (C5 parity, timeline TL3, navigation, app lock, notification…). Tests that
  would wipe app data **skip themselves** on a phone that has a profile, PIN, history or tracked spots.

### B3. Where everything is
| What | Path |
|---|---|
| Android app (Kotlin, Jetpack Compose) | `android/app/src/main/java/com/skinnova/app/` |
| ‑ screens | `ui/screens/` (StartScreens = landing/onboarding/setup, HomeScan, Questions, Result, Ask, Relief, Library, TimelineScreens, Security) |
| ‑ AI on the phone | `ml/` (GemmaEngine = LiteRT-LM, CvClassifier = TFLite image model, AnalysisPipeline, PromptBuilder) |
| ‑ safety (deterministic) | `safety/` (RedFlagRules, TierResolver, ContentGuards, ChatSafety) |
| ‑ app lock / profile / notifications | `security/`, `data/Profile.kt`, `notify/` |
| Image model shipped in the APK | `android/app/src/main/assets/cv/skin_cls.tflite` (+ `preprocess.json`) |
| Prompts (shared by training and app) | `ml/llm/prompts/` ≡ `android/app/src/main/assets/prompts/` (a test enforces equality) |
| Condition notes · care & pharmacy notes | `ml/llm/cards/condition_cards.json` · `android/app/src/main/assets/care/relief.json` |
| Library photos + credits | `android/app/src/main/res/drawable-nodpi/lib_*.webp` · `assets/library/credits.json` |
| Data pipeline | `ml/data/` (download, de-duplicate, patient-grouped splits) |
| Image-model training / export | `ml/cv/` (`train.py`, `eval_cv.py`, `tta_eval.py`, `skin_gate.py`, `export_tflite.py`) |
| Gemma fine-tuning (Kaggle notebooks) | `ml/llm/notebooks/` (`gemma4_e2b_lora_full.ipynb`, `export_*`), SFT builder `ml/llm/` |
| LoRA adapters | Kaggle datasets `harsh11067/skinnova-lora-v1`, `harsh11067/skinnova-lora-v2` |
| Phone model file | HF `kumarharsh11067/skinnova-gemma4-e2b-litertlm` (private) |
| SkinTimeline (spot tracking) | `ml/timeline/` (Python) ≡ `android/.../timeline/Timeline.kt` |
| Every number in the README | `reports/*.json`, summary `reports/final_report.md` |
| Decisions and why | `docs/decisions.md` · run log `PROGRESS.md` · architecture `docs/architecture.md` · contracts `docs/contracts.md` |
| Big local data / models (not in git) | `data/` → `~/skinnova-data/data`, `models/` → `~/skinnova-data/models` (symlinks) |

### B4. Rebuild the data or retrain (optional, heavy)
- Datasets need accounts/keys in `.env` (Kaggle; SCIN is public on Google Cloud). `ml/data/` (`sources.py`,
  `normalize.py`, `dedupe.py`, `make_splits.py`) downloads, de-duplicates and splits them into `data/splits/*.csv`
  (local, not in git — ask Harsh for the frozen split files if you need the exact reported numbers; their summary is
  `reports/splits.json`).
- Image model: `python -m ml.cv.train` on a GPU (~3 h on an RTX 3050), then `python -m ml.cv.export_tflite --gate
  skin_gate_v2.npz --tta id,h,v,r180` (writes the `.tflite`, parity reports and test fixtures).
- Gemma LoRA: run `ml/llm/notebooks/gemma4_e2b_lora_full.ipynb` on Kaggle (T4), then the export notebook →
  `.litertlm`; add its SHA-256 to `model_manifest.json`.

### B5. Rules of the project (please keep)
Read `CLAUDE.md` first. In short: inference stays on the phone; safety rules are code, never the model; the model can
raise an advice level but never lower it; prompts are identical for training and app; tune on validation data only;
every number comes from a script in `ml/eval/`; secrets only in `.env`.

### Licences
Code: see the repository. Gemma 4: [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Some training images are
non-commercial (DermNet mirror, SkinDisNet), so the trained models are for non-commercial, educational use.
SkinNova gives preliminary, educational information — it is not a medical device.
