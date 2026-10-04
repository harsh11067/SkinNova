# SkinNova: On-Device Dermatology Information Assistant

> **Not a medical device.** SkinNova gives preliminary, educational information and tells you when to see a doctor. It does not diagnose and does not suggest prescription treatment.

SkinNova is an Android app that analyzes a skin photo with your symptoms and explains possible condition categories, how uncertain it is, and how soon to see a dermatologist. **Everything runs on the phone.** No internet, no account, no paid API.

## What it does
| Feature | How |
|---|---|
| 📷 Photo check + classification | quality gate → on-device CNN (`.tflite`, LiteRT) → calibrated top-3 categories |
| 🧠 Personalized explanation | Gemma 4 E2B (+ LoRA) via **LiteRT-LM** reads the photo, model scores, your answers and curated condition notes → validated JSON |
| 🛡️ Safety you can audit | deterministic red-flag rules in Kotlin; the AI can raise but never lower the urgency |
| 🗣️ **Bhasha Voice Intake** (USP) | describe symptoms by voice in Hindi/English (more languages beta); Gemma's on-device audio encoder transcribes them, fields are extracted only from what you actually said, and you confirm every field; answers read aloud |
| 📈 **SkinTimeline** (USP) | re-photograph a spot with a ghost overlay; measures size/colour/shape change (coin for scale), escalates on real change, exports a Doctor Visit Summary PDF made on the phone |

## Architecture (short)
```
Photo ─► QualityGate ─► CNN (.tflite) ─┐
Voice ─► Gemma transcribe ─► extract ──┼─► RedFlagRules ─► Gemma 4 E2B (LiteRT-LM) ─► Validator ─► TierResolver ─► Result
Timeline ─► align/segment/metrics ─────┘                                                        └─► Doctor PDF
```
Details: [`docs/architecture.md`](docs/architecture.md). Data contracts: [`docs/contracts.md`](docs/contracts.md).

## Repository
```
CLAUDE.md          brief for the coding agent
docs/              plan · architecture · contracts · resources · test · diy · decisions
design/            Claude Design exports (screens, html, tokens.json) + brief
android/           Kotlin + Jetpack Compose app (flavors: offline / online)
ml/                data pipeline, CV training, Gemma LoRA, timeline & voice eval, evaluation
reports/           every metric in this project comes from a script that writes here
```

## Quick start
**Developers:** read [`docs/diy.md`](docs/diy.md) (§2–§5): Android Studio setup, phone connection, WSL, tokens (`.env.example`).
```bash
cp .env.example .env            # fill HF_TOKEN, KAGGLE_API_TOKEN, …
source .venv/bin/activate && pytest -q
cmd.exe /c "cd /d C:\dev\skinnova\android && gradlew.bat installOfflineDebug"
```
**Testers:** install `SkinNova-vX-offline.apk`, then *Import model file* → `skinnova-e2b-v1.litertlm` (~2.6 GB, one time). Works in airplane mode. Steps: `docs/diy.md §11`.

## Why the model isn't in the APK
LiteRT-LM loads the model from a file path. Bundling a 2.6 GB model would double storage use on first run and make the APK impractical to share. The engine is inside the APK; the model file is imported once.

## Evaluation
Four arms (base Gemma, fine-tuned Gemma, CNN only, combined), parity checks across every conversion (PyTorch→TFLite, LoRA→merged→`.litertlm`→phone), safety suites, and USP evaluations. See [`docs/test.md`](docs/test.md). Results: `reports/final_report.md`.

## Status
- [ ] Phase 0 de-risk · [ ] Data · [ ] CNN · [ ] LoRA · [ ] App core · [ ] USPs · [ ] Release

## Credits & licenses
Gemma 4 (Google, Apache-2.0) · LiteRT / LiteRT-LM (Google, Apache-2.0) · skintaglabs SigLIP classifier (MIT, used as a baseline) · datasets: see `docs/resources.md §3` for each license and attribution · condition notes summarized from AAD, NHS, DermNet (cited in-app).
