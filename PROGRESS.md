# PROGRESS — resume checkpoint (read this first if a session restarts)

Repo: `/home/hash/mini/SkinNova/skinnova` (WSL native; C: drive is full, so NOT /mnt/c/dev). Data/models symlinked from `~/skinnova-data`.
Python: `.venv` (training, CUDA torch) and `.venv-export` (litert-torch 0.9.4 / litert-lm-api 0.17.1, CPU torch).
Long jobs: always `setsid nohup … > logs/<name>.log 2>&1 < /dev/null &` — WSL restarts and session ends kill everything else.
Gradle from WSL: `cd android && JAVA_HOME=~/android/jdk ANDROID_HOME=~/android/sdk ./gradlew …` (Android Studio is open on
`/home/hash/mini` as a plain folder, not the Gradle project). adb: Windows `adb.exe` via `scripts/_adb.sh` (phone not attached yet).
Kaggle CLI: `~/.local/bin/kaggle` (uv tool 2.2.4), env from `.env`.

## Done (verified on disk, 2026-10-06)
- Data pipeline, verified dedupe, splits (train 18,805 / val 3,754 / test 3,759 / external 136), data card — `reports/`.
- CV v1 EfficientNet-B0: val macro-F1 0.747, **test 0.757**, top-3 97.4 %; **external SkinDiseaseBD top-1 16 %** (domain shift:
  brown-skin phone photos → "other"). T = 0.876. `.tflite` C3 pass (3.2e-5), assets + manifest + C4/C5 fixtures written.
- Android: builds, JVM tests 16/16 green, offline APK has no INTERNET (91.9 MB). Commit 2b4b746.
- Gemma: smoke v3–v5 diagnosed (Unsloth template repr bug; random paraphrases; collator resize). Notebook now trains with the
  **LiteRT-LM runtime template** (`ml/llm/chat_template_litertlm.jinja`, from Google's official .litertlm).
- Export: `export_patched.py` makes litert-torch export fit 32 GB; v10 = stock export runs (vision ✓, audio ✗), but our file
  lacks `end_of_vision` and is worse/slower than the official file (valid 0.4 vs 0.9, 116 vs 65 s/case on Kaggle CPU).

## In progress
- Kaggle `skinnova-gemma4-e2b-lora` **v7 full LoRA run** (v6 was hard-killed with no log → lazy images + RAM guard added).
- Download of export v10 `.litertlm` → `models/litertlm/export_v10/` (logs/dl_export_v10.log) for the hybrid test.
- SCIN download (CC BY 4.0, 6,506 images) → `data/raw/scin/images` (resumable: `data/raw/scin/download.sh`).

## Next (in order)
1. Hybrid `.litertlm`: official sections (`models/litertlm/stock/sections/`, rebuilt OK with litert-lm-builder) + OUR decoder
   (`prefill_decode`) → test locally (16-core CPU is ~2× Kaggle). If good: same for the merged LoRA decoder (keeps official
   vision/audio → Gemma voice back). Needs module_diff to confirm LoRA touched only the decoder.
2. After v7: merged export (`make_export --mode merged`) → L5/L6 → hybrid → desktop eval (eval_llm) → arms A–D, safety S4/S5/S9.
3. SCIN → label map (explicit condition table) → pipeline (freeze existing splits, add SCIN groups) → CV v2 → external re-test.
4. Android: LLM image at 512 px like training; manifest sha for the SkinNova model; instrumentation tests when phone attached.
5. Keep documentation.md / d2y.md / teach.md (in /home/hash/mini) current.
