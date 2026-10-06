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

## In progress — chained setsid scripts (re-run any of them after a WSL restart; all restart-safe)
| script | waits for | does | log |
|---|---|---|---|
| scripts/run_v2.sh | — | CV v2 training (`--tag _v2`) → calibrate → val metrics → V2_TRAINED | logs/run_v2.log |
| scripts/v2_finish.sh | V2_TRAINED | adoption rule (ml/cv/adopt_v2.py, val only) → if ADOPT: test/external once, robustness, safety, promote best_v2 → best.pt, .tflite → TL1 v3 | logs/v2_finish.log |
| watcher (bash -c in logs/watch_full.log) | Kaggle LoRA v8 | fetch reports → FETCHED | logs/watch_full.log |
| scripts/after_lora.sh | FETCHED | merged export on Kaggle → stream .litertlm → (after V2_TRAINED) local L6 | logs/after_lora.log |
| scripts/lora_v2.sh | V2_FINISH_DONE + export started | SFT v2 (SCIN) → dataset → LoRA v2 push | logs/lora_v2.log |
| watcher (logs/watch_arms.log) | Kaggle arms kernel | arm A (stock) on frozen data/llm_eval → reports/arms/A.jsonl | logs/watch_arms.log |
Frozen eval sets: data/llm_eval/{llm_test,llm_val}.jsonl + images (never rebuilt).

## Next (in order)
1. After v8: `make_export --mode merged` on Kaggle (module_diff + L6 vs HF E1) → download via scripts/kaggle_output_url.py +
   scripts/dl_url.sh → local `ml.eval.parity_l6` (threads 12) → fill manifest sha → arms A–D + safety on llm_test.
2. After V2_TRAINED: apply adoption rule; if adopted: eval test/external once (`eval_cv --ckpt best_v2.pt --tag _v2`),
   robustness `--tag _v2`, export .tflite from best_v2 (export_tflite reads best.pt → copy/rename after adoption), SFT rebuild
   with SCIN real_q → consider LoRA v2.
3. Android (after ML stable): device tests A1–A10 on the phone, perf bench, design check; .wslconfig (d2y §2).
