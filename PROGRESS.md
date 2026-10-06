# PROGRESS — resume checkpoint (read this first if a session restarts)

Repo: `/home/hash/mini/SkinNova/skinnova` (WSL native; C: drive is full, so NOT /mnt/c/dev). Data/models symlinked from `~/skinnova-data`.
Python: `.venv` (training, CUDA torch) and `.venv-export` (litert-torch 0.9.4 / litert-lm-api 0.17.1, CPU torch).
Long jobs: always `setsid nohup … > logs/<name>.log 2>&1 < /dev/null &` — WSL restarts and session ends kill everything else.
Never `pkill -f`/`pgrep -f` a pattern that also appears in your own command line (exit 144 self-kill): find PIDs, then kill.
Gradle from WSL: `cd android && JAVA_HOME=~/android/jdk ANDROID_HOME=~/android/sdk ./gradlew …`; stop the daemon after
(`./gradlew --stop`, it holds ~4 GB of the 7.8 GB WSL RAM). adb: Windows `adb.exe` via `scripts/_adb.sh` (phone not attached yet).
Kaggle CLI: `~/.local/bin/kaggle`, env from `.env`. Big Kaggle outputs: `scripts/kaggle_output_url.py` + `scripts/dl_url.sh`.
One-line status: `scripts/status.sh`.
**After a PC / WSL restart: `cd ~/mini/SkinNova/skinnova && scripts/resume.sh`** (relaunches only unfinished chains).

## Done (verified on disk, 2026-10-06)
- Data: v2 with SCIN (CC BY 4.0) — frozen v1 splits kept, 0 v1 images moved; data card. PacificRM `Unknown_Normal` (1,471,
  mostly non-skin objects) is mapped to `other` by design (plan T5 / test S5: non-skin → other + high uncertainty).
- CV v2 ADOPTED (val-only pre-registered rule): external top-3 66.2 % (v1 28.7 %), S1 gate 75 % still FAIL (disclosed);
  `.tflite` C3 pass; S2 91.9 %, S3 60/60. `reports/cv_metrics_v2.json`, `cv_v2_adoption.json`, `safety.json`.
- Gemma LoRA v1 (Kaggle full run v8) → adapter dataset `harsh11067/skinnova-lora-v1` → export kernel v14 (streaming manual
  merge W += (α/r)·B·A, 494 modules) → `models/litertlm/skinnova/skinnova-e2b-v1.litertlm` (3.86 GB, sha256 30064f2c…,
  verified against the kernel report; text + vision). The kernel's own greedy suite OOM'd after the export (not needed).
- Arms A (stock .litertlm, no CV) on frozen llm_test n=150 (Kaggle CPU): top-1 56 %, top-3 66 %, JSON valid 76.7 %,
  under-triage HIGH/URGENT 0/32 → `reports/arms/A.jsonl`.
- SkinTimeline (TL1 v7b, `reports/timeline_eval.json`, 6 of 7 gates): mean-only skin re-lighting; coin detector v2 +
  coin-aware GrabCut (coin found in both photos 24 → 92 %); Otsu colour-split GrabCut init (train-chosen) → coin-pair area
  median 5.7 % ✅, p90 25 % ❌ (n 29), light 1.04 / colour 1.08 ΔE ✅. TL1 generator pinned to the v2 disc segmentation
  (the first v7 comparison was circular). Python ≡ Kotlin ported (commit d9d2c13).
- TL3 fixtures (`tests/fixtures/metrics_cases.json` + androidTest `assets/tl3/`) and `TimelineParityTest` (on-device).
- Voice intake topic check (Python + Kotlin, shared fixture 24 cases): off-topic evidence quotes dropped (commit 30979fc).
- LoRA v1 selection run on frozen llm_val: category agreement 0.90, JSON valid 1.0 (`reports/llm_litertlm_select_v1.json`).
- Android: compile (app + androidTest), JVM tests 16/16, lint 0 errors (2026-10-07 02:34). Gradle while an LLM eval runs:
  `./gradlew -q --no-daemon -Dorg.gradle.jvmargs="-Xmx2048m -Dfile.encoding=UTF-8" …` (fits beside the 4 GB engine).

## In progress
| what | where | log |
|---|---|---|
| LoRA v2 (still ignores the photo) → export v15 → L6 PASS (agreement drop 2.5 pts) → v1/v2 selection on frozen llm_val (100, greedy; ties → v1; v1 = 0.90) | `scripts/after_lora_v2.sh` | logs/after_lora_v2.log |
| arms D C B on the selected model (frozen set) + D C on the CV v2 re-render → safety → final report (~6 h, resumes per case) | `scripts/run_arms.sh` | logs/run_arms.log |

## Next (in order)
1. ✅ L6 v1 PASS → manifest filled (sha 30064f2c…).
2. ✅ Android compile + JVM tests + lint → commits d9d2c13, 30979fc.
3. ✅ Selection: v2 0.91 vs v1 0.90 (one case; pre-registered rule) → **v2 ships** (manifest `file` = skinnova-e2b-v2.litertlm; v1 still accepted). ModelManager fixed: same-size models matched by sha.
4. Arms D, C, B locally on the shipping model (`eval_arms --skinnova <model> --arms D C B --n 150`) → safety S4 → final_report.
5. Phone (Harsh, d2y §3): `scripts/device_tests.sh` (A1–A10, C5, TL3), `push_model.sh`, `device_bench.sh`.
