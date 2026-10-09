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
- LoRA v2 SHIPS (frozen llm_val 0.91 vs 0.90; pre-registered rule) — manifest `file` = skinnova-e2b-v2.litertlm.
- Design audit (2026-10-07): light-theme tier colours → WCAG AA 4.6:1, 4 touch targets → 48 dp, answer chips as in the
  design (Hindi fixed), setup shows the real model size, About states measured limits. "other" prior shift rejected on val.
- Android: compile (app + androidTest), JVM tests 16/16, lint 0 errors (2026-10-07 02:34). Gradle while an LLM eval runs:
  `./gradlew -q --no-daemon -Dorg.gradle.jvmargs="-Xmx2048m -Dfile.encoding=UTF-8" …` (fits beside the 4 GB engine).

## Results (2026-10-07) — `reports/final_report.md` has the S1–S11 table
- Arms on the frozen llm_test (shipped LoRA v2): D = image model alone on accuracy (78.7 % top-1 / 92 % top-3); with the
  shipped CV v2 scores 76.0 % / 89.3 %; JSON valid after one retry 100 %; under-triage HIGH/URGENT 0/32 in every arm;
  injection 30/30 safe. Safety S2–S4 PASS. S6 (LoRA vs base, real answers) CIs overlap → reported as no clear gain.
- CV v3 (Shades-of-Gray) NOT adopted by its pre-registered val rule (SCIN-val top-3 0.814 < 0.826). v2 stays.
- APKs: offline 91.9 MB arm64, no INTERNET (S11 static PASS); online has INTERNET for the one-time download.

- TL1 v8 (900 val pairs, same method): coin-pair area median 5.6 % (S9 PASS), p90 31 % (FAIL, disclosed; timeline rules
  only raise tiers). 6 of 7 timeline gates pass.

## Done 2026-10-08 (user's 8 requests; commit a2a3687)
- Result screen fixes (one-tap Save/Track, Learn more never blank), read-aloud on all care cards, "no clear match" card.
- Notifications + AnalysisService (foreground while writing); `EngineHolder.releaseIfIdle` (no mid-generation release).
- Profile (ProfileStore, AES-GCM Keystore) + AppLock (PIN/fingerprint, lock-out, re-lock delay, FLAG_SECURE, route guard).
- Home care & relief (`assets/care/relief.json`, `care/Relief.kt`, ReliefTest) — content *draft*, needs Harsh's review.
- Skin Library: 27 real photos (`ml/data/library_examples.py`, SCIN + PAD-UFES-20 CC BY 4.0).
- CV TTA V4 adopted (`reports/cv_tta.json`); skin gate v2 adopted (`reports/skin_gate_v2.json`); re-export C3 ✅;
  on-device C5 with lossless PNG fixtures 20/20, max Δp 3.4e-6 ✅ (`reports/device/bench/cv_parity_device.json`).
- Ask SkinNova (chat prompt v2, ChatSafety, `reports/chat_probe_v2.json` dev probe 30/30 guard-clean).
- Tests: JVM 32/32, pytest 154/154 (stale voice test updated for the topic check), on-device: C5 ✅, notification ✅.
- Pending on the phone (needs it unlocked): `NavFlowTest` (6 screen tests), A9/S8 bench.

## Done 2026-10-08 (2nd pass: bugs, auth review, model use; commits 3be5c69, a63d95e)
- App lock review: 14 fixes (PIN checked once at its length, off the main thread; fingerprint bound to an enrolment-
  invalidated Keystore key; current PIN to change/disable; no dialogs/keyboard/TTS/Back while locked; session cleared on
  reset). Profile: complete without skin tone, pre-fill note on the questions, saved off the main thread.
- Engine cache moved to cache/engine (Delete everything wiped it → 2-min loads); old caches cleaned (phone 1.0 → 4.5 GB free).
- vivo force-stops (exit-info) → battery-exemption prompt; Early look on the analysing screen; Read in Hindi on demand;
  Ask by voice; Hindi questions answered in Hindi; contagious facts per condition; GPU kept (measured vs CPU).
- On-device: all suites pass (FirstRun when run alone — vivo puts the user's Settings task on top in a combined run);
  S8 warm analysis 84 s (> 60 s budget, disclosed); Ask SkinNova 21–24 s per answer on GPU.

## Done 2026-10-08 (3rd pass; commits 43c2974…, release v0.2.0)
- Crash (gallery + permission prompts) = fragment 1.2.5 → 1.8.5; PIN locks right away; notification every time;
  shadows off on dark (median frame 20 → 14 ms); crisp dot-matrix landing + twinkle; Profile → Start screen; pixel
  rivets/markers. All trialled on the phone (gallery, PIN, notification, analysis). Device suite: 8 pass + 9 skipped to
  protect the tester's data; first-run passes and restores settings. GitHub Release v0.2.0 + docs/SETUP_GUIDE.md.

## Done 2026-10-09 — v2.1 (quickfix_v2_1.md) → v0.3.0
| Item | Result | Evidence |
|---|---|---|
| 1 Greedy decoding | adopted: phone 3/3 repeats byte-identical; PC 100/100 identical to the run 2 days earlier; phone vs PC 3/3 same categories/tier/uncertainty (text 90–100 %) | `reports/llm_litertlm_v21_2048.json`, `reports/device/bench/v21*.json*` |
| 2 Result first | adopted: result on screen 0.9–1.1 s (device test ×3), real trial < 2.8 s; Gemma swaps in place; tier never drops | `V21Tests.kt`, `progressive.json` |
| 3 Honest progress | adopted: pixel bar + "about N s left"; median ETA error at 50 % ≈ 19 % (≤ 25 %) | `v21.jsonl` |
| 4 KV 4096 → 2048 | NOT adopted (rule needs lower peak PSS: 2,734 vs 2,782 MB); budget p99 1,797 passes; llm_val unchanged | `token_budget.json`, `v21.jsonl` |
| 5 Optimising step | adopted: shown after an update/new model, then Home (15 s with an intact GPU cache, ~2 min fresh); warm-up at photo review | trial |
| 6 Phone calibration | NOT adopted (argmax/top-1/low-bucket conditions failed; NLL/ECE improved) | `cv_calibration_phone.json` |
| 7 Conformal short list | NOT adopted (needs 6.7 classes for 90 %, coverage 94.5 %) | `cv_calibration_phone.json` |
| 8 "Other" as an answer | adopted: card + tips + guided retake + doctor advice after 2 in a row | screenshots |
| 9 Capture coach + burst | adopted: green shutter + best-of-3 (simulated blur: top-1 60.9 → 65.4 %, top-3 = clean) | `capture_burst.json` |
| 10 Symptom fusion | NOT adopted (top-1 +0.7, top-3 +2.6, NLL −0.03 on phone holdout) | `fusion.json` |
Regression: pytest 154/154, JVM 37/37, on-device 19 OK (9 skipped to protect the tester's data), safety S2–S4 unchanged, offline APK no INTERNET.

## In progress
Nothing on the PC: every chain finished (`scripts/resume.sh` reports all done). Cleanup 2026-10-07 freed ~3 GB
(regenerable XNNPack caches of v1, duplicate Kaggle downloads, finished training resume states).

## Next — needs Harsh (d2y.md)
1. Phone (d2y §3): `scripts/device_tests.sh` (A1–A10, C5, TL3), `scripts/push_model.sh` (skinnova-e2b-v2.litertlm),
   `scripts/device_bench.sh` → S8 latency; airplane-mode run on 2 phones → S11.
2. Voice recordings (d2y §8) → `ml/eval/eval_voice.py` → S10.
3. GitHub push (d2y §10), optional kvm group for an emulator (d2y §11), hosting decision for the online flavor (d2y §12).
4. The real S1 fix: consented phone photos of brown skin (teach.md §6).
