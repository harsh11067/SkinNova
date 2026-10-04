# CLAUDE.md — SkinNova brief for the coding agent (read first, every session)

You are the coding agent for **SkinNova**: an offline Android app that takes a skin photo + symptoms (tapped or **spoken in Hindi/English**) and returns *preliminary, educational* information: top-3 condition categories, uncertainty, explanation, triage. It also **tracks a spot over time** (SkinTimeline). All inference runs on-device.

Read in order: `docs/plan.md` → `docs/architecture.md` → `docs/contracts.md` → `docs/resources.md` → `docs/test.md`. `docs/diy.md` is Harsh's manual; read it to know which steps are his. Log decisions in `docs/decisions.md`.

## Non-negotiables
1. **On-device inference only.** The `offline` flavor declares no INTERNET permission. Only the `online` flavor may download the model, once.
2. **Safety is deterministic code.** Rules R1–R8, T1–T3 (`contracts.md §4`) run before and after the LLM; `TierResolver` is monotone; the LLM can never lower a tier or remove a red flag. Safety strings come from `strings.xml`, never from the LLM.
3. **Never a single diagnosis.** LLM output is validated JSON (`contracts.md §3`); one repair retry, then Basic-mode fallback (CV + cards + rules).
4. **Voice intake never guesses.** Extracted fields need evidence quotes present in the transcript; failing fields are null; the user confirms every field.
5. **Model file is not in the APK.** Engine via Gradle; `.litertlm` imported via SAF/adb into app storage, sha256-verified against `assets/model_manifest.json`.
6. **No leakage.** Dedupe (pHash) before group split; tune on val only; test runs once per final candidate.
7. **Same prompts for training and app.** `ml/llm/prompts/` ≡ `android/app/src/main/assets/prompts/` (test enforces).
8. **Every number comes from a script** in `ml/eval/` writing to `reports/` with `{git_sha, dataset_rev, model_sha}`.
9. **Secrets** only in `.env` / `~/.kaggle/` / Kaggle Secrets. Never commit `.env`, keystores, tokens.
10. **Parity at every conversion** (test.md L5–L7, C3–C5). An artifact without its parity report isn't shippable.

## Environment
- WSL Ubuntu; repo at `/mnt/c/dev/skinnova` (= `C:\dev\skinnova`, opened by Android Studio on Windows). Heavy data in `~/skinnova-data`, symlinked as `data/`, `models/`.
- adb: use Windows `adb.exe` (alias `adb`). Never start a Linux adb server.
- Gradle from WSL: `cmd.exe /c "cd /d C:\dev\skinnova\android && gradlew.bat <task>"`. Don't run while Android Studio is syncing.
- GPU work = Kaggle notebooks you write in `ml/**/notebooks/`; Harsh runs them. Each notebook: asserts/PASS cells, writes JSON reports, pushes to HF repos named in `.env`.
- Python env: `.venv` via uv; `litert-lm`, `litert-torch-nightly` as uv tools.

## Repo layout (create exactly)
```
skinnova/
├── CLAUDE.md  README.md  .env.example  .gitignore
├── docs/      plan.md architecture.md contracts.md resources.md test.md diy.md decisions.md
├── design/    DESIGN_BRIEF.md screens/ html/ tokens.json
├── tests/fixtures/  redflag_cases.json validator_cases.json tier_cases.json intake_cases.json metrics_cases.json
├── android/   (packages per architecture.md §2; flavors offline/online; arm64 only)
│   └── app/src/main/assets/  cv/{skin_cls.tflite,labels.json,preprocess.json}
│                             prompts/*.txt  prompts/condition_cards.json  safety/rx_terms.txt  model_manifest.json
├── ml/        common/ data/ cv/ llm/ timeline/ voice/ eval/   (architecture.md §11)
├── scripts/   device_bench.sh push_model.sh pull_bench.sh
├── data/ models/   (symlinks, gitignored)
└── reports/
```

## How to work
- Follow phases in `plan.md §9`; respect exit gates. If blocked, record in `decisions.md`, switch to the listed fallback, continue.
- Phase 0 spikes first: no UI until an image (and an audio clip) goes through LiteRT-LM on the phone.
- Need Harsh? Append exact steps/commands to `diy.md §12` and keep working on something else.
- Write tests with the code (test.md §0). Deterministic logic gets shared JSON fixtures run by pytest and JUnit.
- Pin versions in `resources.md §7` once something works. If an API sample here fails, check the official doc linked in resources and update the doc.
- UI: implement `design/` in Compose. Theme from `tokens.json` first, then screens in nav order; compare with PNGs.
- Commit at each green milestone with a clear message.
