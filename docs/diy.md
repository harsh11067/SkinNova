# SkinNova — DIY Guide for Harsh (v2)

Opus writes code in WSL. You own: accounts, phones, Android Studio, Kaggle runs, design, medical-content review, recordings, and Go/No-Go decisions. Follow sections **in order**; each ends with a ✅ check.

**Master order**
```
§1 Mental model ─► §2 Windows+Android Studio ─► §3 Phone connection ─► §4 WSL+Opus setup ─► §5 Phase-0 spikes
 ─► §6 Data & medical content ─► §7 Kaggle training runs ─► §8 Design hand-off ─► §9 USP data collection
 ─► §10 Daily dev loop & debugging ─► §11 Release APK & team install ─► §12 PENDING checklist
```

---

## §1 Mental model — how code becomes an app on a phone

```
Kotlin/Compose code + res/ + assets/ + Gradle dependencies (LiteRT-LM, LiteRT, OpenCV, CameraX…)
        │  Gradle build (inside Android Studio, or gradlew)
        ▼
  .kt → bytecode → DEX;  native .so libs (arm64) merged;  assets packed
        ▼
  app-offline-debug.apk  (auto-signed with debug key)        app-offline-release.apk (signed with YOUR key)
        │  "Run ▶" / adb install                                   │ share file
        ▼                                                          ▼
  Phone installs app ─► app reads filesDir/models/skinnova-e2b-v1.litertlm ─► LiteRT-LM runs Gemma on CPU/GPU
```
| Term | Meaning |
|---|---|
| Gradle | build tool; `build.gradle.kts` lists dependencies. LiteRT-LM is downloaded by Gradle from Google Maven; you never install the engine by hand |
| Build variant | flavor × build type: `offlineDebug`, `offlineRelease`, `onlineDebug`, `onlineRelease` |
| APK | installable zip; ours is ~50–120 MB **without** the model |
| adb | USB/Wi-Fi bridge: install, push files, logs, shell |
| Logcat | live phone log; first place to look when something breaks |
| Compose | Kotlin UI toolkit; screens are `@Composable` functions |

**Why the model isn't inside the APK:** LiteRT-LM needs a real file path, so a bundled 2.6 GB model gets copied out on first run (~5.2 GB used). Huge APKs also fail to share or install reliably, and every code fix would force a 2.6 GB download. So: **APK (engine inside) + model file (imported once) = fully offline.**

---

## §2 Windows + Android Studio setup (≈ 1–2 h)

1. Install **Android Studio** (latest stable) from developer.android.com/studio. Default install.
2. First-run wizard → *Standard*. It installs the SDK, Platform-Tools and the Embedded JDK (JBR).
3. *More Actions → SDK Manager*:
   - **SDK Platforms:** latest stable Android API (and the one matching your phone's Android version).
   - **SDK Tools:** Android SDK Build-Tools (latest), Android SDK Platform-Tools, Android SDK Command-line Tools (latest), **Google USB Driver** (Windows; only needed for Pixel phones).
   - NDK/CMake: **not** needed.
4. Note the SDK path shown at the top (usually `C:\Users\<you>\AppData\Local\Android\Sdk`).
5. Add to Windows PATH (System → Environment Variables → Path → New):
   `C:\Users\<you>\AppData\Local\Android\Sdk\platform-tools`
   and `...\Sdk\build-tools\<latest>` (for `apksigner`, `aapt2`).
6. Open a new PowerShell: `adb version` → prints a version. ✅
7. Settings → Build, Execution, Deployment → Build Tools → Gradle → **Gradle JDK = Embedded JDK (jbr-17 or 21)**. Mismatched JDKs cause most first-sync errors.
8. Settings → Appearance & Behavior → System Settings → Memory Settings → IDE heap 4096 MB if you have ≥ 16 GB RAM.
9. Windows Defender: add exclusions for `C:\dev\skinnova`, `%USERPROFILE%\.gradle`, Android SDK folder. This often makes Gradle 2–3× faster.

✅ `adb version` works and Android Studio opens without errors.

---

## §3 Connecting the phone (USB, wireless, and from WSL)

### 3.1 Enable developer mode on the phone
Settings → About phone → tap **Build number** 7 times (Xiaomi: *MIUI/HyperOS version*; Realme/Oppo: *Version → Build number*). Then Settings → (System →) **Developer options**:
- **USB debugging** → ON
- **Install via USB** → ON *(Xiaomi/Redmi/POCO: requires signing into a Mi account and a SIM)*
- **USB debugging (Security settings)** → ON *(Xiaomi only; lets adb grant permissions/simulate taps for tests)*
- **Disable permission monitoring** → ON *(Oppo/Realme/OnePlus ColorOS if instrumentation tests fail with SecurityException)*
- **Stay awake** → ON (while charging; screen won't lock mid-test)
- Optional: **Don't keep activities** OFF (only turn ON when testing process death, A5)

### 3.2 USB connection
1. Use a **data** cable (many cheap cables are charge-only), preferably a USB 3 port.
2. Plug in → on the phone set USB mode to **File transfer / Android Auto** (some phones need this before adb works).
3. Phone shows "Allow USB debugging? RSA fingerprint" → tick **Always allow** → Allow.
4. PowerShell: `adb devices -l` → `XXXXXXXX device product:... model:...` ✅
| You see | Fix |
|---|---|
| nothing | different cable/port; install OEM USB driver (Samsung: "Samsung Android USB Driver"; others: OEM site; Pixel: Google USB Driver via SDK Manager → Device Manager → update driver → browse to `Sdk\extras\google\usb_driver`) |
| `unauthorized` | revoke: Developer options → *Revoke USB debugging authorizations*, replug, accept prompt |
| `offline` | `adb kill-server` → `adb start-server` → replug |
| `no permissions` | (Linux only) — use Windows adb instead |

### 3.3 Wireless debugging (Android 11+), handy for camera testing without a cable
1. Phone and PC on the same Wi-Fi (college Wi-Fi often blocks device-to-device traffic → use a phone hotspot or home router).
2. Developer options → **Wireless debugging** ON → *Pair device with pairing code* → shows `IP:PAIRPORT` + 6-digit code.
3. PowerShell: `adb pair 192.168.1.23:37xxx` → enter code → "Successfully paired".
4. Wireless debugging screen shows `IP address & Port` (different port) → `adb connect 192.168.1.23:4xxxx` ✅
5. Android Studio also offers *Pair Devices Using Wi-Fi* (QR) in the device dropdown.
⚠️ Use **USB** for pushing the 2.6 GB model; Wi-Fi push is slow.

### 3.4 Android Studio sees the phone
Toolbar device dropdown → your phone's name. Also open **View → Tool Windows → Device Manager** (physical tab) and **Logcat**. ✅

### 3.5 WSL (Opus) using the same phone
Only one adb server should own the USB device: the **Windows** one. From WSL, Opus calls the Windows binary:
```bash
# in WSL ~/.bashrc
alias adb='adb.exe'
adb devices -l        # same serial as PowerShell ✅
```
Don't install Linux `adb` and run it at the same time; two servers will fight over the device.

### 3.6 Letting Opus build/test without opening Android Studio (optional, useful)
Opus can call the Windows Gradle wrapper from WSL:
```bash
cmd.exe /c "cd /d C:\dev\skinnova\android && gradlew.bat testOfflineDebugUnitTest"
cmd.exe /c "cd /d C:\dev\skinnova\android && gradlew.bat assembleOfflineDebug"
cmd.exe /c "cd /d C:\dev\skinnova\android && gradlew.bat installOfflineDebug"
```
Rule: don't run Gradle from Opus while Android Studio is mid-sync/build (lock errors). Close or wait.

✅ T-ENV checks ENV-1 to ENV-9 in `test.md §2` pass → write `reports/env_check.md`.

---

## §4 WSL + repo + accounts + Opus setup (≈ 1 h)

```powershell
# PowerShell (once)
wsl --install -d Ubuntu      # if not installed
mkdir C:\dev; cd C:\dev      # unzip the project pack here → C:\dev\skinnova
```
Optional `C:\Users\<you>\.wslconfig` (needed for local `.litertlm` export, which needs a lot of RAM):
```
[wsl2]
memory=20GB
swap=16GB
networkingMode=mirrored
```
then `wsl --shutdown`.

```bash
# WSL
sudo apt update && sudo apt install -y git git-lfs python3-venv unzip build-essential libgl1
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
cd /mnt/c/dev/skinnova
git init && git config core.autocrlf input
mkdir -p ~/skinnova-data/{data,models} && ln -s ~/skinnova-data/data data && ln -s ~/skinnova-data/models models
cp .env.example .env && nano .env          # fill tokens (see below)
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision timm opencv-python-headless imagehash pydantic pandas scikit-learn \
  huggingface_hub kaggle python-dotenv pytest matplotlib
uv tool install litert-lm && uv tool install litert-torch-nightly
```

**Accounts & tokens**
| Service | Steps | Put in |
|---|---|---|
| Hugging Face | huggingface.co → Settings → Access Tokens → *New token* → type **Write** | `.env` `HF_TOKEN`, `HF_USER`; then `hf auth login --token $HF_TOKEN` |
| Gemma access | open huggingface.co/google/gemma-4-E2B-it → accept terms if shown | — |
| HF repos | create **private** model repos `skinnova-e2b-lora-v1`, `skinnova-e2b-merged`; private dataset `skinnova-sft-data` (or let notebook create) | `.env` repo names |
| Kaggle | kaggle.com → Settings → **Phone verification** (unlocks GPU + internet) → API → *Generate New Token* | new CLI: `KAGGLE_API_TOKEN` in `.env` or file `~/.kaggle/access_token`; legacy `~/.kaggle/kaggle.json` (`chmod 600`) still works |
| Kaggle secrets | in each notebook: Add-ons → Secrets → add `HF_TOKEN` | Kaggle UI |

**Starting Opus (Claude Code) in WSL**
```bash
cd /mnt/c/dev/skinnova && claude
```
First message: *"Read CLAUDE.md, then docs/plan.md, architecture.md, contracts.md, resources.md, test.md. Start Phase 0. Put anything you need from me in docs/diy.md §12."*

✅ ENV-10 to ENV-13 pass.

---

## §5 Phase 0 spikes (Week 1)
**5.1 Feel the model (30 min):** Play Store → *Google AI Edge Gallery* → download Gemma 4 E2B → *Ask Image*: photograph your forearm / a mosquito bite → "Describe this skin finding." Then *Audio* (if offered): say a Hindi sentence → transcribe. Write in `reports/phase0.md`: load time, seconds to first word, words/sec, phone heat, Hindi transcript quality (1–5).
**5.2 Desktop export (Opus):** watch for out-of-memory; if WSL can't, Opus generates `ml/llm/notebooks/export_check.ipynb` for a Kaggle CPU session (high RAM).
**5.3 First app on phone:**
1. Android Studio → *Open* → `C:\dev\skinnova\android` → wait for sync (first time 5–15 min).
2. Push stock model (USB): `adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.skinnova.app/files/models/` (launch the app once first so the folder exists; or `adb shell mkdir -p` that path).
3. Build Variants panel (left edge) → `offlineDebug` → Run ▶.
4. Pick a photo → Analyze → text streams. Crash → Logcat filter `package:com.skinnova.app level:error` → copy to Opus.
**5.4 Audio spike:** record 5 Hindi + 5 English clips (§9 script) → Opus's spike screen transcribes them → note quality.
✅ `reports/phase0.md` complete; fallbacks decided in `docs/decisions.md`.

---

## §6 Data & medical content (Week 2–3)
1. Download sources Opus can't script (Mendeley pages may need a browser click) → `~/skinnova-data/data/raw/<source>/`. Fill the license column in `resources.md §3`.
2. After Opus runs the pipeline: open `reports/data_card.md`. Check counts, then look at **20 random images per class** (Opus generates `reports/spotcheck/<class>.html` grids). Note mislabels.
3. **Condition cards** (≈ 3 h, your most important medical task): Opus drafts from AAD / NHS / DermNet into `ml/llm/cards/condition_cards.json`; you verify every sentence against the cited source, remove anything that sounds like a diagnosis or a prescription, and keep the language plain. Sign off in `docs/decisions.md`.
4. **SFT review** (≈ 2 h): Opus generates `reports/sft_review.html` with 50 random examples (image + prompt + target). Mark each OK/fix/reject. If > 10% are rejected, Opus fixes the generator.

---

## §7 Kaggle runs (CV + LoRA)
**Uploading data:** Opus prepares `~/skinnova-data/kaggle_upload/` + `dataset-metadata.json` → you run `kaggle datasets create -p ~/skinnova-data/kaggle_upload --dir-mode zip` (first time) / `kaggle datasets version -p ... -m "v2"` (updates). Keep it **private**.
**Running a notebook:**
1. kaggle.com → Code → New Notebook → File → Import → upload `ml/cv/notebooks/cv_train.ipynb` (or `ml/llm/notebooks/gemma4_e2b_lora.ipynb`).
2. Right panel: Accelerator **GPU T4 ×2** (or P100), Internet **On**, Add Input → your private dataset; Secrets → `HF_TOKEN`.
3. *Run All*. Do not close the tab during the first cells; long runs: *Save Version → Save & Run All (Commit)* runs in the background (up to 12 h).
4. Outputs: CV notebook → `/kaggle/working/skin_cls.tflite`, `cv_metrics.json`, `cv_parity.json` → download → `models/tflite/` and `reports/`. LoRA notebook → pushes to HF; download `train_E*.json` → `reports/`.
**LoRA run order (do not skip):** L1 cells print PASS → **L2 overfit run (10 min)** passes → E1 full run → read L3 curves → tell Opus "E1 done" → Opus runs L4 eval → then E2/E3/E4 only if worth it.
**Red flags while training:** loss in the hundreds (gradient-accumulation issue), CUDA OOM (lower `max_length` to 1536 or batch 1 + GA 8 already; keep gradient checkpointing "unsloth"), loss flat at the start value (masking wrong → L1 failed).
Weekly GPU quota ≈ 30 h; a full E2B LoRA run on ~4k examples should take roughly 1–3 h on T4 (measure E1, then plan).

---

## §8 Design hand-off (Claude Design → Compose)
1. Paste `design/DESIGN_BRIEF.md` into Claude Design. Iterate until all screens (incl. Timeline, Recapture overlay, Voice intake, Doctor summary preview) look right in light + dark.
2. Export into the repo:
   - `design/screens/<NN>_<name>_{light,dark}.png` at 412×915
   - `design/html/` (HTML/CSS export)
   - `design/tokens.json` (ask Claude Design: "output the design tokens as JSON: colors light/dark, typography scale, spacing, radii, elevation")
3. Tell Opus: "Implement `design/` — theme first from tokens.json, then screens in nav order. Compare against the PNGs."
4. Review side by side; give exact diffs ("Result card radius 16 not 12; triage banner icon left").
⚠️ Claude Design outputs HTML, not Android code. Opus translates it. Avoid web-only effects (backdrop blur, CSS filters on photos), custom fonts without a license, and tiny tap targets.

---

## §9 USP data collection (team, Week 5–7)
**Voice set (`voice_test`):** ≥ 6 speakers (different ages and genders), each records ≥ 20 clips (10 scripted from `ml/voice/scripts_<lang>.txt` + 10 free descriptions of a made-up skin problem), in a quiet room and with a fan/traffic noise. Use the debug *Recorder* screen (saves WAV + a JSON of the intended fields). Get **written consent** (template `ml/voice/consent.md`); no real health details needed: they act out scenarios. `adb pull /sdcard/Android/data/com.skinnova.app/files/voice/ data/raw/voice/`.
**Timeline set (`timeline_real`):** ≥ 10 unchanged spots (moles, freckles, scars, birthmarks) × 5 captures same day, with a coin beside the spot in 3 of the 5. Vary distance/lighting slightly as a real user would. Same consent. Debug *Timeline capture* screen.

---

## §10 Daily dev loop & debugging
```
Opus: writes code + tests → runs pytest / gradlew test (§3.6) → tells you what to run
You:  Android Studio → Sync (elephant icon) → Run ▶ on phone → try the feature
      problem? Logcat (filter: package:com.skinnova.app) → copy red lines + steps → paste to Opus
Opus: fixes → you re-run → commit at each green milestone:  git add -A && git commit -m "phase4: result screen"
```
| Tool | Where | Use |
|---|---|---|
| Logcat | bottom panel | crashes (`FATAL EXCEPTION`), engine logs (tag `EngineHolder`, `GemmaEngine`) |
| Device Explorer | View → Tool Windows | browse `/sdcard/Android/data/com.skinnova.app/files/`, drag-drop files |
| Profiler → Memory | bottom panel | watch RAM while the model loads; OOM investigations |
| Layout Inspector | Tools | Compose hierarchy when UI looks wrong |
| Build → Analyze APK | menu | what makes the APK big |

**adb cheat-sheet**
```bash
adb install -r app-offline-debug.apk          # reinstall keeping data
adb uninstall com.skinnova.app                 # fixes INSTALL_FAILED_UPDATE_INCOMPATIBLE
adb shell pm clear com.skinnova.app            # wipe app data
adb logcat -c && adb logcat --pid=$(adb shell pidof com.skinnova.app)
adb shell dumpsys meminfo com.skinnova.app     # RAM
adb shell dumpsys thermalservice | head -20    # heat
adb shell ls -l /sdcard/Android/data/com.skinnova.app/files/models/
adb exec-out screencap -p > shot.png           # screenshot for bug reports
adb shell cmd connectivity airplane-mode enable   # (Android 11+, may need root on some OEMs; else toggle manually)
```
**Common errors**
| Symptom | Likely cause → fix |
|---|---|
| Sync: `Could not resolve com.google.ai.edge.litertlm…` | `google()` missing in `settings.gradle.kts` repositories / offline Gradle mode on / typo in version |
| `Unsupported class file major version` | Gradle JDK mismatch → §2 step 7 |
| `UnsatisfiedLinkError` / GPU init error | manifest `<uses-native-library>` missing, or no OpenCL on phone → CPU fallback (automatic) |
| App closes while loading model, no stack trace | OOM killer → close other apps; CPU backend; 8 GB phone |
| `FileNotFoundException …litertlm` | wrong folder; check with `adb shell ls`; launch app once so the folder exists |
| `INSTALL_FAILED_INSUFFICIENT_STORAGE` | free space (model + APK + headroom ≈ 4 GB) |
| Very slow first analysis, fast later | normal: cold load + GPU kernel compile; `cacheDir` speeds up the 2nd load |
| Camera black in emulator | use the real phone; Gemma won't run well on the emulator anyway |

---

## §11 Release APK & team install
**Create release key (once):** Build → *Generate Signed App Bundle / APK* → **APK** → *Create new…* → save `C:\keys\skinnova-release.jks` (**outside the repo**), strong passwords → store passwords in a password manager **and** in `.env` (`SKINNOVA_KEYSTORE_*`) so Gradle can sign from the CLI. Losing the key = you can never update the app in place.
**Build:** Build Variants → `offlineRelease` → Build → Generate Signed APK → choose key → `android/app/offline/release/app-offline-release.apk` (or Opus: `gradlew.bat assembleOfflineRelease` with signing config reading env vars).
**Verify (G9–G11):**
```powershell
aapt2 dump permissions app-offline-release.apk     # no INTERNET
apksigner verify --print-certs app-offline-release.apk
```
Rename → `SkinNova-v1.0.0-offline.apk`.

**What a teammate receives:** `SkinNova-v1.0.0-offline.apk` + `skinnova-e2b-v1.litertlm` (Drive link / USB / pendrive) + these steps:
1. Copy both files to the phone's **Download** folder (≥ 6 GB free recommended).
2. Tap the APK → allow *Install unknown apps* for Files/Chrome → Install. (Play Protect warning "unknown developer" → *Install anyway*; expected for sideloaded apps.)
3. Open SkinNova → onboarding → **Import model file** → pick `skinnova-e2b-v1.litertlm` → wait for copy + "Verified ✓" (1–5 min). The Download copy can then be deleted.
4. Airplane mode ON → do one analysis, one voice intake, one "Track this spot" → fill the tester form (`test.md §12`).

---

## §12 PENDING FOR HARSH (Opus appends; you tick)
- [ ] §2 Android Studio + SDK + PATH + Gradle JDK
- [ ] §3 phone USB + wireless + WSL adb → `reports/env_check.md`
- [ ] §4 HF write token, Gemma terms, private repos, Kaggle phone-verify + token + HF_TOKEN secret
- [ ] Phone details in `resources.md §7`
- [ ] §5.1 AI Edge Gallery spike → `reports/phase0.md`
- [ ] §5.4 10 audio clips for spike
- [ ] §6 manual dataset downloads + licenses
- [ ] §6 data card + 20-per-class spot check
- [ ] §6 condition cards verified
- [ ] §6 50 SFT examples reviewed
- [ ] §7 CV notebook run
- [ ] §7 LoRA L2 overfit → E1 → (E2–E4)
- [ ] §8 Claude Design exports in `design/`
- [ ] §9 voice_test recordings + consents
- [ ] §9 timeline_real captures + consents
- [ ] §11 release keystore (outside repo) + passwords stored
- [ ] Field test with ≥ 5 teammates
