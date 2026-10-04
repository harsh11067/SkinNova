# PROGRESS — resume checkpoint (read this first if a session restarts)

Repo: `/home/hash/mini/SkinNova/skinnova` (WSL native; C: drive is full, so NOT /mnt/c/dev).
Data/models: `~/skinnova-data/{data,models}` symlinked as `data/`, `models/`.
Python: `.venv` (uv, `--system-site-packages` → reuses system torch 2.11+cu130). Run `.venv/bin/python`.
Long jobs: always `setsid nohup … > logs/<name>.log 2>&1 &` (session end kills normal background jobs).

## Facts established (don't re-check)
- GPU: RTX 3050 laptop 6 GB VRAM; WSL RAM 7 GB; 16 cores; / has 889 GB free; C: 4.5 GB free.
- No java/adb/gradle in WSL. Windows Android SDK exists at /mnt/c/Users/*/AppData/Local/Android/Sdk.
- HF token works (user kumarharsh11067, fine-grained). Kaggle API token works (user harsh11067).
- skintaglabs/siglip-skin-lesion-classifier: SigLIP-so400m, 3.5 GB, 10 LESION classes only (mel/bcc/scc/ak/nevus/sk/df/vasc/non-neoplastic/other).
  → NOT sufficient for inflammatory classes and not on-device. Role = desktop baseline/teacher for lesion subset (plan §3).
- Datasets + licenses: mgmitesh Kaggle CC BY 4.0 (15 classes, heavily pre-augmented "- Copy"/"_0_NNNN" files → dedupe essential);
  PAD-UFES-20 CC BY 4.0 (6 lesion classes + metadata); SkinDisNet CC BY-NC 4.0 (6 inflammatory classes);
  SkinDiseaseBD CC BY-NC 4.0 (external test).
- Gap: no psoriasis / vitiligo source yet → need another dataset or merge/drop (decide after counts).
- Official design = `design/html/index.html` (copied from "SkinNova mobile app design (1)"): pixel-art night-sky theme,
  IBM Plex Mono + Pixelify Sans, dark default + light, screens: Loading, Home, Scan, Questions, Analyzing, Result,
  Insights, Library, Journey(History), Profile; bottom nav Home/History/Library/Profile. Tokens in THEMES const.

## Done
- [x] Read all docs (top-level + SkinNova/skinnova/docs v2 + design)
- [x] Repo dirs, symlinks, git init, `.env` copied (gitignored)
- [x] ml/common/paths.py, schema.py, labels.json (provisional = contracts §1)
- [x] scripts/fetch_data.sh (resumable) launched

## In progress / next
- [ ] pip deps (logs/pip.log) · data download (logs/fetch_data.log)
- [ ] safety core: ml/eval/redflags.py + tier resolver + ml/llm/validate.py + fixtures + pytest
- [ ] data pipeline: normalize → dedupe → label map → splits → data card
- [ ] CV train locally (GPU) → calibrate → tflite → parity
- [ ] cards + prompts + SFT builder + Kaggle LoRA notebook (pushed via Kaggle API)
- [ ] Android app (Compose, official design) — install JDK + cmdline-tools in WSL to build
- [ ] documentation.md, d2y.md, teach.md (at /home/hash/mini/)
