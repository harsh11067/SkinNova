#!/usr/bin/env bash
# After CV training: probabilities → full SFT data (+L0) → Kaggle dataset version → .tflite export + C3 → app tests/APK → TL1.
# The full LoRA Kaggle run is pushed separately, only after the smoke test passes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q CV_DONE logs/cv.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] full SFT build";   MALLOC_ARENA_MAX=2 $PY -m ml.llm.build_sft_dataset --n 3200
echo "[$(date +%T)] L0 tests";         $PY -m pytest -q ml/tests 2>&1 | tail -2
echo "[$(date +%T)] kaggle dataset";   set -a; . ./.env; set +a; $PY -m ml.llm.notebooks.make_kaggle dataset 2>&1 | tail -2
echo "[$(date +%T)] tflite export";    CUDA_VISIBLE_DEVICES="" .venv-export/bin/python -m ml.cv.export_tflite 2>&1 | grep -vE "I0000|WARNING|Failed to load|oneDNN|absl" | tail -6
echo "[$(date +%T)] sync + JUnit + APK"
$PY scripts/sync_assets.py > /dev/null
( cd android && JAVA_HOME=~/android/jdk ANDROID_HOME=~/android/sdk ./gradlew -q :app:testOfflineDebugUnitTest :app:assembleOfflineDebug :app:assembleOnlineDebug 2>&1 | tail -5; JAVA_HOME=~/android/jdk ./gradlew --stop >/dev/null )
echo "[$(date +%T)] TL1 v3";           $PY -m ml.timeline.eval_timeline --n 300 --workers 6 2>&1 | grep -A9 '"gates"' || true
echo "[$(date +%T)] AFTER_CV_DONE"
