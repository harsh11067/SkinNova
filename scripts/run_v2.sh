#!/usr/bin/env bash
# Phase v2 (2026-10-06): after the stock comparison → SCIN data pipeline (frozen v1 splits) → data tests → CV v2 training
# → calibration → val metrics → (adoption rule on val, decisions.md) → test/external once → robustness.
# Every stage is restart-safe (incremental pipeline, --resume training); re-run this script after a WSL restart.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MALLOC_ARENA_MAX=2
until grep -q COMPARE_DONE logs/compare_stock.log 2>/dev/null; do sleep 30; done
if [ ! -f logs/v2_pipeline.done ]; then
  echo "[$(date +%T)] sources";    $PY -m ml.data.sources || exit 1
  echo "[$(date +%T)] label map";  $PY -m ml.data.build_label_map || exit 1
  echo "[$(date +%T)] normalize";  $PY -m ml.data.normalize --workers 8 || exit 1
  echo "[$(date +%T)] embed";      $PY -m ml.data.embed_dedupe embed || exit 1
  echo "[$(date +%T)] pairs";      $PY -m ml.data.embed_dedupe pairs || exit 1
  echo "[$(date +%T)] dedupe";     $PY -m ml.data.dedupe || exit 1
  echo "[$(date +%T)] splits";     $PY -m ml.data.make_splits || exit 1
  echo "[$(date +%T)] data card";  $PY -m ml.data.data_card > /dev/null || exit 1
  echo "[$(date +%T)] data tests"; $PY -m pytest -q ml/tests/test_data.py 2>&1 | tail -3
  $PY -m pytest -q ml/tests/test_data.py > /dev/null 2>&1 || { echo "DATA TESTS FAILED — stop before training"; exit 1; }
  touch logs/v2_pipeline.done
fi
echo "[$(date +%T)] CV v2 training"
[ -f reports/cv_train_full_v2.json ] || $PY -m ml.cv.train --epochs 25 --resume --tag _v2 2>&1 | grep -E --line-buffered "epoch|DONE|resumed|Error|Traceback"
[ -f reports/cv_train_full_v2.json ] || { echo "training did not finish"; exit 1; }
[ -f reports/cv_calibration_v2.json ] || $PY -m ml.cv.calibrate --tag _v2
echo "[$(date +%T)] val metrics v2"; $PY -m ml.cv.eval_cv --ckpt models/cv/ckpt/best_v2.pt --splits val --tag _v2_val
echo "[$(date +%T)] V2_TRAINED"
