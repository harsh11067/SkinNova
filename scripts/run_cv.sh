#!/usr/bin/env bash
# CV classifier: sanity checks → full training → calibration → val metrics → probability dumps.
# Idempotent and restart-safe: each stage is skipped when its output exists; training resumes from last_full.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q PIPELINE_DONE logs/data_pipeline.log 2>/dev/null; do sleep 15; done
echo "[$(date +%T)] data tests"
$PY -m pytest -q ml/tests/test_data.py 2>&1 | tail -3
[ -f reports/cv_train_overfit64.json ] || { echo "[$(date +%T)] C1a overfit64"; $PY -m ml.cv.train --overfit64 --epochs 30 --bs 16 --workers 4 2>&1 | grep -E "DONE|Error|Traceback" ; }
[ -f reports/cv_train_random_labels.json ] || { echo "[$(date +%T)] C1b random labels"; $PY -m ml.cv.train --random-labels --epochs 3 2>&1 | grep -E "DONE|Error|Traceback" ; }
[ -f reports/cv_train_full.json ] || { echo "[$(date +%T)] full training"; $PY -m ml.cv.train --epochs 25 --resume 2>&1 | grep -E --line-buffered "epoch|DONE|resumed|Error|Traceback" ; }
[ -f reports/cv_calibration.json ] || { echo "[$(date +%T)] calibrate"; $PY -m ml.cv.calibrate; }
echo "[$(date +%T)] eval val"; $PY -m ml.cv.eval_cv --splits val
echo "[$(date +%T)] predict";  $PY -m ml.cv.predict --splits val test external_test
echo "[$(date +%T)] CV_DONE"
