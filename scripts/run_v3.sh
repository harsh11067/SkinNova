#!/usr/bin/env bash
# CV v3 experiment (decisions.md 2026-10-07, pre-registered): train with Shades-of-Gray → calibrate → val probabilities →
# adoption rule on val only. Never touches test/external. Restart-safe (training resumes from last_v3.pt).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f models/cv/ckpt/best_v3.pt ] && grep -q V3_TRAINED logs/run_v3.log 2>/dev/null || {
  MALLOC_ARENA_MAX=2 $PY -m ml.cv.train --tag _v3 --color-constancy sog6 --workers 2 --resume && echo "[$(date +%T)] V3_TRAINED"; }
$PY -m ml.cv.calibrate --tag _v3 | tail -2
$PY -m ml.cv.predict --ckpt models/cv/ckpt/best_v3.pt --splits val --suffix _v3 | tail -2
$PY -m ml.cv.adopt_v3 | tail -12
echo "[$(date +%T)] RUN_V3_DONE"
