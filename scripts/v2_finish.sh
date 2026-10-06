#!/usr/bin/env bash
# After CV v2 training (logs/run_v2.log: V2_TRAINED): adoption rule on VAL → if adopted: test/external once, robustness,
# safety, promote best_v2 → best.pt (v1 kept as best_v1.pt), .tflite export + parity → TL1 v3 timeline eval. Restart-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MALLOC_ARENA_MAX=2
until grep -q V2_TRAINED logs/run_v2.log 2>/dev/null; do sleep 60; done
C=models/cv/ckpt
[ -f $C/best_v1.pt ] || cp $C/best.pt $C/best_v1.pt
[ -f data/processed/cv_probs_val_v1on2.npz ] || $PY -m ml.cv.predict --ckpt $C/best_v1.pt --splits val --suffix _v1on2
[ -f data/processed/cv_probs_val_v2.npz ] || $PY -m ml.cv.predict --ckpt $C/best_v2.pt --splits val --suffix _v2
D=$($PY -m ml.cv.adopt_v2 | tail -1)
echo "[$(date +%T)] adoption: $D"
if [ "$D" = "ADOPT" ]; then
  [ -f reports/cv_metrics_v2.json ] || $PY -m ml.cv.eval_cv --ckpt $C/best_v2.pt --splits val test external_test --tag _v2
  [ -f reports/cv_robustness_v2.json ] || $PY -m ml.cv.robustness --ckpt $C/best_v2.pt --tag _v2
  $PY -m ml.cv.predict --ckpt $C/best_v2.pt --splits test external_test --suffix _v2
  $PY -m ml.eval.safety_report --probs-tag _v2
  if ! cmp -s $C/best.pt $C/best_v2.pt; then cp $C/best_v2.pt $C/best.pt; fi
  CUDA_VISIBLE_DEVICES="" .venv-export/bin/python -m ml.cv.export_tflite 2>&1 | grep -vE "I0000|WARNING|Failed to load|oneDNN|absl|ERROR\]" | tail -4
  # the SFT data and the app read the probabilities of the shipped model
  for s in val test external_test; do cp data/processed/cv_probs_${s}_v2.npz data/processed/cv_probs_${s}.npz; done
fi
echo "[$(date +%T)] TL1 v3"
[ -f reports/timeline_eval_v3.done ] || { $PY -m ml.timeline.eval_timeline --n 300 --workers 6 2>&1 | grep -A12 '"gates"'; touch reports/timeline_eval_v3.done; }
echo "[$(date +%T)] V2_FINISH_DONE"
