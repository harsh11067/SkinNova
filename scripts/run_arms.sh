#!/usr/bin/env bash
# Arms on the shipping SkinNova model, after the v1/v2 selection (scripts/after_lora_v2.sh). Restart-safe (eval_arms resumes).
#  1. frozen arms set (data/llm_eval; arm A already ran on Kaggle): D, C, B → reports/arms/, reports/llm_arms.json
#  2. the same records re-rendered with the SHIPPED image model (ml/eval/rescore_cv.py): D, C → reports/arms_cv_v2/
#  3. safety summary (S4 from arm D) + final report
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; PYX=.venv-export/bin/python; D=models/litertlm/skinnova
until grep -q AFTER_LORA_V2_DONE logs/after_lora_v2.log 2>/dev/null; do sleep 120; done
SHIP=$(grep -o "→ SHIP v[12]" logs/after_lora_v2.log | tail -1 | grep -o "v[12]")
[ -n "$SHIP" ] || { echo "no selection result in logs/after_lora_v2.log"; exit 1; }
M=$D/skinnova-e2b-$SHIP.litertlm
echo "[$(date +%T)] shipping model $SHIP: $M"
F='^(INFO|WARNING|W0000|I0000|E0000)'
$PYX -m ml.eval.eval_arms --skinnova "$M" --arms D C B --n 150 --threads 12 2>&1 | grep -vE "$F" | tail -30
echo "[$(date +%T)] ARMS_FROZEN_DONE"
[ -f data/llm_eval_cv_v2/llm_test.jsonl ] || $PY -m ml.eval.rescore_cv --suffix _v2
$PYX -m ml.eval.eval_arms --skinnova "$M" --arms D C --n 150 --threads 12 --test-dir data/llm_eval_cv_v2 --tag _cv_v2 \
    2>&1 | grep -vE "$F" | tail -30
echo "[$(date +%T)] ARMS_CV_V2_DONE"
$PY -m ml.eval.safety_report --probs-tag _v2 2>&1 | tail -5
$PY -m ml.eval.final_report > /dev/null && echo "final report → reports/final_report.md"
echo "[$(date +%T)] RUN_ARMS_DONE"
