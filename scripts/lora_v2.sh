#!/usr/bin/env bash
# LoRA v2 (rule pre-registered in decisions.md): SFT rebuilt with SCIN phone photos + real answers and the shipped CV's
# probabilities → L0 tests → Kaggle dataset → full run, same config as v1. Waits until (1) CV v2 is decided and finished
# (logs/v2_finish.log: V2_FINISH_DONE) and (2) the v1 merged export has STARTED on Kaggle — it mounts the training
# kernel's latest output at start, so pushing v2 earlier would hand it the wrong weights. Restart-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python; KAG=~/.local/bin/kaggle
until grep -q V2_FINISH_DONE logs/v2_finish.log 2>/dev/null; do sleep 60; done
until [ -f logs/merged_export.pushed ] && $KAG kernels status harsh11067/skinnova-gemma4-e2b-export 2>&1 | tail -1 | grep -qiE "RUNNING|COMPLETE"; do sleep 60; done
if [ ! -f logs/sft_v2.done ]; then
  echo "[$(date +%T)] SFT v2 build"; MALLOC_ARENA_MAX=2 $PY -m ml.llm.build_sft_dataset --n 3200 2>&1 | tail -3 || exit 1
  $PY -m pytest -q ml/tests/test_sft_data.py ml/tests/test_assets_sync.py 2>&1 | tail -2
  $PY -m pytest -q ml/tests/test_sft_data.py > /dev/null 2>&1 || { echo "L0 FAILED"; exit 1; }
  $PY - <<'PY'
import json; d = json.load(open("reports/sft_data.json")); print("SFT v2:", d["n"], "real_q_share", round(d["real_q_share"], 3))
PY
  touch logs/sft_v2.done
fi
if [ ! -f logs/lora_v2.pushed ]; then
  $PY -m ml.llm.notebooks.make_kaggle dataset 2>&1 | grep -iE "success|error" | tail -1
  for i in $(seq 1 40); do $KAG datasets status harsh11067/skinnova-sft-data 2>&1 | tail -1 | grep -qi ready && break; sleep 20; done
  $PY -m ml.llm.notebooks.make_kaggle push --mode full 2>&1 | grep -iE "pushed|error" && touch logs/lora_v2.pushed
fi
echo "[$(date +%T)] LORA_V2_PUSHED"
