#!/usr/bin/env bash
# LoRA v1 export via the adapter route (export kernel v12, --mode lora) + LoRA v2 push. Restart-safe.
#  1. when the v1 export has STARTED (it has mounted the v1 SFT dataset): upload SFT v2 + push the LoRA v2 training run
#  2. when it finishes: stream the .litertlm (sha256 vs the kernel report) → local L6 parity vs the v1 training report
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python; KAG=~/.local/bin/kaggle; K=harsh11067/skinnova-gemma4-e2b-export
st() { $KAG kernels status "$1" 2>&1 | tail -1; }
until st $K | grep -qiE "RUNNING|COMPLETE|ERROR"; do sleep 60; done
if [ ! -f logs/lora_v2.pushed ]; then
  $PY -m pytest -q ml/tests/test_sft_data.py > /dev/null 2>&1 || { echo "L0 FAILED — not pushing LoRA v2"; exit 1; }
  $PY -m ml.llm.notebooks.make_kaggle dataset 2>&1 | grep -iE "success|error" | tail -1
  for i in $(seq 1 40); do $KAG datasets status harsh11067/skinnova-sft-data 2>&1 | tail -1 | grep -qi ready && break; sleep 20; done
  sleep 300   # let the new version propagate (arms v1 picked up a stale version right after "ready")
  $PY -m ml.llm.notebooks.make_kaggle push --mode full 2>&1 | grep -iE "pushed|error" && touch logs/lora_v2.pushed
fi
echo "[$(date +%T)] LORA_V2_PUSHED"
until st $K | grep -qiE "COMPLETE|ERROR|CANCEL"; do sleep 180; done
echo "[$(date +%T)] export: $(st $K)"
rm -rf reports/kaggle/export_lora && mkdir -p reports/kaggle/export_lora
$KAG kernels output $K -p reports/kaggle/export_lora --file-pattern '.*\.(json|log)$' >/dev/null 2>&1
R=reports/kaggle/export_lora/reports/export_lora.json
[ -f "$R" ] && python3 -c "import json;d=json.load(open('$R'));print({k:d.get(k) for k in ['merge','litertlm','capabilities','L6']}, 'module_diff', str(d.get('module_diff'))[:600])"
mkdir -p models/litertlm/skinnova; OUT=models/litertlm/skinnova/skinnova-e2b-v1.litertlm
if [ ! -f "$OUT.ok" ]; then
  ~/.local/share/uv/tools/kaggle/bin/python scripts/kaggle_output_url.py $K '\.litertlm$' > models/litertlm/skinnova/urls.tsv || exit 1
  [ -s models/litertlm/skinnova/urls.tsv ] || { echo "no .litertlm in the export output"; exit 1; }
  scripts/dl_url.sh "$(cut -f2 models/litertlm/skinnova/urls.tsv | head -1)" "$OUT"
  want=$(python3 -c "import json;print(json.load(open('$R'))['litertlm']['sha256'])"); have=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$want" = "$have" ] && touch "$OUT.ok" || { echo "sha mismatch $have vs $want"; exit 1; }
fi
[ -f reports/parity_l6_skinnova_v1.json ] || .venv-export/bin/python -m ml.eval.parity_l6 --model "$OUT" \
    --hf-report reports/kaggle/full/reports/full.json --tag skinnova_v1 --threads 12 2>&1 | grep -vE "^(INFO|WARNING|W0000|I0000|E0000)"
echo "[$(date +%T)] AFTER_LORA_V1_DONE"
