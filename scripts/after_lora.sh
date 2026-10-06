#!/usr/bin/env bash
# After the full LoRA run: merged export on Kaggle (module_diff + L6 there) → stream the .litertlm here → local L6 parity
# (waits for local CV v2 training to finish first: WSL has 7.8 GB and has crashed under parallel load). Restart-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
KAG=~/.local/bin/kaggle
until grep -q FETCHED logs/watch_full.log 2>/dev/null; do sleep 60; done
R=reports/kaggle/full/reports/full.json
[ -f "$R" ] || { echo "no full.json — training run failed; see reports/kaggle/full"; exit 1; }
python3 - "$R" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
assert "train" in d and "L4_analysis" in d, "training/eval incomplete: " + ", ".join(d)
print("train", {k: d["train"][k] for k in ["wall_h", "best", "peak_mem_gb"]}); print("L4", json.dumps(d["L4_analysis"]))
print("L5", {k: v for k, v in d.get("L5_merge_parity", {}).items() if k != "rows"})
PY
if [ ! -f logs/merged_export.pushed ]; then
  .venv/bin/python -m ml.llm.notebooks.make_export --mode merged 2>&1 | grep -iE "pushed|error" && touch logs/merged_export.pushed
  sleep 600
fi
until $KAG kernels status harsh11067/skinnova-gemma4-e2b-export 2>&1 | tail -1 | grep -qiE "COMPLETE|ERROR|CANCEL"; do sleep 180; done
$KAG kernels status harsh11067/skinnova-gemma4-e2b-export 2>&1 | tail -1
rm -rf reports/kaggle/export_merged && mkdir -p reports/kaggle/export_merged
$KAG kernels output harsh11067/skinnova-gemma4-e2b-export -p reports/kaggle/export_merged --file-pattern '.*\.(json|log)$' >/dev/null 2>&1
mkdir -p models/litertlm/skinnova
OUT=models/litertlm/skinnova/skinnova-e2b-v1.litertlm
if [ ! -f "$OUT.ok" ]; then
  ~/.local/share/uv/tools/kaggle/bin/python scripts/kaggle_output_url.py harsh11067/skinnova-gemma4-e2b-export '\.litertlm$' > models/litertlm/skinnova/urls.tsv || exit 1
  scripts/dl_url.sh "$(cut -f2 models/litertlm/skinnova/urls.tsv | head -1)" "$OUT"
  want=$(python3 -c "import json;print(json.load(open('reports/kaggle/export_merged/reports/export_merged.json'))['litertlm']['sha256'])")
  have=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$want" = "$have" ] && touch "$OUT.ok" || { echo "sha mismatch $have vs $want"; exit 1; }
fi
until grep -q V2_TRAINED logs/run_v2.log 2>/dev/null; do sleep 120; done
[ -f reports/parity_l6_skinnova_v1.json ] || .venv-export/bin/python -m ml.eval.parity_l6 --model "$OUT" \
    --hf-report "$R" --tag skinnova_v1 --threads 12 2>&1 | grep -vE "^(INFO|WARNING|W0000|I0000|E0000)"
echo AFTER_LORA_DONE
