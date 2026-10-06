#!/usr/bin/env bash
# Same 10 llm_val analyses + 5 extractions, greedy, one model at a time (WSL RAM): official vs our export vs hybrid.
cd "$(dirname "$0")/.."
for spec in "official_greedy10:models/litertlm/stock/gemma-4-E2B-it.litertlm" \
            "hybrid_stock_greedy10:models/litertlm/hybrid/stock-hybrid.litertlm" \
            "export_v10_greedy10:models/litertlm/export_v10/skinnova-e2b-stock.litertlm"; do
  tag=${spec%%:*}; m=${spec#*:}
  [ -f reports/llm_litertlm_${tag}.json ] && { echo "skip $tag"; continue; }
  echo "[$(date +%T)] $tag"
  .venv-export/bin/python -m ml.eval.eval_llm --model "$m" --tag "$tag" --n 10 --greedy 2>&1 | grep -vE "^(INFO|WARNING|W0000|I0000|E0000)"
done
echo COMPARE_DONE
