#!/usr/bin/env bash
# After the arms run frees the CPU: TL1 v8 = the v7b method on 900 synthetic val pairs instead of 300 (same generator, same
# gates) — the coin-pair p90 rested on only 29 pairs. Not tuning: no parameter changes. Restart-safe.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q RUN_ARMS_DONE logs/run_arms.log 2>/dev/null; do sleep 120; done
if [ ! -f reports/timeline_eval_v8.json ]; then
  MALLOC_ARENA_MAX=2 .venv/bin/python -m ml.timeline.eval_timeline --n 900 --workers 8 --out timeline_eval_v8.json 2>&1 | tail -3
fi
python3 -c "import json; d=json.load(open('reports/timeline_eval_v8.json')); print('TL1 v8', d['n_pairs'], 'pairs', d['area_rel_err_coin'], d['gates'])"
echo "[$(date +%T)] AFTER_ARMS_DONE"
