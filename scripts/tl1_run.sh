#!/usr/bin/env bash
# Full TL1 synthetic evaluation (restart-safe wrapper) → reports/timeline_eval.json, logs/tl1_${TL1_TAG:-v5}.log
cd "$(dirname "$0")/.."
MALLOC_ARENA_MAX=2 .venv/bin/python -m ml.timeline.eval_timeline --n 300 --workers 8 > logs/tl1_${TL1_TAG:-v5}.log 2>&1
echo TL1_DONE >> logs/tl1_${TL1_TAG:-v5}.log
