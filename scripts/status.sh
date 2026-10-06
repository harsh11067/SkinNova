#!/usr/bin/env bash
# One-line status of every running job (used by the periodic check).
cd "$(dirname "$0")/.."
ep=$(grep -E '"epoch"' logs/run_v2.log 2>/dev/null | tail -1 | python3 -c "import sys,json
l=sys.stdin.read()
print('ep%d f1 %.3f' % (json.loads(l[l.index('{'):])['epoch'], json.loads(l[l.index('{'):])['val_macro_f1'])) if '{' in l else print('-')" 2>/dev/null)
last() { tail -1 "$1" 2>/dev/null | grep -oE "(KernelWorkerStatus\.[A-Z]+|[A-Z_]{6,}DONE|FETCHED|ADOPT|KEEP_V1|LORA_V2_PUSHED|V2_TRAINED)" | tail -1; }
v2f=$(grep -oE "adoption: [A-Z_0-9]+|V2_FINISH_DONE" logs/v2_finish.log 2>/dev/null | tail -1)
echo "$(date +%H:%M) up$(uptime -p | sed 's/up //;s/ hours\?/h/;s/ minutes\?/m/;s/,//g') | CV-v2 ${ep:-?} | v2fin ${v2f:-wait} | LoRA2 $(last logs/watch_lora2.log) | v2chain $(last logs/after_lora_v2.log) | arms $(last logs/watch_arms.log) | loraV2 $(grep -o "LORA_V2_PUSHED" logs/after_lora_v1.log 2>/dev/null | tail -1) | mem $(free -m | awk '/Mem/{print $7}')MB"
