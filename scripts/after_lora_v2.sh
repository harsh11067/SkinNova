#!/usr/bin/env bash
# LoRA v2 → .litertlm → L6 → pre-registered v1/v2 selection (decisions.md 2026-10-06). Restart-safe (each step skips if done).
#  1. adapter dataset harsh11067/skinnova-lora-v2 ready (+5 min: a fresh version is mounted stale right after "ready")
#  2. export kernel --mode lora2 (streaming manual merge) → stream the .litertlm, sha256 vs the kernel report
#  3. L6 parity vs the v2 training report (its own val build, data/llm)
#  4. selection: v1 and v2 .litertlm, greedy, the same first 100 records of the FROZEN llm_val (data/llm_eval);
#     ship v2 only if its category agreement is higher; ties → v1
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python; PYX=.venv-export/bin/python; KAG=~/.local/bin/kaggle; K=harsh11067/skinnova-gemma4-e2b-export
st() { $KAG kernels status "$1" 2>&1 | tail -1; }
D=models/litertlm/skinnova; V1=$D/skinnova-e2b-v1.litertlm; V2=$D/skinnova-e2b-v2.litertlm
if [ ! -f logs/export_lora2.pushed ]; then
  until $KAG datasets status harsh11067/skinnova-lora-v2 2>&1 | tail -1 | grep -qi ready; do sleep 30; done
  sleep 300
  $PY -m ml.llm.notebooks.make_export --mode lora2 2>&1 | grep -iE "pushed|error" && touch logs/export_lora2.pushed || exit 1
  sleep 120
fi
echo "[$(date +%T)] EXPORT_V2_PUSHED"
# v1's selection run needs nothing from v2: do it while Kaggle exports, once the CPU timeline jobs are gone (RAM: engine ~4 GB)
F='^(INFO|WARNING|W0000|I0000|E0000)'
while pgrep -f "ml.timeline.eval_timeline|seg_tune_frozen|make_tl3_fixtures" >/dev/null; do sleep 60; done
[ -f reports/llm_litertlm_select_v1.json ] || $PYX -m ml.eval.eval_llm --model "$V1" --tag select_v1 --greedy \
    --data-dir data/llm_eval --set llm_val --n 100 --threads 12 2>&1 | grep -vE "$F" | tail -4
echo "[$(date +%T)] SELECT_V1_DONE"
until st $K | grep -qiE "COMPLETE|ERROR|CANCEL"; do sleep 180; done
echo "[$(date +%T)] export: $(st $K)"
R=reports/kaggle/export_lora2/reports/export_lora2.json
if [ ! -f "$R" ]; then
  mkdir -p reports/kaggle/export_lora2
  $KAG kernels output $K -p reports/kaggle/export_lora2 --file-pattern '.*\.(json|log)$' >/dev/null 2>&1
fi
[ -f "$R" ] || { echo "no export report"; exit 1; }
python3 -c "import json;d=json.load(open('$R'));print({k:d.get(k) for k in ['merge','litertlm','capabilities']})"
if [ ! -f "$V2.ok" ]; then
  ~/.local/share/uv/tools/kaggle/bin/python scripts/kaggle_output_url.py $K '\.litertlm$' > $D/urls_v2.tsv || exit 1
  [ -s $D/urls_v2.tsv ] || { echo "no .litertlm in the export output"; exit 1; }
  scripts/dl_url.sh "$(cut -f2 $D/urls_v2.tsv | head -1)" "$V2"
  want=$(python3 -c "import json;print(json.load(open('$R'))['litertlm']['sha256'])"); have=$(sha256sum "$V2" | cut -d' ' -f1)
  [ "$want" = "$have" ] && touch "$V2.ok" || { echo "sha mismatch $have vs $want"; exit 1; }
fi
echo "[$(date +%T)] V2_DOWNLOADED"
[ -f reports/parity_l6_skinnova_v2.json ] || $PYX -m ml.eval.parity_l6 --model "$V2" \
    --hf-report reports/kaggle/full_v2/reports/full.json --tag skinnova_v2 --threads 12 2>&1 | grep -vE "$F" | tail -16
echo "[$(date +%T)] L6_V2_DONE"
for v in 1 2; do
  M=$D/skinnova-e2b-v$v.litertlm
  [ -f reports/llm_litertlm_select_v$v.json ] || $PYX -m ml.eval.eval_llm --model "$M" --tag select_v$v --greedy \
      --data-dir data/llm_eval --set llm_val --n 100 --threads 12 2>&1 | grep -vE "$F" | tail -4
done
python3 - <<'PY'
import json
s = {v: json.load(open(f"reports/llm_litertlm_select_v{v}.json")) for v in (1, 2)}
a1, a2 = s[1]["analysis"]["cat_agree"], s[2]["analysis"]["cat_agree"]
print("selection on frozen llm_val, n_analysis", s[1]["n_analysis"], s[2]["n_analysis"], "| category agreement v1", a1, "v2", a2,
      "| valid v1", s[1]["analysis"]["valid"], "v2", s[2]["analysis"]["valid"], "→ SHIP", "v2" if a2 > a1 else "v1 (rule: ties → v1)")
PY
echo "[$(date +%T)] AFTER_LORA_V2_DONE"
