#!/usr/bin/env bash
# make_lora_dataset.sh <version> — publish a finished LoRA run's adapter as the private Kaggle dataset
# harsh11067/skinnova-lora-v<version> (input of `make_export --mode lora<version>`; a kernel in ERROR can't be mounted).
#   scripts/make_lora_dataset.sh 2      # kernel output lora/* + reports/kaggle/full_v2/reports/full.json → models/lora_v2/
# Signed download URLs are kept OUTSIDE the uploaded folder (models/lora_v2.urls.tsv) and never uploaded.
set -euo pipefail; cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
V=$1; K=harsh11067/skinnova-gemma4-e2b-lora; D=models/lora_v$V; U=models/lora_v$V.urls.tsv
REP=reports/kaggle/full_v$V/reports/full.json
[ -f "$REP" ] || { echo "no $REP — fetch the run's reports first"; exit 1; }
mkdir -p "$D/reports"
~/.local/share/uv/tools/kaggle/bin/python scripts/kaggle_output_url.py $K '^lora/[^/]+$' > "$U"
[ -s "$U" ] || { echo "no lora/ files in the kernel output"; exit 1; }
while IFS=$'\t' read -r name url; do
  f="$D/$(basename "$name")"
  curl -s --retry 10 --retry-all-errors -C - -o "$f" "$url"
  echo "$(basename "$name") $(stat -c %s "$f")"
done < "$U"
cp "$REP" "$D/reports/full.json"
python3 - "$D" "$V" <<'PY'
import json, sys
d, v = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{d}/adapter_config.json")); print("adapter r", cfg["r"], "alpha", cfg["lora_alpha"], "rslora", cfg.get("use_rslora"))
json.dump({"title": f"skinnova-lora-v{v}", "id": f"harsh11067/skinnova-lora-v{v}", "licenses": [{"name": "other"}]},
          open(f"{d}/dataset-metadata.json", "w"), indent=1)
PY
# nothing but adapter files, configs, tokenizer, README and the training report may be uploaded
bad=$(find "$D" -type f ! -name "*.json" ! -name "*.safetensors" ! -name "*.jinja" ! -name "README.md")
[ -z "$bad" ] || { echo "refusing to upload unexpected files: $bad"; exit 1; }
grep -rlE "kaggleusercontent|KGAT_|hf_[A-Za-z0-9]{20,}" "$D" && { echo "refusing: URL/token found in $D"; exit 1; }
~/.local/bin/kaggle datasets create -p "$D" -r zip 2>&1 | tail -2
echo LORA_DATASET_V${V}_CREATED
