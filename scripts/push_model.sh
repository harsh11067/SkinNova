#!/usr/bin/env bash
# Push a .litertlm to the app's external files dir (dev path, plan §5). The app hashes it once (sha256 vs model_manifest.json).
#   scripts/push_model.sh models/litertlm/skinnova/model.litertlm      (or the stock gemma-4-E2B-it.litertlm)
set -euo pipefail; cd "$(dirname "$0")"; . ./_adb.sh
SRC="${1:?path to .litertlm}"
DEST="/sdcard/Android/data/$PKG/files/models/skinnova-e2b-v1.litertlm"
"$ADB" shell mkdir -p "/sdcard/Android/data/$PKG/files/models/"
WIN=$(wslpath -w "$(realpath "$SRC")")
time "$ADB" push "$WIN" "$DEST"
"$ADB" shell ls -l "$DEST"
