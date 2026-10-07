#!/usr/bin/env bash
# Push a .litertlm to the app's external files dir (dev path, plan §5). The app hashes it once (sha256 vs model_manifest.json).
#   scripts/push_model.sh models/litertlm/skinnova/model.litertlm      (or the stock gemma-4-E2B-it.litertlm)
set -euo pipefail
SRC="$(realpath -e "${1:?path to .litertlm}")"   # resolve BEFORE cd: a relative path once pushed the scripts/ folder instead
[ -f "$SRC" ] || { echo "not a file: $SRC"; exit 1; }
cd "$(dirname "$0")"; . ./_adb.sh
DEST="/sdcard/Android/data/$PKG/files/models/$(basename "$SRC")"   # the app identifies a model by sha256, not by name
"$ADB" shell mkdir -p "/sdcard/Android/data/$PKG/files/models/"
WIN=$(wslpath -w "$SRC")
time "$ADB" push "$WIN" "$DEST"
"$ADB" shell ls -l "$DEST"
