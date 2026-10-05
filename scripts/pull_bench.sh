#!/usr/bin/env bash
# Pull on-device bench outputs (L7, C5) into reports/device/.
set -euo pipefail; cd "$(dirname "$0")/.."; . scripts/_adb.sh
mkdir -p reports/device
"$ADB" pull "/sdcard/Android/data/$PKG/files/bench/" "$(wslpath -w reports/device)" || echo "no bench outputs yet"
ls -la reports/device/
