#!/usr/bin/env bash
# Device perf (test.md §10): run the real-engine benchmark test while sampling PSS every second + thermal status.
set -euo pipefail; cd "$(dirname "$0")/.."; . scripts/_adb.sh
mkdir -p reports/device
"$ADB" shell getprop ro.product.model | tr -d '\r' > reports/device/model.txt 2>/dev/null || true
"$ADB" shell cat /proc/meminfo | head -1 | tr -d '\r' >> reports/device/model.txt
( for i in $(seq 1 600); do "$ADB" shell dumpsys meminfo "$PKG" 2>/dev/null | tr -d '\r' | grep -E "TOTAL PSS|TOTAL:" | head -1; sleep 1; done ) > reports/device/pss.txt &
PSS=$!
"$ADB" shell dumpsys thermalservice | tr -d '\r' | grep -i status | head -3 > reports/device/thermal_before.txt || true
scripts/device_tests.sh "com.skinnova.app.DeviceTests#realEngineBench"
"$ADB" shell dumpsys thermalservice | tr -d '\r' | grep -i status | head -3 > reports/device/thermal_after.txt || true
kill $PSS 2>/dev/null || true
echo "peak PSS (kB): $(grep -oE '[0-9]+' reports/device/pss.txt | sort -n | tail -1)"
