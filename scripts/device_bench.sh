#!/usr/bin/env bash
# Device perf (test.md §10): install offline debug, run the on-device instrumentation tests, sample PSS, pull results.
set -euo pipefail; cd "$(dirname "$0")/.."; . scripts/_adb.sh
"$ADB" devices -l
"$ADB" shell getprop ro.product.model; "$ADB" shell cat /proc/meminfo | head -1
cmd.exe /c "cd /d $(wslpath -w android) && gradlew.bat connectedOfflineDebugAndroidTest" || true
( for i in $(seq 1 120); do "$ADB" shell dumpsys meminfo "$PKG" 2>/dev/null | grep -E "TOTAL PSS|TOTAL:" | head -1; sleep 1; done ) > reports/device_pss.txt &
"$ADB" shell dumpsys thermalservice | grep -i status | head -3 > reports/device_thermal.txt || true
scripts/pull_bench.sh
