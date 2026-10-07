#!/usr/bin/env bash
# On-device tests WITHOUT a Linux adb server (CLAUDE.md): build the app + test APKs in WSL, install and run them with
# Windows adb.exe, save the raw instrumentation log, pull bench outputs.
#   scripts/device_tests.sh                                   # every instrumentation test
#   scripts/device_tests.sh com.skinnova.app.ResultFlowTest   # one class (or Class#method)
set -euo pipefail; cd "$(dirname "$0")/.."; . scripts/_adb.sh
"$ADB" get-state >/dev/null 2>&1 || { echo "no device: connect the phone with USB debugging (d2y §3)"; exit 1; }
( cd android && JAVA_HOME=~/android/jdk ANDROID_HOME=~/android/sdk ./gradlew -q :app:assembleOfflineDebug :app:assembleOfflineDebugAndroidTest )
APK=android/app/build/outputs/apk/offline/debug/app-offline-debug.apk
TAPK=android/app/build/outputs/apk/androidTest/offline/debug/app-offline-debug-androidTest.apk
"$ADB" install -r -g "$(wslpath -w "$APK")" >/dev/null
"$ADB" install -r -g "$(wslpath -w "$TAPK")" >/dev/null
mkdir -p reports/device
LOG=reports/device/instrument_$(date +%Y%m%d_%H%M%S).txt
ARGS=(); [ -n "${1:-}" ] && ARGS=(-e class "$1")
# capture the whole run's logcat on the PC: phone log buffers are small (a crash on 2026-10-07 left no trace on a vivo)
"$ADB" logcat -c >/dev/null 2>&1 || true
"$ADB" logcat -v threadtime > "${LOG%.txt}.logcat.txt" 2>&1 & LCPID=$!
"$ADB" shell am instrument -w -r "${ARGS[@]}" "$PKG.test/androidx.test.runner.AndroidJUnitRunner" | tr -d '\r' > "$LOG"
kill $LCPID 2>/dev/null || true
python3 - "$LOG" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
blocks = t.split("INSTRUMENTATION_STATUS_CODE:")
res = {}
cur = {}
for line in t.splitlines():
    m = re.match(r"INSTRUMENTATION_STATUS: (class|test)=(.*)", line)
    if m: cur[m.group(1)] = m.group(2)
    m = re.match(r"INSTRUMENTATION_STATUS_CODE: (-?\d+)", line)
    if m and "test" in cur:
        code = int(m.group(1)); key = f'{cur["class"].split(".")[-1]}#{cur["test"]}'
        if code != 1: res[key] = {0: "PASS", -1: "ERROR", -2: "FAIL", -3: "IGNORED", -4: "ASSUMPTION_SKIPPED"}.get(code, str(code))
for k, v in res.items(): print(f"{v:20} {k}")
print(re.findall(r"(OK \(\d+ tests?\)|FAILURES!!!\nTests run: \d+,  Failures: \d+)", t))
PY
echo "log: $LOG"
scripts/pull_bench.sh >/dev/null || true
