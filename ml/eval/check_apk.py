"""S11 (static half) + release facts from the built APKs → reports/apk_check.json.

The `offline` flavor must declare no INTERNET permission (CLAUDE.md non-negotiable 1); `online` may (one model download).
Reads the APKs Gradle just built with `aapt2 dump permissions / badging`. The airplane-mode run on two phones is the
device half of S11 (test.md §11).

  python -m ml.eval.check_apk          # after :app:assembleOfflineDebug :app:assembleOnlineDebug
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

from ml.common.paths import REPO, REPORTS, report_meta

AAPT2 = sorted(glob.glob(str(Path.home() / "android/sdk/build-tools/*/aapt2")))[-1]
APKS = {f: REPO / f"android/app/build/outputs/apk/{f}/debug/app-{f}-debug.apk" for f in ("offline", "online")}


def main():
    rep = {**report_meta(), "aapt2": AAPT2, "flavors": {}}
    for flavor, apk in APKS.items():
        perms = re.findall(r"uses-permission: name='([^']+)'", subprocess.run([AAPT2, "dump", "permissions", str(apk)],
                                                                             capture_output=True, text=True, check=True).stdout)
        badging = subprocess.run([AAPT2, "dump", "badging", str(apk)], capture_output=True, text=True, check=True).stdout
        ver = re.search(r"versionName='([^']*)'", badging)
        abis = re.search(r"native-code: (.*)", badging)
        rep["flavors"][flavor] = {"apk": str(apk.relative_to(REPO)), "bytes": apk.stat().st_size,
                                  "sha256": hashlib.sha256(apk.read_bytes()).hexdigest(), "version": ver.group(1) if ver else None,
                                  "native_abis": abis.group(1).replace("'", "").split() if abis else [],
                                  "permissions": perms, "internet": "android.permission.INTERNET" in perms}
    rep["S11_offline_no_internet"] = not rep["flavors"]["offline"]["internet"]
    (REPORTS / "apk_check.json").write_text(json.dumps(rep, indent=1))
    for f, d in rep["flavors"].items():
        print(f, d["version"], f"{d['bytes'] / 1e6:.1f} MB", d["native_abis"], "INTERNET" if d["internet"] else "no INTERNET", d["permissions"])
    print("S11 static (offline has no INTERNET):", "PASS" if rep["S11_offline_no_internet"] else "FAIL")


if __name__ == "__main__":
    main()
