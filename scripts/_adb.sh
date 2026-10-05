# shared: Windows adb.exe from WSL (never start a Linux adb server — CLAUDE.md)
ADB="${ADB:-$(ls /mnt/c/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe 2>/dev/null | head -1)}"
[ -x "$ADB" ] || { echo "adb.exe not found; set ADB=/mnt/c/.../adb.exe"; exit 1; }
PKG="${PKG:-com.skinnova.app}"
