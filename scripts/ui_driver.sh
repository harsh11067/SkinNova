#!/bin/bash
# Tiny adb UI driver for hands-on trials on a USB phone (Windows adb.exe via scripts/_adb.sh).
#   ui_driver.sh tap "<text|content-desc>[*]" [tries]   tap the first node whose text/desc matches (* = prefix)
#   ui_driver.sh wait_text "<substring>" [secs]         wait until the screen contains it
#   ui_driver.sh texts [n]                              list visible texts
#   ui_driver.sh swipe_up | swipe_down                  scroll
#   ui_driver.sh gfx_reset | gfx                        frame statistics (dumpsys gfxinfo)
#   ui_driver.sh shot <file.png>                        screenshot
cd "$(dirname "$0")/.." && . scripts/_adb.sh
dump() { "$ADB" shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; "$ADB" shell cat /sdcard/ui.xml | tr -d '\r'; }
center() { python3 -c "
import re,sys
x=sys.stdin.read(); t=sys.argv[1]
for m in re.finditer(r'<node [^>]*>', x):
    n=m.group(0)
    if ('text=\"'+t+'\"' in n) or ('content-desc=\"'+t+'\"' in n) or (t.endswith('*') and ('text=\"'+t[:-1]) in n):
        b=re.search(r'bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"', n); a=list(map(int,b.groups())); print((a[0]+a[2])//2, (a[1]+a[3])//2); break
" "$1"; }
case "$1" in
  tap) for i in $(seq 1 ${3:-10}); do c=$(dump | center "$2"); [ -n "$c" ] && { "$ADB" shell input tap $c; echo "tapped $2 at $c"; exit 0; }; sleep 1; done; echo "NOT FOUND: $2"; exit 1;;
  wait_text) for i in $(seq 1 ${3:-30}); do dump | grep -q "$2" && { echo "seen $2"; exit 0; }; sleep 1; done; echo "NOT SEEN: $2"; exit 1;;
  texts) dump | grep -o 'text="[^"]\+"' | head -${2:-25};;
  swipe_up) "$ADB" shell input swipe 540 1800 540 600 300;;
  swipe_down) "$ADB" shell input swipe 540 600 540 1800 300;;
  gfx_reset) "$ADB" shell dumpsys gfxinfo com.skinnova.app reset >/dev/null;;
  gfx) "$ADB" shell dumpsys gfxinfo com.skinnova.app | tr -d '\r' | grep -E "Total frames|Janky frames|50th|90th|95th|99th" | head -6;;
  shot) "$ADB" exec-out screencap -p > "$2";;
esac
