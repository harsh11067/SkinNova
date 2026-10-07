#!/usr/bin/env bash
# After a PC / WSL restart: relaunch every unfinished background chain (each one skips the steps already done; eval_arms
# resumes per case; a selection run that was cut off restarts from its first record). Safe to run twice: a chain that is
# already running is left alone.
#   scripts/resume.sh
set -uo pipefail
cd "$(dirname "$0")/.."
launch() {   # launch <script> <log> <done-marker>
  if grep -q "$3" "$2" 2>/dev/null; then echo "done     $1"; return; fi
  if pgrep -f "bash $1" >/dev/null; then echo "running  $1"; return; fi
  setsid nohup "$1" >> "$2" 2>&1 < /dev/null & disown
  echo "started  $1 (log $2)"
}
launch scripts/after_lora_v2.sh logs/after_lora_v2.log AFTER_LORA_V2_DONE
launch scripts/run_arms.sh logs/run_arms.log RUN_ARMS_DONE
launch scripts/run_v3.sh logs/run_v3.log RUN_V3_DONE
launch scripts/after_arms.sh logs/after_arms.log AFTER_ARMS_DONE
scripts/status.sh
