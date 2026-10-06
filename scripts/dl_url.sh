#!/usr/bin/env bash
# dl_url.sh <url> <out> — single resumable streaming download (retries; prints sha256 at the end)
for i in 1 2 3 4 5; do curl -s --retry 10 --retry-all-errors -C - -o "$2" "$1" && break; sleep 10; done
sha256sum "$2"; echo DL_DONE
