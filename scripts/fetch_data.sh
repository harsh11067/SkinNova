#!/usr/bin/env bash
# Resumable raw-data download (safe to re-run after a WSL restart). Kaggle supports byte-range resume (-C -).
# .ok = download verified (zip test), .done = extracted. Mendeley was dropped: it has no range support and a
# restart threw away 480 MB twice; SkinDisNet and PAD-UFES-20 come from Kaggle mirrors of the original releases.
set -u
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
R=data/raw
kag() { # owner/slug out.zip
  [ -f "$2.ok" ] && return 0
  for i in $(seq 1 30); do
    curl -sSL -C - --retry 10 --retry-delay 5 -H "Authorization: Bearer $KAGGLE_API_TOKEN" -o "$2" \
      "https://www.kaggle.com/api/v1/datasets/download/$1"
    python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).testzip()" "$2" 2>/dev/null && touch "$2.ok" && return 0
    echo "retry $i $2 ($(du -h "$2" | cut -f1))"; sleep 10
  done; return 1; }
unz() { [ -f "$1.done" ] || { python3 -m zipfile -e "$1" "$(dirname "$1")/x_$(basename "$1" .zip)" && touch "$1.done"; }; }
mkdir -p $R/{mgmitesh,pacificrm,pad_ufes20,skindisnet}
kag mgmitesh/skin-disease-detection-dataset $R/mgmitesh/mgmitesh.zip && unz $R/mgmitesh/mgmitesh.zip && echo "MG ok"
kag nipamridha12/skindisnet $R/skindisnet/skindisnet.zip && unz $R/skindisnet/skindisnet.zip && echo "SDN ok"
kag mahdavi1202/skin-cancer $R/pad_ufes20/pad.zip && unz $R/pad_ufes20/pad.zip && echo "PAD ok"
echo ALL_DONE
