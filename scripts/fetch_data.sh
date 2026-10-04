#!/usr/bin/env bash
# Resumable raw-data download (safe to re-run; curl -C - resumes partial files).
set -u
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
R=data/raw; mkdir -p $R/{mgmitesh,pad_ufes20,skindisnet,skindiseasebd}
get() { # url out [auth]
  for i in 1 2 3 4 5; do curl -sSL -C - --retry 5 ${3:+-H "$3"} -o "$2" "$1" && return 0; echo "retry $i $2"; sleep 5; done; return 1; }
unz() { [ -f "$1.done" ] || { python3 -m zipfile -e "$1" "$(dirname "$1")/x_$(basename "$1" .zip)" && touch "$1.done"; }; }
get https://data.mendeley.com/public-files/datasets/9ggd3shdr7/files/e7fe24c7-e29d-4f9b-8934-8f54ba5170da/file_downloaded $R/skindiseasebd/Images.zip && unz $R/skindiseasebd/Images.zip && echo "BD ok"
get "https://www.kaggle.com/api/v1/datasets/download/mgmitesh/skin-disease-detection-dataset" $R/mgmitesh/mgmitesh.zip "Authorization: Bearer $KAGGLE_API_TOKEN" && unz $R/mgmitesh/mgmitesh.zip && echo "MG ok"
get https://data.mendeley.com/public-files/datasets/yj3md44hxg/files/2e67c136-67c5-41c0-ae7b-298d601c447d/file_downloaded $R/skindisnet/SkinDisNet_2.zip && unz $R/skindisnet/SkinDisNet_2.zip && echo "SDN2 ok"
get https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1 $R/pad_ufes20/pad.zip && unz $R/pad_ufes20/pad.zip && echo "PAD ok"
echo ALL_DONE
