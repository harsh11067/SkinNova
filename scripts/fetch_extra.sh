#!/usr/bin/env bash
# Extra source for psoriasis/vitiligo (DermNet-derived per class list; Kaggle label CC0) — see docs/decisions.md
set -u
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
D=data/raw/pacificrm; mkdir -p $D
for i in 1 2 3 4 5; do curl -sSL -C - --retry 5 -H "Authorization: Bearer $KAGGLE_API_TOKEN" -o $D/pacificrm.zip \
  "https://www.kaggle.com/api/v1/datasets/download/pacificrm/skindiseasedataset" && break; sleep 5; done
[ -f $D/pacificrm.zip.done ] || { python3 -m zipfile -e $D/pacificrm.zip $D/x_pacificrm && touch $D/pacificrm.zip.done; }
echo PACIFICRM_DONE
