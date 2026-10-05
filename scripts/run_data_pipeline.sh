#!/usr/bin/env bash
# Full data pipeline; every stage is incremental/restart-safe, so re-running after a WSL restart is cheap.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until [ -f data/raw/pad_ufes20/pad.zip.done ]; do sleep 10; done
echo "[$(date +%T)] sources";      $PY -m ml.data.sources
echo "[$(date +%T)] label map";    $PY -m ml.data.build_label_map
echo "[$(date +%T)] normalize";    $PY -m ml.data.normalize --workers 12
echo "[$(date +%T)] embed";        $PY -m ml.data.embed_dedupe embed
echo "[$(date +%T)] copy pairs";   $PY -m ml.data.embed_dedupe pairs
echo "[$(date +%T)] dedupe";       $PY -m ml.data.dedupe
echo "[$(date +%T)] splits";       $PY -m ml.data.make_splits
echo "[$(date +%T)] data card";    $PY -m ml.data.data_card > /dev/null
echo "[$(date +%T)] PIPELINE_DONE"
