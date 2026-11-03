#!/bin/bash
# One command after recapturing: ingest -> split -> synthetic test images -> train -> damage library -> trained-damage line
# (match) -> evaluate (rewrites REPORT.md).
# ingest runs when footage/ exists or folders were ingested before (./ingest.sh <folder> --label ...):
#   iPhone photos/videos in footage/good, footage/defect/<kind>, footage/empty -> data/footage/
# Other photos: data/capture/good, data/capture/defect/<kind> (console captures), data/defect/<kind>,
# and data/good (first photo set, only used when there is no footage).
# Every stage writes its result to disk, so after an interruption you can resume from a stage:
#   ./retrain.sh            all four stages
#   ./retrain.sh train      start at train (also: ingest, split, synth, evaluate)
# A running service picks the new model up on restart (./run.sh).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -f models/backbone.onnx ]; then
  echo "models/backbone.onnx is missing: download it from the release, or export it with a torch venv:"
  echo "  ~/Downloads/NAZAR/10-AI/.venv/bin/python -m nazar_vision.export"
  exit 1
fi
FROM=${1:-ingest}
case "$FROM" in ingest|split|synth|train|damage|match|evaluate) ;; *) echo "unknown stage '$FROM' (ingest, split, synth, train, damage, match, evaluate)"; exit 1;; esac
run=0
for stage in ingest split synth train damage match evaluate; do
  [ "$stage" = "$FROM" ] && run=1
  [ $run = 1 ] || { echo "== skip $stage"; continue; }
  if [ "$stage" = ingest ] && [ ! -d footage ] && [ ! -d data/footage/_records ]; then
    echo "== skip ingest (no footage/ folder)"; continue
  fi
  echo "== $stage"
  $PY -m nazar_vision.$stage
done
echo "Done. Read REPORT.md, then restart the service (./run.sh)."
