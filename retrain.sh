#!/bin/bash
# One command after recapturing: split -> synthetic test images -> train -> evaluate (rewrites REPORT.md).
# Photos come from data/good, data/capture/good, data/defect/<kind>, data/capture/defect/<kind>.
# Every stage writes its result to disk, so after an interruption you can resume from a stage:
#   ./retrain.sh            all four stages
#   ./retrain.sh train      start at train (also: split, synth, evaluate)
# A running service picks the new model up on restart (./run.sh).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -f models/backbone.onnx ]; then
  echo "models/backbone.onnx is missing: download it from the release, or export it with a torch venv:"
  echo "  ~/Downloads/NAZAR/10-AI/.venv/bin/python -m nazar_vision.export"
  exit 1
fi
FROM=${1:-split}
case "$FROM" in split|synth|train|evaluate) ;; *) echo "unknown stage '$FROM' (split, synth, train, evaluate)"; exit 1;; esac
run=0
for stage in split synth train evaluate; do
  [ "$stage" = "$FROM" ] && run=1
  [ $run = 1 ] || { echo "== skip $stage"; continue; }
  echo "== $stage"
  $PY -m nazar_vision.$stage
done
echo "Done. Read REPORT.md, then restart the service (./run.sh)."
