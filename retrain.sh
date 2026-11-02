#!/bin/bash
# One command after recapturing: split -> synthetic test images -> train -> evaluate (rewrites REPORT.md).
# Photos come from data/good, data/capture/good, data/defect/<kind>, data/capture/defect/<kind>.
# A running service picks the new model up on restart (./run.sh).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -f models/backbone.onnx ]; then
  echo "models/backbone.onnx is missing: download it from the release, or export it with a torch venv:"
  echo "  ~/Downloads/NAZAR/10-AI/.venv/bin/python -m nazar_vision.export"
  exit 1
fi
echo "== 1/4 split by placement";        $PY -m nazar_vision.split
echo "== 2/4 synthetic test images";     $PY -m nazar_vision.synth
echo "== 3/4 train memory bank";         $PY -m nazar_vision.train
echo "== 4/4 evaluate";                  $PY -m nazar_vision.evaluate
echo "Done. Read REPORT.md, then restart the service (./run.sh)."
