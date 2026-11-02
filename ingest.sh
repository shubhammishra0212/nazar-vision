#!/bin/bash
# Turn iPhone photos / videos into training images (see nazar_vision/ingest.py).
#   ./ingest.sh                                          everything under footage/good, footage/defect/<kind>, footage/empty
#   ./ingest.sh ~/Downloads/washers --label good         a flat folder, e.g. straight from AirDrop
#   ./ingest.sh ~/Downloads/bad --label defect --kind scratch
#   ./ingest.sh --reset                                  forget everything ingested so far
# Sources already ingested are skipped, so it is safe to rerun after an interruption. Then run ./retrain.sh.
set -e
cd "$(dirname "$0")"
exec .venv/bin/python -m nazar_vision.ingest "$@"
