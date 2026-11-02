#!/bin/bash
# Start the NAZAR vision service on 127.0.0.1:8765. Extra arguments are passed on, e.g.
#   ./run.sh --test-images data/synthetic        (demo without a camera)
#   ./run.sh --port 8775
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv yet. Make it once:  ~/.local/bin/uv venv -p 3.11 .venv && ~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi
exec .venv/bin/python -m nazar_vision.service "$@"
