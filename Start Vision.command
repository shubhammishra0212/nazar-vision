#!/bin/bash
# Double-click in Finder to start the NAZAR vision service. Close this window to stop it.
cd "$(dirname "$0")"
echo "NAZAR vision - washer inspection on http://127.0.0.1:8765"
./run.sh
echo "Service stopped. Press Return to close."
read -r
