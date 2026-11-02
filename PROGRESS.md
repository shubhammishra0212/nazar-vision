# nazar-vision progress

Updated after every step (power cuts). After a restart: read this file and `git log` first.

## Done
- `.venv` (Python 3.11, onnxruntime numpy opencv-python-headless pillow; no torch) - verified imports.
- `models/backbone.onnx` (103 MB) exported fresh from torchvision WRN50 ImageNet weights via `nazar_vision/export.py`
  with the old torch venv; ONNX vs torch max diff 1.6e-06. `models/` is gitignored (release asset).
- Package `nazar_vision/`: `locate.py`, `features.py`, `memory.py`, `inspector.py`, `split.py`, `synth.py`, `train.py`,
  `evaluate.py`, `service.py`, `export.py`, `images.py`.
- Data copied to `data/good` (26 good photos), `data/unsure` (Photo-16/17), `data/placements.csv` (hand placements of the first set).
- Split: 9 placements -> train 18 photos (3), calib 5 (3), test 3 (3). Synthetic: 15 defects (5 kinds) + 5 no-part frames.
- Trained: bank 15,206 patches (15.6 MB float16), HOLD 1.347 (max of 5 calib), REJECT 1.8184 (hold x 1.35, no real defects).
- Evaluated (`REPORT.md`, `docs/score_histogram.png`, `models/eval.json`): good test 3/3 PASS, calib 5/5 PASS,
  synthetic 15/15 REJECT, no-part 5/5 NO_PART, unsure 2/2 PASS. Latency median 50 ms, p95 54 ms (in-process).
- Service tested with curl on 127.0.0.1:8765 (before the power cut): health, inspect good (PASS 0.896), synthetic notch
  (REJECT 2.741), no washer (NO_PART), grey PNG (NO_PART), hold/reject override, bad override, reject<hold, text body,
  empty body, truncated JPEG, capture good, capture defect PNG + kind sanitising, bad label, OPTIONS 204 + CORS, unknown path.
  All answered HTTP 200 JSON; service stayed up. Test captures deleted afterwards.

## In progress
- PROGRESS.md, requirements.txt, README.md.

## Next
- Test `--test-images` mode and HTTP round-trip latency (curl, 20 calls).
- `retrain.sh` resumable by stage; one full `./retrain.sh` run.
- `docs/vision.md` -> `docs/NAZAR-02-Vision-AI.pdf` (tools/mdpdf.py), copy to `../PDFs/`, check render with qlmanage.
