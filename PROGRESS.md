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

- `requirements.txt`, `PROGRESS.md`.
- `retrain.sh` resumable by stage (`./retrain.sh train`); full run 62 s wall clock, same numbers.
- `--test-images data/synthetic` tested: frames replaced in order (burr, burr, ..., nopart), `test_images` in /health.
- HTTP round trip (curl, 20 calls, warm): 1280x720 JPEG median 50 ms p95 53 ms; 1920x1080 median 47 ms p95 53 ms
  (1080p verdict PASS 0.886, same as 720p). 4 parallel requests all answered. First call after start ~115 ms.
- `README.md` (setup, run, API, recapture + retrain, layout).
- Heatmap overlay fades in smoothly (no hard-edged tint); `docs/heatmap_examples.png` (crack, burr, good, unsure).

- `docs/vision.md` explainer (14 sections, histogram + heatmap images) rendered to `docs/NAZAR-02-Vision-AI.pdf`
  (7 pages, checked page by page) and copied to `../PDFs/NAZAR-02-Vision-AI.pdf`.
- Re-evaluated on an idle machine: in-process median 45 ms, p95 52 ms.

## Task 2: iPhone photos (primary) + videos -> training images
### Done
- `nazar_vision/ingest.py` + `ingest.sh` (commit 838946c): photos HEIC (sips -> JPEG, primary/SDR image) / JPG / PNG,
  EXIF orientation applied, EXIF capture time, bursts (< 2 s, centre within 6 % r) = one group; videos via ffmpeg
  (4 fps, HLG/PQ tone-map, rotation metadata); filters: one washer, inside frame (10 % r margin), rim sharpness,
  near-copies, 12/clip cap, size outliers; manifest `data/footage/manifest.csv`;
  per-source records in `data/footage/_records/` so reruns skip finished sources.
- Locator crash fixed (flat ray -> IndexError) in `locate.py` (commit 6c2c920) - also protects the live service.
- `nazar_vision/fake_footage.py` (FAKE test data only).
- Photo path tested on a FAKE set (scratchpad, not in repo): 21 good files -> 16 kept. Burst of 3 -> 1 kept (2 near-copy),
  blurred -> dropped, washer cut by edge -> dropped, two washers -> dropped, orientation-6 HEIC stored sideways ->
  upright (mean abs diff 0.74 vs 50.8 if wrong), 48 MP HEIC -> 1920x1080, PNG without EXIF kept; 2 fake chipped
  defects + 1 empty belt kept. White-pixel warning fires (77 %). Rerun skips all sources (0.17 s).

- Pipeline wired in: `split.py` (footage groups; data/good first set skipped when footage exists; REAL defects
  alternate REJECT-set/test, a single one sets REJECT; empty belt), `train.py` (<= 600 crops, <= 250k patches,
  bank <= 20k, live size range 0.5-2x), `evaluate.py` (REAL vs SYNTHETIC rows, real empty belt, distances in washer
  radii), `retrain.sh` (stage `ingest` first when footage/ or earlier ingests exist).
- Found: this Mac's ffmpeg (~/Library/Python/3.14/bin, Remotion build) has NO fps/select filters -> frames sampled
  with `-r`. libx265 there is 8-bit only -> fake 10-bit clips use hevc_videotoolbox Main10.
- HLG tone-map chosen by test: zscale npl=203 + tonemap mobius (mean abs diff 13.6 vs SDR source; hable 65.6).
- Blur judged by rim-edge sharpness (`locate.edge_sharpness`, threshold 0.22): Laplacian variance called a smooth,
  well-exposed HLG washer blurry (13-21 < 15).
- END-TO-END on FAKE footage (`./retrain.sh`, 2.5 min): 10 good clips (1 HEVC Main10 HLG, 1 HEVC Main10 portrait
  rotation 90, 8 H.264) + 2 FAKE crack clips (HLG) + empty clip + the fake photo set. Kept 7/42 frames per good clip
  (30 near-copies, 5 motion frames), 6/34 per crack clip, empty capped 6/14. Split 86 good images / 26 placements.
  REJECT = 1.9636 by the midway rule from 7 REAL-labelled (fake) defects; held-out REAL 7/7 REJECT, synthetic 25/25,
  real empty 7/7 NO_PART, good test 4 PASS 1 HOLD; 49 ms median. Rerun of ingest skips all sources (0.2 s).
- Fake footage deleted (footage/ removed, `./ingest.sh --reset`), baseline retrained: same HOLD 1.3469 / REJECT 1.8184,
  same confusion matrix; REPORT.md regenerated with the new wording.

- README "Recording washers with the iPhone" (photo tips first, video tips) + "Ingest, retrain, check" + resolution
  notes; docs/vision.md section 11 rewritten (taking photos, what ingest does, tested how) + 2 new unverified items;
  PDF re-rendered (8 pages, checked) and copied to ../PDFs/NAZAR-02-Vision-AI.pdf.

### In progress
- Nothing. Task 2 complete; footage/ and data/footage/ removed, baseline model in models/.

## Next (on site)
- iPhone photos arrive: `./ingest.sh <folder> --label good` (and defect/empty), `./retrain.sh`, read `REPORT.md`,
  check a few live good washers through Continuity Camera.
- Attach `models/backbone.onnx` (103 MB), `backbone.json`, `memory_bank.npy` (15.6 MB), `config.json` to a GitHub release.
