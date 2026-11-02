# nazar-vision

Washer inspection for NAZAR. The conveyor stops each washer under the camera, the NAZAR console posts the photo here,
and this service answers **PASS / HOLD / REJECT / NO_PART** with a score, the washer crop and a heatmap.
It runs offline on the Mac CPU (ONNX Runtime + numpy + OpenCV), on `127.0.0.1:8765`, as set in
`../nazar/CONTRACT.md` section 4.

Method: anomaly detection in the style of PatchCore (Roth et al., CVPR 2022). It learns only from **good** washers and
flags anything that does not look like them. A plain-language explanation is in `docs/vision.md`
(PDF: `docs/NAZAR-02-Vision-AI.pdf`). Current numbers are in `REPORT.md`.

## Set up (once)

```bash
cd nazar-vision
~/.local/bin/uv venv -p 3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt
```

The models are not in git. Put them in `models/`:

| File | Size | Where it comes from |
|---|---|---|
| `backbone.onnx` + `backbone.json` | 103 MB | GitHub release, or `python -m nazar_vision.export` in a venv that has torch + torchvision |
| `memory_bank.npy` + `config.json` | 16 MB | GitHub release, or `./retrain.sh` |

## Run

```bash
./run.sh                                   # or double-click "Start Vision.command"
./run.sh --test-images data/synthetic      # demo without a camera: each posted frame is replaced by the next photo
./run.sh --port 8775                       # another port
```

## API (CONTRACT.md section 4)

All responses are JSON with `Access-Control-Allow-Origin: *`; `OPTIONS` is answered. Errors are
`{"ok": false, "error": "..."}` with HTTP 200. The service never exits on a bad request.

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `{"ok":true,"model":"patchcore-wrn50","hold":1.347,"reject":1.8184,"bank":15206,"version":"2.0.0"}` |
| `POST /inspect` (optional `?hold=&reject=`) | JPEG or PNG bytes | `{"ok":true,"verdict":"PASS","score":0.896,"hold":..,"reject":..,"ms":50,"crop":"<base64 PNG>","heatmap":"<base64 PNG>","box":[x,y,w,h],"washer":{...}}` |
| `POST /capture?label=good\|defect&kind=scratch` | JPEG or PNG bytes | `{"ok":true,"saved":"data/capture/defect/scratch/20261102-130512-345.jpg",...}` |

- `NO_PART`: no washer found. `score` is 0, `box` is null, `heatmap` is `""` and `crop` is a small thumbnail of the
  whole frame (so the operator sees what the camera saw). It is never a defect.
- `washer` (extra): centre, rim and hole radius in frame pixels, and `size_ratio` against the training washers.
- Without a trained model the service still starts: `/capture` works, `/health` and `/inspect` return `ok:false`
  saying what is missing, and the model is loaded as soon as it appears.
- `--test-images`: a body is still required (the console posts its frame); responses carry `test_image`.

## Recapture and retrain on site

1. Fix the light first: the washer face should be grey with texture, not pure white.
2. In the console, capture good washers (`label=good`): 40 or more, each a different washer or a new spot on the belt.
   Capture every real damaged washer too (`label=defect&kind=scratch|burr|rust|crack|...`).
3. Run **one command**:

   ```bash
   ./retrain.sh          # split -> synthetic test images -> train -> evaluate, about 1 minute
   ```

4. Read `REPORT.md`, then restart the service.

If it is interrupted, resume from a stage: `./retrain.sh train` (or `synth`, `evaluate`).

Photos are read from `data/good/`, `data/capture/good/`, `data/defect/<kind>/`, `data/capture/defect/<kind>/`;
`data/unsure/` is scored in the report only. The split is by placement (see `nazar_vision/split.py`);
`data/placements.csv` (`file,placement`) can name placements by hand.

## Layout

| File | What it does |
|---|---|
| `nazar_vision/locate.py` | finds the washer (Hough + blob candidates, circle fit, radial edge search for rim and hole), square crop, ring mask |
| `nazar_vision/features.py` | ONNX Runtime backbone: crop -> 28 x 28 patch vectors (512-d) |
| `nazar_vision/memory.py` | greedy k-center coreset, chunked nearest-neighbour distance |
| `nazar_vision/inspector.py` | one inspection: verdict, score, crop, heatmap |
| `nazar_vision/service.py` | the HTTP service |
| `nazar_vision/split.py` | train / calibration / test split by placement -> `data/split.json` |
| `nazar_vision/synth.py` | SYNTHETIC defects and no-part frames, for testing only -> `data/synthetic/` |
| `nazar_vision/train.py` | memory bank + HOLD / REJECT lines -> `models/memory_bank.npy`, `models/config.json` |
| `nazar_vision/evaluate.py` | confusion matrix, latency, histogram -> `REPORT.md`, `docs/score_histogram.png`, `models/eval.json` |
| `nazar_vision/export.py` | one-off backbone export (needs torch) -> `models/backbone.onnx` |

## Credits

Backbone weights: torchvision `wide_resnet50_2` ImageNet weights (BSD-3-Clause). Method: K. Roth et al.,
"Towards Total Recall in Industrial Anomaly Detection", CVPR 2022. Re-implemented here in numpy.
