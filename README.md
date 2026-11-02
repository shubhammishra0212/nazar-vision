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

## Recording washers with the iPhone

The model learns from photos of **good** washers taken by the same kind of camera, in the same light, at the same
spot as the live camera. Photos are the main path; videos work too.

**Photos (preferred)**

- Camera app, **Photo** mode at **1x**. **Live Photo OFF**, **flash OFF**.
- Stop the lens from switching: Settings > Camera > **Macro Control ON**, then in the Camera app tap the flower
  icon so macro is **off**. Otherwise the phone may jump to the ultra-wide lens close up.
- Long-press the washer for **AE/AF LOCK**, then drag the sun icon **down** until the washer face shows texture
  (grey, not pure white).
- Phone fixed at the **same height, angle and position** as the live camera in the light tunnel. Tunnel LEDs on,
  belt stopped.
- One washer per photo. Move it to a new spot and turn it between photos (6-8 photos per washer), 30-40 good washers.
  Shots less than 2 s apart with the washer not moved count as one placement.
- Every damaged washer gets its own photos, sorted by kind: scratch, dent, rust, burr, crack, chipped, wrong-size, other.
- A few photos of the empty belt.
- AirDrop to the Mac. HEIC is fine (it is converted with `sips`; the SDR image of an HDR photo is used and the
  EXIF orientation is applied).

**Videos (also supported)**

- Camera app, Video **1080p at 30 fps**, **HDR Video OFF** (Settings > Camera > Record Video). HDR clips still
  work (they are tone-mapped to SDR) but SDR is closer to the live camera.
- Same lock, exposure, position and light as for photos. One washer per clip, moved to 6-8 positions and turned,
  10-15 s per clip; every damaged washer in its own clip; one empty-belt clip.
- About 4 frames per second are taken; frames where the washer did not move, blurred frames (hand moving) and frames
  over 12 per clip are dropped. Each clip is one placement group.

## Ingest, retrain, check

```bash
# straight from an AirDrop folder (no sorting needed):
./ingest.sh ~/Downloads/good-washers --label good
./ingest.sh ~/Downloads/scratched   --label defect --kind scratch
./ingest.sh ~/Downloads/empty-belt  --label empty
# or sort files into footage/good, footage/defect/<kind>, footage/empty and let retrain.sh ingest them.

./retrain.sh          # ingest -> split -> synthetic test images -> train -> evaluate (1-3 minutes)
```

Then read `REPORT.md` and restart the service. The ingest summary prints one line per photo or clip
(kept / dropped and why, rim sharpness, % of pure-white washer pixels) and `data/footage/manifest.csv` has one row
per image. It warns when the washer face is overexposed or when there are fewer than 9 good placements.

- Real damaged washers set the **REJECT** line (halfway between HOLD and the lowest real defect score) and are
  reported as **REAL**, apart from the synthetic defects. With two or more defect placements, they alternate between
  setting the line and the held-out test.
- When footage exists, the first photo set (`data/good`, a webcam) is not used: it comes from another camera
  (`python -m nazar_vision.split --with-first-set` to include it).
- Everything is resumable: ingested files are skipped on a rerun, and `./retrain.sh train` (or `split`, `synth`,
  `evaluate`) starts from that stage. `./ingest.sh --reset` forgets all ingested images.
- Console captures (`/capture`) are used too: `data/capture/good/`, `data/capture/defect/<kind>/`.

**Resolution.** Nothing assumes a frame size. The washer is found on a copy scaled to 640 px, every limit is relative
to the image or the washer, and the crop around the washer is resized to 224 x 224. Ingested images are scaled to a
long side of at most 1920 px, like the live 1920 x 1080 frames. What does matter is **how big and how sharp the washer
looks**: if Continuity Camera uses a different lens or zoom than the Camera app, the washer is a different size in
pixels. The live check accepts 0.5-2x the training size; check a few live inspections of good washers after training.


### Only a few good washers? (small-data mode)

The normal split needs at least 9 good washers. With 4-8, run

    ./ingest.sh <folder> --label good          (and --label defect --kind ... for damaged ones)
    .venv/bin/python -m nazar_vision.train_small [--unsure <folder of photos you are not sure about>]

It scores every good washer with a model built from the others (leave-one-out) to set HOLD, builds the final
bank from all of them, sets REJECT from the real defects, and writes SMALL-DATA-REPORT.md. It is a stopgap:
add 20-30 good washers and go back to ./retrain.sh as soon as you can.

## Layout
## Layout

| File | What it does |
|---|---|
| `nazar_vision/locate.py` | finds the washer (Hough + blob candidates, circle fit, radial edge search for rim and hole), square crop, ring mask |
| `nazar_vision/features.py` | ONNX Runtime backbone: crop -> 28 x 28 patch vectors (512-d) |
| `nazar_vision/memory.py` | greedy k-center coreset, chunked nearest-neighbour distance |
| `nazar_vision/inspector.py` | one inspection: verdict, score, crop, heatmap |
| `nazar_vision/service.py` | the HTTP service |
| `nazar_vision/ingest.py` | iPhone photos / videos -> one-washer training images in `data/footage/`, manifest |
| `nazar_vision/split.py` | train / calibration / test split by placement -> `data/split.json` |
| `nazar_vision/synth.py` | SYNTHETIC defects and no-part frames, for testing only -> `data/synthetic/` |
| `nazar_vision/train.py` | memory bank + HOLD / REJECT lines -> `models/memory_bank.npy`, `models/config.json` |
| `nazar_vision/evaluate.py` | confusion matrix, latency, histogram -> `REPORT.md`, `docs/score_histogram.png`, `models/eval.json` |
| `nazar_vision/export.py` | one-off backbone export (needs torch) -> `models/backbone.onnx` |
| `nazar_vision/fake_footage.py` | FAKE photos and videos for testing ingest only |

## Credits

Backbone weights: torchvision `wide_resnet50_2` ImageNet weights (BSD-3-Clause). Method: K. Roth et al.,
"Towards Total Recall in Industrial Anomaly Detection", CVPR 2022. Re-implemented here in numpy.
