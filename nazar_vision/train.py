"""Build the memory bank from good washers and set the PASS / HOLD / REJECT lines.

    python -m nazar_vision.train            # reads data/split.json, writes models/memory_bank.npy + config.json

1. Every training photo: find the washer, crop it, and make 12 copies turned 30 deg apart with the brightness
   scaled by a random 0.9-1.1 (a washer can lie at any angle; the light drifts a little). With many photos
   (iPhone footage) fewer copies are made so there are at most 600 crops.
2. Backbone -> 784 patch vectors per crop. Patches far outside the ring are dropped (they never get scored);
   with very many crops a random subset (<= 250,000 patches in all) goes on to the coreset.
3. Greedy k-center coreset keeps ~10 % of the patches: the memory bank.
4. HOLD line = conformal quantile of the calibration scores (good photos the bank has never seen):
   with n scores, the ceil((n+1)(1-alpha))-th smallest, so that a new good washer lands above it with
   probability <= alpha (alpha = 0.1). With few photos this is simply the largest calibration score.
5. REJECT line = halfway between HOLD and the lowest score of the real damaged photos kept for this
   (data/defect/), or HOLD x 1.35 when there are none.
"""
import argparse
import json
import math
import time

import cv2
import numpy as np

from . import CROP, GRID, MODEL_NAME, MODELS, ROOT, VERSION, images, locate
from .features import Backbone
from .inspector import Inspector
from .memory import kcenter_coreset
from .split import load_split

SIGMA = 4.0
MASK_OUTER = 1.04       # score up to 4 % beyond the rim (a burr sticking out still counts)
MASK_INNER = 0.70       # ignore the middle of the hole (belt seen through it) up to 0.7 x hole radius
ALPHA = 0.10
REJECT_FACTOR = 1.35


def conformal_hold(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores, np.float64))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[min(k, len(s)) - 1])


def bank_args(argv=None, rotations=12):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coreset", type=float, default=0.10, help="fraction of patches kept (default 0.10)")
    ap.add_argument("--rotations", type=int, default=rotations)
    ap.add_argument("--max-bank", type=int, default=20000)
    ap.add_argument("--max-crops", type=int, default=600, help="cap on training crops (photos x rotations)")
    ap.add_argument("--max-patches", type=int, default=250000, help="random subsample before the coreset")
    ap.add_argument("--camera-aug", type=float, default=0.0,
                    help="share of crops given a random camera change (light, contrast, focus, resolution)")
    return ap


def find_washers(rels, log=print):
    """Step 1: [(img, washer)] for the photos where a washer is found."""
    found = []
    for rel in rels:
        img = images.load(ROOT / rel)
        w = locate.find(img)
        if w is None:
            log(f"  WARNING {rel}: no washer found, not used")
            continue
        found.append((img, w))
    return found


def build_bank(found, args, log=print):
    """Steps 1-3 for a list of (img, washer): geometry, augmented crops -> patches, coreset.
    -> (bank float32 as the service will load it, cfg without hold / reject)."""
    if len(found) < 3:
        raise SystemExit("fewer than 3 training photos with a washer found")
    geo = np.array([(w.r / min(img.shape[:2]), (w.hole / w.r) if w.hole else np.nan, w.contrast) for img, w in found])
    radius_frac = float(np.median(geo[:, 0]))
    hole_frac = float(np.nanmedian(geo[:, 1])) if np.isfinite(geo[:, 1]).any() else None
    polarity = int(np.sign(np.median(geo[:, 2])))
    min_contrast = float(max(20.0, 0.4 * np.median(np.abs(geo[:, 2]))))
    log(f"train: {len(found)} photos, rim radius {radius_frac:.4f} x short side, hole {hole_frac:.3f} x rim, "
        f"ring {'brighter' if polarity > 0 else 'darker'} than belt by {np.median(np.abs(geo[:, 2])):.0f}")

    # ---- 2. augmented crops -> patches inside the scored region
    rng = np.random.default_rng(0)
    region = locate.ring_mask(hole_frac, outer=MASK_OUTER + 0.12, inner=MASK_INNER * 0.6)
    keep = cv2.resize(region, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1) > 0.02
    # many photos (iPhone footage) -> fewer synthetic rotations each, so the crops stay <= max_crops
    rotations = max(1, min(args.rotations, args.max_crops // len(found)))
    jobs = [(i, a) for i in range(len(found)) for a in np.arange(rotations) * (360.0 / rotations)]
    if len(jobs) > args.max_crops:
        jobs = [jobs[k] for k in sorted(rng.choice(len(jobs), args.max_crops, replace=False))]
    per_crop = min(int(keep.sum()), max(1, args.max_patches // len(jobs)))
    t0 = time.time()
    parts = []
    for b in range(0, len(jobs), 32):                        # stream: never hold all crops' features at once
        crops = [locate.crop(found[i][0], found[i][1], angle=a, gain=float(rng.uniform(0.9, 1.1)))
                 for i, a in jobs[b:b + 32]]
        if args.camera_aug:
            crops = [locate.camera_change(c, rng) if rng.random() < args.camera_aug else c for c in crops]
        f = backbone().embed(crops)[:, keep]
        if per_crop < f.shape[1]:
            f = np.stack([x[rng.choice(len(x), per_crop, replace=False)] for x in f])
        parts.append(f.reshape(-1, f.shape[-1]))
    feats = np.ascontiguousarray(np.concatenate(parts))
    log(f"features: {len(jobs)} crops ({len(found)} photos x {rotations} rotations) -> {len(feats)} patches "
        f"({per_crop} of {GRID * GRID} per crop) in {time.time() - t0:.1f} s")

    # ---- 3. coreset
    k = min(args.max_bank, max(1000, int(len(feats) * args.coreset)))
    t0 = time.time()
    bank = feats[kcenter_coreset(feats, k, log=log)]
    log(f"coreset: kept {len(bank)} of {len(feats)} patches in {time.time() - t0:.1f} s")
    bank = bank.astype(np.float16).astype(np.float32)        # score with exactly what the service will load
    cfg = {"model": MODEL_NAME, "version": VERSION, "dim": int(feats.shape[-1]), "grid": GRID, "crop": CROP,
           "margin": locate.MARGIN, "sigma": SIGMA, "mask_outer": MASK_OUTER, "mask_inner": MASK_INNER,
           "radius_frac": round(radius_frac, 5), "hole_frac": round(hole_frac, 4) if hole_frac else None,
           "polarity": polarity, "min_contrast": round(min_contrast, 1), "bank": int(len(bank)),
           "radius_range": [0.5, 2.0],      # live washer may look 0.5-2x the training size (other lens / zoom)
           "hold": 0.0, "reject": 0.0, "n_train": len(found), "rotations": rotations,
           "coreset": args.coreset, "patches_before_coreset": int(len(feats)), "camera_aug": args.camera_aug}
    return bank, cfg


_backbone = None


def backbone():
    global _backbone
    if _backbone is None:
        _backbone = Backbone(MODELS / "backbone.onnx")
    return _backbone


def main():
    args = bank_args().parse_args()
    t_start = time.time()
    split = load_split()
    bank, cfg = build_bank(find_washers(split["train"]), args)
    MODELS.mkdir(exist_ok=True)
    np.save(MODELS / "memory_bank.npy", bank.astype(np.float16))
    insp = Inspector(cfg, backbone(), bank)

    # ---- 4. HOLD from calibration photos
    def scores(rels):
        out = {}
        for rel in rels:
            img = images.load(ROOT / rel)
            w = insp.find(img)
            out[rel] = None if w is None else insp.score_crop(locate.crop(img, w))[0]
        return out

    calib = scores(split["calib"])
    good_scores = [v for v in calib.values() if v is not None]
    if len(good_scores) < 3:
        raise SystemExit("fewer than 3 calibration photos with a washer found")
    hold = conformal_hold(good_scores)
    n = len(good_scores)
    k_rank = math.ceil((n + 1) * (1 - ALPHA))
    hold_rule = (f"conformal: score rank {min(k_rank, n)} of {n} calibration photos (alpha {ALPHA})"
                 + (" = the largest, too few photos for a tighter rule" if k_rank > n else ""))

    # ---- 5. REJECT from real defects, if any
    defect = {}
    for kind, rels in split["defect"].items():
        defect.update({rel: s for rel, s in scores(rels).items()})
    dvals = [v for v in defect.values() if v is not None]
    if dvals and min(dvals) > hold:
        reject, reject_rule = (hold + min(dvals)) / 2, f"halfway between hold and the lowest of {len(dvals)} real defect scores"
    else:
        reject, reject_rule = hold * REJECT_FACTOR, f"hold x {REJECT_FACTOR}" + (
            " (real defects scored below hold!)" if dvals else " (no real defect photos yet)")

    cfg.update(hold=math.ceil(hold * 1e4) / 1e4, reject=math.ceil(reject * 1e4) / 1e4, hold_rule=hold_rule, reject_rule=reject_rule,
               calib_scores={k: v and round(v, 4) for k, v in calib.items()},
               defect_scores={k: v and round(v, 4) for k, v in defect.items()}, sources=split.get("sources", ""),
               trained=time.strftime("%Y-%m-%dT%H:%M:%S"), train_seconds=round(time.time() - t_start, 1))
    (MODELS / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"calibration scores: {[round(v, 4) for v in sorted(good_scores)]}")
    print(f"HOLD   = {hold:.4f}  ({hold_rule})")
    print(f"REJECT = {reject:.4f}  ({reject_rule})")
    print(f"wrote models/memory_bank.npy ({(MODELS / 'memory_bank.npy').stat().st_size / 1e6:.1f} MB) and "
          f"models/config.json in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
