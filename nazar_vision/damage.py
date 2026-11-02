"""Name the damage on a flagged washer: what kind, where on the washer, how many spots.

    python -m nazar_vision.damage        # builds models/damage_library.npz from data/footage/defect/<kind>/

The anomaly map already says WHERE a washer differs from the good ones. To say WHAT it is, the patches that
light up on each damaged training washer are kept per kind (the folder name: dent, scar, ...), turned 12 ways.
On a new flagged washer the patches that light up are compared with each kind's patches, and the closest kind
names the damage. With one washer per kind this recognises damage that looks like those washers; other damage
is still flagged by the score, it only gets the nearest name (the match value shows how close it was).

Where: the clock position of the strongest spot as the camera sees it (12 = top of the picture) and the zone
(hole edge, face, outer rim). Spots: separate regions above the PASS line.
"""
import json
import math
import time

import cv2
import numpy as np

from . import CROP, DATA, GRID, MODELS, images, locate
from .memory import Memory

FOOT = DATA / "footage"
LIBRARY = MODELS / "damage_library.npz"
KIND_TEXT = {"dent": "Dent", "scar": "Scar", "scratch": "Scratch", "chipped": "Chipped edge", "rust": "Rust",
             "burr": "Burr", "crack": "Crack", "wrong-size": "Wrong size", "other": "Damage"}
MIN_SPOT_PX = 30           # regions smaller than this (of a 224 x 224 crop) are noise
PER_CROP = 40              # brightest patches kept per turned copy of a damaged washer


class Library:
    def __init__(self, feats, kinds_of_rows):
        self.kinds = sorted(set(kinds_of_rows))
        rows = np.asarray(kinds_of_rows)
        self.memory = {k: Memory(feats[rows == k]) for k in self.kinds}

    @classmethod
    def load(cls, path=LIBRARY):
        if not path.is_file():
            return None
        z = np.load(path)
        return cls(z["feats"].astype(np.float32), [str(k) for k in z["kind"]])


def _region(amap, mask, thr):
    return ((amap >= thr) & (mask > 0.5)).astype(np.uint8)


def describe(cfg, amap, mask, feats, score, library):
    """-> {kind, label, spots, clock, zone, where, area_pct, match, text} for a flagged washer."""
    thr = max(cfg["hold"], 0.85 * score)
    region = _region(amap, mask, thr)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    spots = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_SPOT_PX]
    peak_yx = np.unravel_index(np.argmax(amap * (mask > 0.5)), amap.shape)

    # where: clock position and zone of the strongest point
    c = CROP / 2
    rim = c / (1 + locate.MARGIN)
    hole = (cfg.get("hole_frac") or 0.4) * rim
    dy, dx = peak_yx[0] - c, peak_yx[1] - c
    hour = round(math.degrees(math.atan2(dx, -dy)) % 360 / 30) % 12 or 12
    r = math.hypot(dx, dy)
    band = rim - hole
    zone = "hole edge" if r < hole + 0.25 * band else "outer rim" if r > rim - 0.18 * band else "face"
    ring_px = float((mask > 0.5).sum()) or 1.0
    area_pct = round(100.0 * float(region.sum()) / ring_px, 1)

    # what: nearest kind of the patches inside the damaged region
    kind, match = "other", None
    if library is not None and library.kinds:
        cells = cv2.resize(region.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1) > 0.3
        if not cells.any():
            cells = np.zeros(GRID * GRID, bool)
            cells[int(peak_yx[0] * GRID / CROP) * GRID + int(peak_yx[1] * GRID / CROP)] = True
        q = feats[cells]
        d = {k: float(np.mean(m.nearest(q))) for k, m in library.memory.items()}
        order = sorted(d, key=d.get)
        kind = order[0]
        if len(order) > 1:
            match = round(d[order[1]] / (d[order[0]] + d[order[1]]), 2)      # 0.5 = undecided, 1 = certain
    label = KIND_TEXT.get(kind, kind.capitalize())
    where = f"{zone}, {hour} o'clock"
    count = max(1, len(spots))
    text = f"{label}{f' x{count}' if count > 1 else ''} at {where}"
    return {"kind": kind, "label": label, "spots": count, "clock": hour, "zone": zone, "where": where,
            "area_pct": area_pct, "match": match, "text": text}


def build(rotations=12, log=print):
    """Damage library from data/footage/defect/<kind>/ scored against the current memory bank."""
    from .inspector import Inspector
    t0 = time.time()
    insp = Inspector.load()
    grid_mask = cv2.resize(insp.mask, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1) > 0.5
    feats, kinds, seen = [], [], {}
    for kdir in sorted(d for d in (FOOT / "defect").glob("*") if d.is_dir()):
        for p in images.list_images(kdir):
            img = images.load(p)
            w = insp.find(img)
            if w is None:
                log(f"  {p.name}: no washer found, skipped")
                continue
            for a in np.arange(rotations) * (360.0 / rotations):
                f = insp.backbone.embed([locate.crop(img, w, angle=a)])[0]
                dist = insp.memory.nearest(f)
                dist[~grid_mask] = -1
                top = np.argsort(dist)[-PER_CROP:]
                feats.append(f[top])
                kinds += [kdir.name] * len(top)
            seen[kdir.name] = seen.get(kdir.name, 0) + 1
    if not feats:
        raise SystemExit("no damaged washers in data/footage/defect/<kind>/")
    np.savez_compressed(LIBRARY, feats=np.concatenate(feats).astype(np.float16), kind=np.array(kinds))
    cfg_path = MODELS / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["damage_kinds"] = seen
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"damage library: {seen} ({len(kinds)} patches) in {time.time() - t0:.0f} s -> {LIBRARY.name}")


if __name__ == "__main__":
    build()
