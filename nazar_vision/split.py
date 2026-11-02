"""Split the photos into train / calibration / test, by PLACEMENT, not by photo.

    python -m nazar_vision.split            # writes data/split.json (no files are copied)

Sources (any may be empty):
    data/good/, data/capture/good/                      good washers
    data/defect/<kind>/, data/capture/defect/<kind>/    real damaged washers (kind = scratch, burr, ...)
    data/unsure/                                        reported, never trained on

Several photos of the same washer lying in the same spot are near-copies. If one is in training and the
other in the test, the test is too easy. So photos are grouped first and whole groups are assigned:
  - byte-identical files are dropped (only the first is kept);
  - data/placements.csv (file,placement), if present, names the group by hand;
  - otherwise photos whose small greyscale versions are near-identical (mean difference < 0.35 grey levels,
    i.e. the camera noise of a washer that was not moved) share a group.
Good groups: shuffled with a fixed seed, then ordered smallest first; the test takes 20 % of the groups
(at least 3), calibration the next 20 % (at least 3), the rest (the biggest groups) trains the memory bank.
At least 9 good placements are needed. Defects: per kind, groups alternate between data used to set
the REJECT line and the test; a kind with a single group goes to the test.
"""
import argparse
import csv
import json
import random
import time

import cv2
import numpy as np

from . import DATA, ROOT, images

NEAR_COPY = 0.35


def _rel(p):
    return str(p.resolve().relative_to(ROOT))


def _thumb(path):
    try:
        g = cv2.cvtColor(images.load(path), cv2.COLOR_BGR2GRAY)
    except ValueError:
        return None
    return cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)


def group(paths, hand, dupes):
    """-> list of groups (lists of paths). Drops byte-identical copies (recorded in dupes)."""
    seen, uniq = {}, []
    for p in paths:
        h = images.sha1(p)
        if h in seen:
            dupes.append([_rel(p), _rel(seen[h])])
        else:
            seen[h] = p
            uniq.append(p)
    parent = list(range(len(uniq)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_hand = {}
    for i, p in enumerate(uniq):
        if p.name in hand:
            j = by_hand.setdefault(hand[p.name], i)
            parent[root(i)] = root(j)
    auto = [i for i, p in enumerate(uniq) if p.name not in hand]
    thumbs = {i: _thumb(uniq[i]) for i in auto}
    for a in auto:
        for b in auto:
            if a < b and thumbs[a] is not None and thumbs[b] is not None and thumbs[a].shape == thumbs[b].shape \
                    and float(np.mean(np.abs(thumbs[a] - thumbs[b]))) < NEAR_COPY:
                parent[root(a)] = root(b)
    groups = {}
    for i, p in enumerate(uniq):
        groups.setdefault(root(i), []).append(p)
    return sorted(groups.values(), key=lambda g: g[0].name)


def make_split(seed=0, test_frac=0.2, calib_frac=0.2, log=print):
    hand = {}
    pl = DATA / "placements.csv"
    if pl.is_file():
        with pl.open(newline="") as f:
            hand = {r["file"].strip(): r["placement"].strip() for r in csv.DictReader(f) if r.get("placement")}
    dupes = []
    good = group(images.list_images(DATA / "good") + images.list_images(DATA / "capture" / "good"), hand, dupes)
    n_good = sum(len(g) for g in good)
    if len(good) < 9:
        raise SystemExit(f"only {len(good)} distinct good placements ({n_good} photos); need at least 9 "
                         "(capture more good washers, moved between photos)")
    rng = random.Random(seed)
    order = good[:]
    rng.shuffle(order)
    order.sort(key=len)          # stable: small placements go to test/calib first, big ones keep training rich
    G = len(order)
    n_test, n_cal = max(3, round(test_frac * G)), max(3, round(calib_frac * G))
    sets = {"test_good": order[:n_test], "calib": order[n_test:n_test + n_cal], "train": order[n_test + n_cal:]}

    kinds = {}
    for base in (DATA / "defect", DATA / "capture" / "defect"):
        for p in images.list_images(base):
            kind = p.parent.name if p.parent != base else "unknown"
            kinds.setdefault(kind.lower(), []).append(p)
    defect, test_defect = {}, {}
    for kind, paths in sorted(kinds.items()):
        gs = group(paths, hand, dupes)
        rng.shuffle(gs)
        if len(gs) == 1:
            test_defect[kind] = gs
        else:
            defect[kind] = gs[0::2]
            test_defect[kind] = gs[1::2]

    flat = lambda gs: [_rel(p) for g in gs for p in g]
    out = {
        "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
        "train": flat(sets["train"]), "calib": flat(sets["calib"]), "test_good": flat(sets["test_good"]),
        "defect": {k: flat(v) for k, v in defect.items()},
        "test_defect": {k: flat(v) for k, v in test_defect.items()},
        "unsure": [_rel(p) for p in images.list_images(DATA / "unsure")],
        "groups": {name: len(sets[name]) for name in sets},
        "duplicates_skipped": dupes,
    }
    (DATA / "split.json").write_text(json.dumps(out, indent=1))
    log(f"good: {n_good} distinct photos in {len(good)} placements -> "
        f"train {len(out['train'])} ({len(sets['train'])} placements), calib {len(out['calib'])} "
        f"({len(sets['calib'])}), test {len(out['test_good'])} ({len(sets['test_good'])}); "
        f"{len(dupes)} byte-identical copies skipped")
    for k in sorted(set(defect) | set(test_defect)):
        log(f"defect {k}: {len(out['defect'].get(k, []))} set the reject line, {len(out['test_defect'].get(k, []))} test")
    if not kinds:
        log("defect: none yet (reject line will be hold x 1.35)")
    return out


def load_split():
    p = DATA / "split.json"
    if not p.is_file():
        raise SystemExit("data/split.json missing: run  python -m nazar_vision.split  first")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_split(args.seed)


if __name__ == "__main__":
    main()
