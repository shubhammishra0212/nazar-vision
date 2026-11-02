"""Split the photos into train / calibration / test, by PLACEMENT, not by photo.

    python -m nazar_vision.split            # writes data/split.json (no files are copied)

Sources (any may be empty):
    data/footage/good/<group>/                          good washers from iPhone photos/videos (nazar_vision.ingest)
    data/capture/good/                                  good washers captured through the console (/capture)
    data/good/                                          the first photo set (webcam); NOT used when footage exists,
                                                        because it comes from a different camera (--with-first-set)
    data/footage/defect/<kind>/<group>/, data/defect/<kind>/, data/capture/defect/<kind>/   REAL damaged washers
    data/footage/empty/<group>/                         belt with no washer (evaluation: must be NO_PART)
    data/unsure/                                        reported, never trained on

Several photos of the same washer lying in the same spot are near-copies. If one is in training and the
other in the test, the test is too easy. So photos are grouped first and whole groups are assigned:
  - byte-identical files are dropped (only the first is kept);
  - ingested footage: the group folder (one video clip, one photo, or one photo burst);
  - data/placements.csv (file,placement), if present, names the group by hand;
  - otherwise photos whose small greyscale versions are near-identical (mean difference < 0.35 grey levels,
    i.e. the camera noise of a washer that was not moved) share a group.
Good groups: shuffled with a fixed seed, then ordered smallest first; the test takes 20 % of the groups
(at least 3), calibration the next 20 % (at least 3), the rest (the biggest groups) trains the memory bank.
At least 9 good placements are needed. Defects: per kind, groups alternate between data used to set
the REJECT line and the test (the REJECT set gets the odd one out, so a single real defect still sets REJECT).
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
FOOTAGE = (DATA / "footage").resolve()


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

    def fixed(p):
        if FOOTAGE in p.resolve().parents:
            return "footage:" + str(p.parent.resolve().relative_to(FOOTAGE))
        return hand.get(p.name)

    by_hand = {}
    for i, p in enumerate(uniq):
        key = fixed(p)
        if key:
            j = by_hand.setdefault(key, i)
            parent[root(i)] = root(j)
    auto = [i for i, p in enumerate(uniq) if not fixed(p)]
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


def make_split(seed=0, test_frac=0.2, calib_frac=0.2, with_first_set=False, log=print):
    hand = {}
    pl = DATA / "placements.csv"
    if pl.is_file():
        with pl.open(newline="") as f:
            hand = {r["file"].strip(): r["placement"].strip() for r in csv.DictReader(f) if r.get("placement")}
    dupes = []
    footage_good = images.list_images(DATA / "footage" / "good")
    first_set = images.list_images(DATA / "good")
    use_first = with_first_set or not footage_good
    if footage_good and not use_first and first_set:
        log(f"data/good ({len(first_set)} photos, first set from another camera) not used: footage exists "
            "(--with-first-set to include it)")
    good = group(footage_good + images.list_images(DATA / "capture" / "good") + (first_set if use_first else []),
                 hand, dupes)
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
    for base in (DATA / "footage" / "defect", DATA / "defect", DATA / "capture" / "defect"):
        for p in images.list_images(base):
            parts = p.relative_to(base).parts
            kinds.setdefault(parts[0].lower() if len(parts) > 1 else "other", []).append(p)
    pool = []                                    # all defect groups, kind by kind, shuffled inside a kind
    for kind, paths in sorted(kinds.items()):
        gs = group(paths, hand, dupes)
        rng.shuffle(gs)
        pool += [(kind, g) for g in gs]
    defect, test_defect = {}, {}
    for i, (kind, g) in enumerate(pool):         # alternate: REJECT set, test, REJECT set, ...
        (defect if i % 2 == 0 else test_defect).setdefault(kind, []).append(g)

    flat = lambda gs: [_rel(p) for g in gs for p in g]
    out = {
        "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
        "train": flat(sets["train"]), "calib": flat(sets["calib"]), "test_good": flat(sets["test_good"]),
        "defect": {k: flat(v) for k, v in defect.items()},
        "test_defect": {k: flat(v) for k, v in test_defect.items()},
        "unsure": [_rel(p) for p in images.list_images(DATA / "unsure")],
        "empty": [_rel(p) for p in images.list_images(DATA / "footage" / "empty")],
        "sources": "footage + capture" if not use_first else "first set + capture" + (" + footage" if footage_good else ""),
        "groups": {name: len(sets[name]) for name in sets},
        "duplicates_skipped": dupes,
    }
    (DATA / "split.json").write_text(json.dumps(out, indent=1))
    log(f"good: {n_good} distinct photos in {len(good)} placements -> "
        f"train {len(out['train'])} ({len(sets['train'])} placements), calib {len(out['calib'])} "
        f"({len(sets['calib'])}), test {len(out['test_good'])} ({len(sets['test_good'])}); "
        f"{len(dupes)} byte-identical copies skipped")
    for k in sorted(set(defect) | set(test_defect)):
        log(f"REAL defect {k}: {len(out['defect'].get(k, []))} images set the reject line, "
            f"{len(out['test_defect'].get(k, []))} test")
    if not kinds:
        log("defect: none yet (reject line will be hold x 1.35)")
    if out["empty"]:
        log(f"empty belt: {len(out['empty'])} images (must give NO_PART)")
    return out


def load_split():
    p = DATA / "split.json"
    if not p.is_file():
        raise SystemExit("data/split.json missing: run  python -m nazar_vision.split  first")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-first-set", action="store_true", help="also train on data/good when footage exists")
    args = ap.parse_args()
    make_split(args.seed, with_first_set=args.with_first_set)


if __name__ == "__main__":
    main()
