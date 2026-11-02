"""Train from a handful of good washers (fewer than the 9 placements the normal split needs).

    python -m nazar_vision.train_small            # good + defect photos from ingest (data/footage/)
    python -m nazar_vision.train_small --unsure DIR   # also score photos nobody is sure about (never trained on)

With 4-8 good placements there is nothing left over for a separate calibration and test set, so:
1. Leave-one-out: for every good photo, a bank is built from the OTHER good photos and that photo is scored.
   Every good photo thus gets an honest "never seen" score. HOLD = the largest of them (conformal rule with
   n < 9 is simply the maximum).
2. The final bank uses all good photos (a new washer is compared with every good washer we have).
3. REJECT = halfway between HOLD and the lowest real-defect score, as in train.py. Each defect is also checked
   leave-one-out: would it still be REJECT if the line had been set from the other defects only?
4. Synthetic defects (notch, burr, crack, spot, scratch) are drawn on each good photo and scored against the
   bank built WITHOUT that photo, to show the model also catches kinds of damage it has never seen.
Writes models/memory_bank.npy, models/config.json and SMALL-DATA-REPORT.md.
"""
import json
import math
import time

import numpy as np

from . import DATA, MODELS, ROOT, images, locate, synth
from .inspector import Inspector
from .train import ALPHA, REJECT_FACTOR, backbone, bank_args, build_bank, conformal_hold, find_washers

FOOT = DATA / "footage"


def rel(p):
    return str(p.relative_to(ROOT))


def main():
    ap = bank_args(rotations=24)          # few photos: more turned copies each
    ap.add_argument("--unsure", help="folder of photos to score only (never trained on, never sets a line)")
    args = ap.parse_args()
    t_start = time.time()
    quiet = lambda *a, **k: None

    goods = [rel(p) for p in images.list_images(FOOT / "good")]
    defects = {d.name: [rel(p) for p in images.list_images(d)] for d in sorted((FOOT / "defect").glob("*")) if d.is_dir()}
    found = find_washers(goods)
    names = [g.split("/")[-1] for g in goods]
    if len(found) != len(goods):
        raise SystemExit("a good photo has no washer found: check it before training")
    if len(found) < 4:
        raise SystemExit("need at least 4 good photos for leave-one-out")
    print(f"{len(found)} good photos, {sum(len(v) for v in defects.values())} defect photos")

    # ---- 1. leave-one-out scores of the good photos, plus synthetic damage on each held-out photo
    rng = np.random.default_rng(1)
    loo, synth_scores = {}, []
    for i, (img, w) in enumerate(found):
        bank, cfg = build_bank(found[:i] + found[i + 1:], args, log=quiet)
        insp = Inspector(cfg, backbone(), bank)
        loo[names[i]] = insp.score_crop(locate.crop(img, w))[0]
        for kind in synth.KINDS:
            bad = synth.draw(img.copy(), w, kind, rng)
            ww = insp.find(bad) or w
            synth_scores.append((names[i], kind, insp.score_crop(locate.crop(bad, ww))[0]))
        print(f"  leave out {names[i]}: score {loo[names[i]]:.4f}")
    hold = conformal_hold(list(loo.values()))

    # ---- 2. final bank from every good photo
    bank, cfg = build_bank(found, args)
    cfg["radius_range"] = [0.4, 3.0]      # the live view (Camo crop) frames the washer tighter than the photos
    insp = Inspector(cfg, backbone(), bank)
    MODELS.mkdir(exist_ok=True)
    np.save(MODELS / "memory_bank.npy", bank.astype(np.float16))

    def score_files(rels):
        out = {}
        for r in rels:
            img = images.load(ROOT / r)
            w = insp.find(img)
            out[r.split("/")[-1]] = None if w is None else insp.score_crop(locate.crop(img, w))[0]
        return out

    # ---- 3. REJECT from real defects, checked leave-one-out
    dscore = {}
    for kind, rels in defects.items():
        for name, s in score_files(rels).items():
            dscore[name] = (kind, s)
    dvals = [s for _, s in dscore.values() if s is not None]
    if dvals and min(dvals) > hold:
        reject = (hold + min(dvals)) / 2
        reject_rule = f"halfway between HOLD and the lowest of {len(dvals)} real defect scores"
    else:
        reject = hold * REJECT_FACTOR
        reject_rule = f"HOLD x {REJECT_FACTOR}" + (" (a real defect scored below HOLD!)" if dvals else " (no real defects)")
    loo_defect = {}
    for name, (kind, s) in dscore.items():
        others = [v for n, (_, v) in dscore.items() if n != name and v is not None]
        line = (hold + min(others)) / 2 if others and min(others) > hold else hold * REJECT_FACTOR
        loo_defect[name] = None if s is None else ("REJECT" if s >= line else "HOLD" if s > hold else "PASS")

    unsure = {}
    if args.unsure:
        from pathlib import Path
        for p in images.list_images(Path(args.unsure).expanduser()):
            img = images.load(p)
            w = insp.find(img)
            unsure[p.name] = None if w is None else insp.score_crop(locate.crop(img, w))[0]

    cfg.update(hold=math.ceil(hold * 1e4) / 1e4, reject=math.ceil(reject * 1e4) / 1e4,
               hold_rule=f"largest of {len(loo)} leave-one-out good scores (alpha {ALPHA}, n < 9)",
               reject_rule=reject_rule, mode="small-data leave-one-out",
               calib_scores={k: round(v, 4) for k, v in loo.items()},
               defect_scores={k: s and round(s, 4) for k, (_, s) in dscore.items()},
               sources="iPhone photos (data/footage)", trained=time.strftime("%Y-%m-%dT%H:%M:%S"),
               train_seconds=round(time.time() - t_start, 1))
    (MODELS / "config.json").write_text(json.dumps(cfg, indent=2))
    write_report(cfg, loo, dscore, loo_defect, synth_scores, unsure)
    print(f"HOLD {cfg['hold']}  REJECT {cfg['reject']}  ({reject_rule}); wrote models/ and SMALL-DATA-REPORT.md "
          f"in {time.time() - t_start:.0f} s")


def verdict(s, cfg):
    return "no washer" if s is None else "PASS" if s <= cfg["hold"] else "HOLD" if s < cfg["reject"] else "REJECT"


def write_report(cfg, loo, dscore, loo_defect, synth_scores, unsure):
    h, r = cfg["hold"], cfg["reject"]
    L = [f"# Small-data model ({time.strftime('%d %b %Y %H:%M')})", "",
         f"Trained on {len(loo)} good iPhone photos ({cfg['n_train']} washers x {cfg['rotations']} turned copies, "
         f"bank {cfg['bank']} patches). HOLD = {h} ({cfg['hold_rule']}); REJECT = {r} ({cfg['reject_rule']}).", "",
         "## Good washers, each scored by a model that never saw it (leave-one-out)", "",
         "| Photo | Score | Verdict |", "|---|---|---|"]
    L += [f"| {n} | {s:.3f} | {verdict(s, cfg)} |" for n, s in sorted(loo.items())]
    L += ["", "All of these are PASS by construction (HOLD is their maximum); the spread shows how much room there is.", "",
          "## Real damaged washers (final model)", "",
          "| Photo | Kind | Score | Verdict | Verdict if its own score had not set REJECT |", "|---|---|---|---|---|"]
    L += [f"| {n} | {k} | {'-' if s is None else f'{s:.3f}'} | {verdict(s, cfg)} | {loo_defect[n]} |" for n, (k, s) in sorted(dscore.items())]
    caught = sum(v == "REJECT" for v in loo_defect.values())
    L += ["", f"Leave-one-out on the defects: {caught} of {len(loo_defect)} still REJECT.", "",
          "## Synthetic damage on held-out good photos (kinds never trained on)", "",
          "| Kind | Lowest score | REJECT | HOLD | PASS |", "|---|---|---|---|---|"]
    for kind in synth.KINDS:
        v = [s for _, k, s in synth_scores if k == kind]
        L.append(f"| {kind} | {min(v):.3f} | {sum(s >= r for s in v)} | {sum(h < s < r for s in v)} | {sum(s <= h for s in v)} |")
    if unsure:
        L += ["", "## Unsure photos (not trained on, did not set any line)", "", "| Photo | Score | Verdict |", "|---|---|---|"]
        L += [f"| {n} | {'-' if s is None else f'{s:.3f}'} | {verdict(s, cfg)} |" for n, s in sorted(unsure.items())]
    L += ["", "## Limits of this model", "",
          f"- Only {len(loo)} good washers. A new good washer that looks unlike all of them can get HOLD. "
          "Add 20-30 more good photos and rerun to make the HOLD line trustworthy.",
          "- Photos come from the iPhone Camera app; the live picture comes through Camo (720p, its own exposure). "
          "Check a few good washers live: if they score above HOLD, retrain from Camo snapshots.",
          "- REJECT was set from the same damaged photos it is tested on; the leave-one-out column is the honest check."]
    (ROOT / "SMALL-DATA-REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
