"""Switch the model to "trained damage only": REJECT just the kinds of damage it was trained on, pass everything else.

    python -m nazar_vision.match        # after train_small + damage; sets match_reject / match_heat in models/config.json

Demo rule (3 Nov): only the two deliberately damaged washers (data/footage/defect/dent, defect/scar) may be rejected;
good washers and washers with small marks (data/footage/excluded) must pass, also through the line camera, whose
light and focus make every washer look a little unusual to a plain anomaly score.

Calibration: every good, excluded and defect photo is taken through fixed camera changes (darker, brighter, contrast
curves, blur, low resolution, a mix) at angles between the library's turned copies. The evidence of each crop is its
summed per-patch "closer to the damage library than to any good patch" (Inspector.evidence). match_reject is twice
the highest pass-side evidence (false rejects matter most in the demo). Writes MATCH-REPORT.md.
"""
import json
import math

import cv2
import numpy as np

from . import DATA, MODELS, ROOT, images, locate
from .inspector import Inspector, anomaly_map

FOOT = DATA / "footage"
CHANGES = [("as taken", 1.0, 1.0, 0.0, 1.0), ("darker x0.75", 0.75, 1.0, 0.0, 1.0), ("brighter x1.3", 1.3, 1.0, 0.0, 1.0),
           ("gamma 0.8", 1.0, 0.8, 0.0, 1.0), ("gamma 1.25", 1.0, 1.25, 0.0, 1.0), ("blur 1.2 px", 1.0, 1.0, 1.2, 1.0),
           ("resolution x0.35", 1.0, 1.0, 0.0, 0.35), ("mix", 1.2, 0.85, 1.0, 0.5)]
ANGLES = (15, 105, 195, 285)


def change(img, gain, gamma, blur, down):
    x = (np.clip((img.astype(np.float32) / 255) ** gamma * gain, 0, 1) * 255).astype(np.uint8)
    if down < 1:
        h, w = x.shape[:2]
        small = cv2.resize(x, (int(w * down), int(h * down)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(x, (0, 0), blur) if blur else x


def main():
    insp = Inspector.load()
    if not insp.library:
        raise SystemExit("no damage library: run python -m nazar_vision.damage first")
    sets = [("good", p) for p in images.list_images(FOOT / "good")]
    sets += [("pass: small marks", p) for p in images.list_images(FOOT / "excluded")]
    sets += [(f"defect: {p.parent.parent.name}", p) for p in images.list_images(FOOT / "defect")]
    rows = []                                 # (set, photo, change, evidence, evidence-map peak)
    for label, p in sets:
        img0 = images.load(p)
        for name, *c in CHANGES:
            img = change(img0, *c)
            w = insp.find(img)
            if w is None:
                print(f"  {p.name} {name}: no washer found")
                continue
            for a in ANGLES:
                f = insp.backbone.embed([locate.crop(img, w, angle=a)])[0]
                ev = insp.evidence(f, insp.memory.nearest(f))
                peak = float((anomaly_map(ev, insp.cfg["sigma"]) * insp.mask).max())
                rows.append((label, p.name.split("-")[0], name, float(ev.sum()), peak))
        print(f"  {label:18s} {p.name.split('-')[0]}: done")
    neg = [r[3] for r in rows if not r[0].startswith("defect")]
    pos = [r[3] for r in rows if r[0].startswith("defect")]
    hi_neg, lo_pos = max(neg), min(pos)
    line, rule = 2.0 * hi_neg, "twice the highest pass-side evidence (a good or lightly marked washer must not be rejected)"
    if line >= np.percentile(pos, 25):     # too close to the defects: meet in the middle instead
        line, rule = math.sqrt(max(hi_neg, 1e-3) * float(np.median(pos))), "geometric mean of the highest pass-side and the median defect evidence"
    heat = 0.6 * float(np.median([r[4] for r in rows if r[0].startswith("defect")]))
    cfg_path = MODELS / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(match_reject=round(line, 4), match_heat=round(heat, 4), match_rule=rule,
               decide="trained damage only: " + ", ".join(insp.library.kinds))
    cfg_path.write_text(json.dumps(cfg, indent=2))
    write_report(rows, line, rule)
    print(f"match_reject {line:.2f} ({rule}); pass side max {hi_neg:.2f}, defect min {lo_pos:.2f}; heat {heat:.3f}")


def write_report(rows, line, rule):
    L = ["# Trained damage only: calibration", "",
         f"REJECT when the summed evidence is at least **{line:.2f}** ({rule}). Score = evidence / {line:.2f}, "
         "so 1.00 is the reject line. No HOLD band.", "",
         "| Set | Photo | " + " | ".join(c[0] for c in CHANGES) + " |", "|---|---|" + "---|" * len(CHANGES)]
    for key in dict.fromkeys((r[0], r[1]) for r in rows):
        cells = []
        for c in CHANGES:
            v = [r[3] / line for r in rows if (r[0], r[1]) == key and r[2] == c[0]]
            cells.append(f"{min(v):.2f}-{max(v):.2f}" if v else "-")
        L.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    L += ["", "Scores (min-max over 4 angles between the library's turned copies). Below 1.00 = PASS.", ""]
    (ROOT / "MATCH-REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
