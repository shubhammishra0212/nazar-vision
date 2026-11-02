"""Score every held-out image and write the numbers: models/eval.json, docs/score_histogram.png, REPORT.md.

    python -m nazar_vision.evaluate

Sets (from data/split.json and data/synthetic/):
    good (test)       held-out good placements, never seen in training or calibration
    good (calib)      the calibration photos (they SET the hold line, so not independent; shown for reference)
    real defect       data/defect + data/capture/defect photos kept for testing (none in the first photo set)
    SYNTHETIC defect  fake damage drawn on the held-out good photos (nazar_vision.synth)
    no part           frames with the washer painted out, a grey frame, a zoomed belt
    unsure            photos a human could not label (reported only)
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import DATA, MODELS, ROOT, VERSION, images, locate
from .inspector import Inspector
from .split import load_split

VERDICTS = ["PASS", "HOLD", "REJECT", "NO_PART"]
DOCS = ROOT / "docs"


def _font(size, bold=True):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def histogram(rows, hold, reject, path):
    """Stacked histogram of scores per set, with the hold and reject lines."""
    sets = [("good (calibration)", "#9aa3ab"), ("good (test)", "#2f8f5b"), ("real defect", "#7a1fa2"),
            ("SYNTHETIC defect", "#d0452f"), ("unsure", "#e0a800")]
    vals = [r["score"] for r in rows if r["verdict"] != "NO_PART"]
    lo, hi = 0.0, max(vals + [reject]) * 1.08
    W, H, L, R, T, B = 1400, 640, 110, 40, 120, 110
    nb = 48
    edges = np.linspace(lo, hi, nb + 1)
    counts = {name: np.histogram([r["score"] for r in rows if r["set"] == name and r["verdict"] != "NO_PART"], edges)[0]
              for name, _ in sets}
    top = max(1, int(sum(counts.values()).max()))
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    y = lambda c: H - B - c / top * (H - T - B)
    d.text((L, 22), "Anomaly score per image (higher = less like the good washers)", fill="#1d2124", font=_font(30))
    lx = L
    for name, colour in sets:
        if counts[name].sum():
            d.rectangle([lx, 72, lx + 22, 94], fill=colour)
            d.text((lx + 30, 70), f"{name} ({counts[name].sum()})", fill="#1d2124", font=_font(22))
            lx += 50 + d.textlength(f"{name} ({counts[name].sum()})", font=_font(22))
    for c in range(0, top + 1, max(1, top // 5)):
        d.line([L, y(c), W - R, y(c)], fill="#e3e5e1", width=1)
        d.text((L - 14, y(c)), str(c), fill="#4a5056", font=_font(22), anchor="rm")
    for i in range(nb):
        base = 0
        for name, colour in sets:
            c = int(counts[name][i])
            if c:
                d.rectangle([x(edges[i]) + 1, y(base + c), x(edges[i + 1]) - 1, y(base)], fill=colour)
                base += c
    d.line([L, H - B, W - R, H - B], fill="#1d2124", width=2)
    for v in np.linspace(lo, hi, 7):
        d.text((x(v), H - B + 10), f"{v:.2f}", fill="#4a5056", font=_font(22), anchor="mt")
    d.text(((L + W - R) / 2, H - 40), "score", fill="#1d2124", font=_font(24), anchor="mm")
    d.text((L - 14, T - 22), "images", fill="#1d2124", font=_font(22), anchor="rs")
    for v, label, colour in ((hold, "HOLD", "#c98a00"), (reject, "REJECT", "#b3261e")):
        for yy in range(T - 10, H - B, 14):
            d.line([x(v), yy, x(v), yy + 8], fill=colour, width=3)
        d.text((x(v) + 6, T - 12), f"{label} {v:.3f}", fill=colour, font=_font(22))
    path.parent.mkdir(exist_ok=True)
    im.save(path)


def confusion(rows, name):
    sub = [r for r in rows if r["set"] == name]
    return {v: sum(r["verdict"] == v for r in sub) for v in VERDICTS}, len(sub)


def stage_timing(insp, paths, repeat=3):
    """Median time of each step on these images (ms)."""
    from .images import png_b64
    from .inspector import heat_overlay
    t = {"decode": [], "find washer": [], "crop": [], "backbone": [], "nearest neighbour": [], "map + PNGs": []}
    for _ in range(repeat):
        for p in paths:
            data = Path(p).read_bytes()
            t0 = time.perf_counter(); img = images.decode(data)
            t1 = time.perf_counter(); w = insp.find(img)
            t2 = time.perf_counter(); crop = locate.crop(img, w)
            t3 = time.perf_counter(); f = insp.backbone.embed([crop])[0]
            t4 = time.perf_counter(); dist = insp.memory.nearest(f)
            t5 = time.perf_counter()
            from .inspector import anomaly_map
            amap = anomaly_map(dist, insp.cfg["sigma"]) * insp.mask
            png_b64(crop); png_b64(heat_overlay(crop, amap, insp.mask, 0.6 * insp.hold, insp.reject))
            t6 = time.perf_counter()
            for k, a, b in zip(t, (t0, t1, t2, t3, t4, t5), (t1, t2, t3, t4, t5, t6)):
                t[k].append((b - a) * 1000)
    return {k: round(float(np.median(v)), 1) for k, v in t.items()}


def main():
    split = load_split()
    insp = Inspector.load()
    hold, reject = insp.hold, insp.reject
    todo = [("good (test)", rel) for rel in split["test_good"]]
    todo += [("good (calibration)", rel) for rel in split["calib"]]
    todo += [("real defect", rel) for rels in split["test_defect"].values() for rel in rels]
    for kind_dir in sorted((DATA / "synthetic").glob("*")):
        name = "no part" if kind_dir.name == "nopart" else "SYNTHETIC defect"
        todo += [(name, str(p.relative_to(ROOT))) for p in images.list_images(kind_dir)]
    todo += [("unsure", rel) for rel in split["unsure"]]

    train_centres = []
    for rel in split["train"]:
        w = insp.find(images.load(ROOT / rel))
        if w is not None:
            train_centres.append((w.cx, w.cy))
    train_centres = np.array(train_centres)
    rows, times = [], []
    for name, rel in todo:
        data = (ROOT / rel).read_bytes()
        t0 = time.perf_counter()
        res = insp.inspect(images.decode(data))
        times.append((time.perf_counter() - t0) * 1000)
        kind = Path(rel).parent.name if name in ("SYNTHETIC defect", "real defect") else ""
        row = {"set": name, "file": rel, "kind": kind, "verdict": res["verdict"], "score": res["score"],
               "ms": round(times[-1], 1)}
        if name.startswith("good") and res.get("washer"):
            c = np.array([res["washer"]["cx"], res["washer"]["cy"]])
            row["px_to_train"] = round(float(np.min(np.linalg.norm(train_centres - c, axis=1))), 1)
        rows.append(row)

    good_paths = [ROOT / rel for rel in split["test_good"] + split["calib"]]
    stages = stage_timing(insp, good_paths)
    lat = {"median": round(float(np.median(times)), 1), "p95": round(float(np.percentile(times, 95)), 1),
           "max": round(float(np.max(times)), 1), "n": len(times)}
    histogram(rows, hold, reject, DOCS / "score_histogram.png")
    out = {"version": VERSION, "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"), "hold": hold, "reject": reject,
           "latency_ms": lat, "stages_ms": stages, "rows": rows,
           "confusion": {n: confusion(rows, n)[0] for n in dict.fromkeys(r["set"] for r in rows)}}
    (MODELS / "eval.json").write_text(json.dumps(out, indent=1))
    write_report(out, split, insp.cfg)
    for n, c in out["confusion"].items():
        print(f"  {n:20s} " + "  ".join(f"{v} {c[v]}" for v in VERDICTS))
    print(f"latency per inspection (decode excluded): median {lat['median']} ms, p95 {lat['p95']} ms over {lat['n']}")
    print("stages (ms):", stages)
    print("wrote models/eval.json, docs/score_histogram.png, REPORT.md")


def write_report(ev, split, cfg):
    rows = ev["rows"]
    hold, reject = ev["hold"], ev["reject"]
    real_defects = sum(len(v) for v in split["test_defect"].values())
    order = ["good (test)", "good (calibration)", "real defect", "SYNTHETIC defect", "no part", "unsure"]
    present = [n for n in order if n in ev["confusion"]]
    L = []
    a = L.append
    a("# NAZAR vision: evaluation report")
    a("")
    a(f"*Generated by `python -m nazar_vision.evaluate` on {ev['evaluated'].replace('T', ' ')} "
      f"(model trained {cfg['trained'].replace('T', ' ')}). Rerun `./retrain.sh` after recapturing and this file is rewritten.*")
    a("")
    a("## In short")
    a("")
    g, gn = confusion(rows, "good (test)")
    s, sn = confusion(rows, "SYNTHETIC defect")
    npc, npn = confusion(rows, "no part")
    a(f"- **Held-out good washers:** {g['PASS']} of {gn} PASS, {g['HOLD']} HOLD, {g['REJECT']} REJECT. "
      f"Only {gn} independent good test photos exist, so this says little yet.")
    a(f"- **SYNTHETIC defects (drawn on, not real):** {s['REJECT']} of {sn} REJECT, {s['HOLD']} HOLD, {s['PASS']} PASS. "
      "This shows the pipeline reacts to a changed outline or face; it is not an accuracy figure for real damage.")
    if real_defects:
        rd, rdn = confusion(rows, "real defect")
        a(f"- **Real defects:** {rd['REJECT']} of {rdn} REJECT, {rd['HOLD']} HOLD, {rd['PASS']} PASS.")
    else:
        a("- **Real defects: none tested.** No photo of a real damaged washer exists yet.")
    a(f"- **No washer in frame:** {npc['NO_PART']} of {npn} NO_PART (never a defect).")
    a(f"- **Speed:** median {ev['latency_ms']['median']} ms, p95 {ev['latency_ms']['p95']} ms per inspection "
      "(after JPEG decode) on the Mac mini M4 CPU. Target < 150 ms.")
    a("")
    a("## Data and split")
    a("")
    a(f"Split by **placement** (a washer lying in one spot, photographed several times), not by photo, so a test photo is never "
      f"a near-copy of a training photo. Byte-identical files are dropped. Placements: {split['groups']['train']} train, "
      f"{split['groups']['calib']} calibration, {split['groups']['test_good']} test.")
    a("")
    a("| Set | Photos | Used for |")
    a("|---|---|---|")
    a(f"| Train | {len(split['train'])} | memory bank (x {cfg['rotations']} rotations, brightness +-10 %) |")
    a(f"| Calibration | {len(split['calib'])} | HOLD line |")
    a(f"| Test, good | {len(split['test_good'])} | confusion matrix |")
    a(f"| Real defects for REJECT line | {sum(len(v) for v in split['defect'].values())} | REJECT line |")
    a(f"| Test, real defects | {real_defects} | confusion matrix |")
    a(f"| Unsure | {len(split['unsure'])} | reported only |")
    a("")
    a("## Model and thresholds")
    a("")
    a(f"- WideResNet50-2 (ImageNet) layer2 + layer3 patch features, 3x3 averaged, projected to {cfg['dim']}-d, 28 x 28 grid.")
    a(f"- Memory bank: {cfg['patches_before_coreset']:,} patches from {cfg['n_train']} photos -> greedy k-center coreset "
      f"of **{cfg['bank']:,}** patches.")
    a(f"- Only the ring is scored: mask to {cfg['mask_outer']} x rim radius, hole middle ignored up to "
      f"{cfg['mask_inner']} x hole radius. Image score = max of the Gaussian-blurred (sigma {cfg['sigma']:g}) map.")
    a(f"- **HOLD = {hold:.4f}**: {cfg['hold_rule']}.")
    a(f"- **REJECT = {reject:.4f}**: {cfg['reject_rule']}.")
    cs = sorted(v for v in cfg["calib_scores"].values() if v is not None)
    a(f"- Calibration scores: {', '.join(f'{v:.3f}' for v in cs)}.")
    a("")
    a("## Confusion matrix")
    a("")
    a("Rows: what the image is. Columns: the verdict.")
    a("")
    a("| | PASS | HOLD | REJECT | NO_PART | n |")
    a("|---|---|---|---|---|---|")
    for n in present:
        c, k = confusion(rows, n)
        label = {"good (calibration)": "good (calibration, not independent)",
                 "SYNTHETIC defect": "**SYNTHETIC** defect (drawn on)"}.get(n, n)
        a(f"| {label} | {c['PASS']} | {c['HOLD']} | {c['REJECT']} | {c['NO_PART']} | {k} |")
    a("")
    a("![Score histogram](docs/score_histogram.png)")
    a("")
    a("## Every image")
    a("")
    a("*Distance to train* = how far (pixels) the washer centre is from the nearest training placement. A good test photo "
      "lying where a training washer lay is an easier test than one somewhere new on the belt.")
    a("")
    a("| Set | Image | Verdict | Score | Distance to train |")
    a("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (order.index(r["set"]), r["kind"], r["file"])):
        sc = "-" if r["verdict"] == "NO_PART" else f"{r['score']:.3f}"
        dist = f"{r['px_to_train']:.0f} px" if "px_to_train" in r else ""
        a(f"| {r['set']}{' / ' + r['kind'] if r['kind'] else ''} | {Path(r['file']).name} | {r['verdict']} | {sc} | {dist} |")
    a("")
    a("## Speed (CPU only)")
    a("")
    lat = ev["latency_ms"]
    a(f"Full inspection after decoding (find washer, crop, backbone, nearest-neighbour search, heatmap, two PNGs): "
      f"**median {lat['median']} ms, p95 {lat['p95']} ms**, max {lat['max']} ms over {lat['n']} images.")
    a("")
    a("| Step | Median ms |")
    a("|---|---|")
    for k, v in ev["stages_ms"].items():
        a(f"| {k} | {v} |")
    a("")
    a("## What is NOT verified")
    a("")
    a("- **Real defects.** " + ("None photographed yet; every defect number above is synthetic." if not real_defects else
                                 f"Only {real_defects} real defect photos; small sample."))
    a("- **The first photo set is weak:** the washer face is overexposed (pure white), so real scratches or rust may be invisible "
      "to the camera; most placements sit at the same spot on the belt; many files were byte-identical copies.")
    a(f"- **Few independent good tests** ({len(split['test_good'])}). The false-HOLD rate cannot be estimated from that.")
    tg = [r for r in rows if r["set"] == "good (test)" and "px_to_train" in r]
    cg = [r for r in rows if r["set"] == "good (calibration)" and "px_to_train" in r]
    if tg and cg and np.median([r["score"] for r in tg]) < np.median([r["score"] for r in cg]):
        a(f"- **The good test is easier than it looks:** the test washers lie a median "
          f"{np.median([r['px_to_train'] for r in tg]):.0f} px from a training placement and score lower than the "
          f"calibration washers ({np.median([r['px_to_train'] for r in cg]):.0f} px away). A washer somewhere new on the belt "
          "scores higher, closer to HOLD.")
    a("- **The real camera at belt speed** (motion blur, exposure drift) was not tested here.")
    a("")
    a("## Fix on site")
    a("")
    a("1. Lower exposure or light until the washer face is grey, not white.")
    a("2. Capture 40+ good washers through the console (each a different washer or a new spot), and every real damaged washer you have, "
      "with its kind (`/capture?label=defect&kind=scratch`).")
    a("3. `./retrain.sh` (about a minute), then read this report again.")
    (ROOT / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
