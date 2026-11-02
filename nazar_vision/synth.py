"""SYNTHETIC test images, for TESTING ONLY: fake damage drawn onto held-out good photos, and frames with
the washer removed (should give NO_PART). Never used for training or for setting thresholds.

    python -m nazar_vision.synth        # reads data/split.json, writes data/synthetic/<kind>/SYNTH-*.jpg

Kinds: notch (bite out of the rim), burr (lump of metal in the hole), crack (dark line across the ring),
spot (brown rust spot), scratch (thin grey line on the face), nopart (washer painted out with belt texture).
"""
import shutil

import cv2
import numpy as np

from . import DATA, ROOT, images, locate
from .split import load_split

OUT = DATA / "synthetic"
KINDS = ("notch", "burr", "crack", "spot", "scratch")


def _belt_colour(img, w):
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    xs = np.clip((w.cx + 1.25 * w.r * np.cos(a)).astype(int), 0, img.shape[1] - 1)
    ys = np.clip((w.cy + 1.25 * w.r * np.sin(a)).astype(int), 0, img.shape[0] - 1)
    return tuple(int(v) for v in np.median(img[ys, xs], axis=0))


def draw(img, w, kind, rng):
    out = img.copy()
    hole = w.hole or 0.35 * w.r
    t = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(t), np.sin(t)])
    c = np.array([w.cx, w.cy])
    pt = lambda rad, ang=0.0: tuple(np.round(c + rad * np.array([np.cos(t + ang), np.sin(t + ang)])).astype(int))
    if kind == "notch":
        r = rng.uniform(0.10, 0.16) * w.r
        cv2.circle(out, pt(w.r), int(r), _belt_colour(img, w), -1, cv2.LINE_AA)
    elif kind == "burr":
        r = rng.uniform(0.07, 0.11) * w.r
        cv2.circle(out, pt(hole * 0.95), int(r), (238, 240, 242), -1, cv2.LINE_AA)
    elif kind == "crack":
        cv2.line(out, pt(hole * 1.05), pt(w.r * 0.97, rng.uniform(-0.08, 0.08)), (60, 60, 64),
                 max(2, int(0.025 * w.r)), cv2.LINE_AA)
    elif kind == "spot":
        r = rng.uniform(0.07, 0.10) * w.r
        mid = (hole + w.r) / 2
        cv2.circle(out, pt(mid), int(r), (40, 85, 150), -1, cv2.LINE_AA)        # BGR rust brown
        out = cv2.addWeighted(out, 0.85, cv2.GaussianBlur(out, (0, 0), 2), 0.15, 0)
    elif kind == "scratch":
        a = pt(hole * 1.3, -0.5)
        b = pt(w.r * 0.85, 0.35)
        cv2.line(out, a, b, (150, 150, 155), max(1, int(0.012 * w.r)), cv2.LINE_AA)
    return out


def remove_washer(img, w):
    """Inpaint the washer away so only belt is left."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (int(w.cx), int(w.cy)), int(w.r * 1.12), 255, -1)
    return cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)


def main():
    split = load_split()
    shutil.rmtree(OUT, ignore_errors=True)
    for k in KINDS + ("nopart",):
        (OUT / k).mkdir(parents=True)
    n = 0
    for i, rel in enumerate(split["test_good"]):
        p = ROOT / rel
        img = images.load(p)
        w = locate.find(img)
        if w is None:
            print(f"  {rel}: no washer found, skipped")
            continue
        for j, kind in enumerate(KINDS):
            rng = np.random.default_rng(1000 * i + j)
            cv2.imwrite(str(OUT / kind / f"SYNTH-{kind}-{p.stem}.jpg"), draw(img, w, kind, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        cv2.imwrite(str(OUT / "nopart" / f"SYNTH-nopart-{p.stem}.jpg"), remove_washer(img, w), [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, wd = img.shape[:2]
    cv2.imwrite(str(OUT / "nopart" / "SYNTH-nopart-grey.jpg"), np.full((h, wd, 3), 120, np.uint8))
    belt = cv2.resize(img[: h // 3, wd // 6: wd * 5 // 6], (wd, h))
    cv2.imwrite(str(OUT / "nopart" / "SYNTH-nopart-belt-zoom.jpg"), belt)
    print(f"synthetic: {n} defect images ({len(KINDS)} kinds) + {len(images.list_images(OUT / 'nopart'))} no-part frames in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
