"""Find the washer in a frame, cut a square crop around it, and build the ring mask.

Steps:
  1. Work on a copy scaled to <= 640 px. Candidate centres come from HoughCircles; if Hough finds nothing,
     from the largest round blob (Otsu threshold, both polarities).
  2. Hough keeps one circle per centre and often picks the HOLE, not the rim. So for each candidate we read a
     radial brightness profile (median over 96 rays) and look for the outermost edge that goes all the way
     round: that is the outer rim. The hole is the strongest all-round edge inside 0.75 x rim.
  3. A washer is only accepted if the rim edge is strong and the ring is clearly brighter or darker than
     the belt just outside it. Otherwise: no part (NO_PART), never a defect.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import CROP

MARGIN = 0.08            # space kept around the outer rim, as a fraction of the radius
WORK = 640               # working resolution for the search
N_RAYS = 96


@dataclass
class Washer:
    cx: float
    cy: float
    r: float                # outer rim radius, pixels of the original frame
    hole: float | None      # hole radius, pixels (None if no clear hole)
    contrast: float         # ring vs belt brightness difference (grey levels)

    def box(self, shape):
        h, w = shape[:2]
        x0, y0 = max(0, int(self.cx - self.r)), max(0, int(self.cy - self.r))
        x1, y1 = min(w, int(np.ceil(self.cx + self.r))), min(h, int(np.ceil(self.cy + self.r)))
        return [x0, y0, x1 - x0, y1 - y0]


def _rays(gray, cx, cy, radii):
    """Brightness at each radius (rows) along N_RAYS directions (columns), bilinear."""
    a = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    xs = (cx + radii[:, None] * np.cos(a)[None]).astype(np.float32)
    ys = (cy + radii[:, None] * np.sin(a)[None]).astype(np.float32)
    return cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _measure(gray, cx, cy, r_lo, r_hi):
    """Outer rim + hole for a candidate centre, or None."""
    radii = np.arange(1.0, r_hi * 1.15, 0.5, dtype=np.float32)
    prof = _rays(gray, cx, cy, radii)                       # (R, rays)
    grad = np.abs(np.diff(prof, axis=0))                     # edge strength per ray
    allround = np.percentile(grad, 25, axis=1)               # high only if the edge exists on most rays
    allround = np.convolve(allround, np.ones(3) / 3, mode="same")
    rr = radii[:-1] + 0.25
    ok = (rr >= r_lo) & (rr <= r_hi)
    if not ok.any():
        return None
    peak = allround[ok].max()
    if peak < 4.0:                                           # no edge that goes all round
        return None
    # outermost strong all-round edge = outer rim
    strong = np.where(ok & (allround > 0.5 * peak))[0]
    i = strong[-1]
    while i + 1 < len(allround) and allround[i + 1] >= allround[i]:
        i += 1
    r = float(rr[i])
    # contrast: ring band vs belt band (median over rays, so a notch or a neighbour object does not count)
    band = lambda f0, f1: float(np.median(prof[(radii >= f0 * r) & (radii <= f1 * r)]))
    ring, belt = band(0.80, 0.92), band(1.10, 1.15)
    contrast = ring - belt
    # hole: strongest all-round edge between 0.12 r and 0.75 r
    inner = (rr > 0.12 * r) & (rr < 0.75 * r)
    hole = None
    if inner.any():
        j = np.where(inner)[0][np.argmax(allround[inner])]
        if allround[j] > 0.35 * allround[i]:
            hole = float(rr[j])
    return r, hole, contrast


def _fit_circle(x, y):
    """Least-squares circle through points (Kasa fit) -> cx, cy, r."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (p, q, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = p / 2, q / 2
    return cx, cy, float(np.sqrt(max(c + cx * cx + cy * cy, 0)))


def _refine(gray, cx, cy, lo, hi, rounds=3):
    """Move a rough centre onto the true one: on every ray take the outermost strong edge, fit a circle,
    drop rays that disagree (a notch, the rail), refit."""
    a = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    radii = np.arange(max(2.0, lo * 0.5), hi * 1.15, 0.5, dtype=np.float32)
    for _ in range(rounds):
        grad = np.abs(np.diff(_rays(gray, cx, cy, radii), axis=0))     # (R, rays)
        top = grad.max(0)
        if np.median(top) < 6:
            return None
        edge = np.array([radii[np.where(grad[:, k] > 0.5 * top[k])[0][-1]] for k in range(N_RAYS)]) + 0.25
        x, y = cx + edge * np.cos(a), cy + edge * np.sin(a)
        keep = np.abs(edge - np.median(edge)) < 0.15 * np.median(edge)    # robust start: ignore far outliers
        if keep.sum() < 8:
            return None
        for _ in range(3):
            fx, fy, fr = _fit_circle(x[keep], y[keep])
            res = np.abs(np.hypot(x - fx, y - fy) - fr)
            keep = res < max(2.0, 2.5 * np.median(res))
        if keep.mean() < 0.6:
            return None
        moved = np.hypot(fx - cx, fy - cy)
        cx, cy = fx, fy
        if moved < 0.3:
            break
    return float(cx), float(cy)


def _hough_centres(blur, lo, hi):
    c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, lo), param1=100, param2=30,
                         minRadius=max(4, lo // 3), maxRadius=hi)
    return [] if c is None else [(float(x), float(y)) for x, y, _ in c[0][:5]]


def _blob_centres(blur, lo, hi):
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = []
    for m in (th, 255 - th):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y), r = cv2.minEnclosingCircle(c)
            if lo <= r <= hi and cv2.contourArea(c) > 0.5 * np.pi * r * r:
                out.append((r, float(x), float(y)))
    return [(x, y) for _, x, y in sorted(out, reverse=True)[:3]]


def find(img, radius_px=None, min_contrast=30.0):
    """Washer in a BGR frame or None. radius_px = (lo, hi) expected outer radius if known from training."""
    h, w = img.shape[:2]
    s = min(1.0, WORK / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    side = min(gray.shape)
    lo, hi = int(side * 0.06), int(side * 0.48)
    if radius_px:
        lo, hi = max(lo, int(radius_px[0] * s)), min(hi, int(radius_px[1] * s) + 1)
        if lo >= hi:
            return None
    gf = blur.astype(np.float32)
    best = None
    seen = []
    for cx, cy in _hough_centres(blur, lo, hi) + _blob_centres(blur, lo, hi):
        c = _refine(gf, cx, cy, lo, hi)
        if c is None or any(np.hypot(c[0] - u, c[1] - v) < 3 for u, v in seen):
            continue
        seen.append(c)
        cx, cy = c
        m = _measure(gf, cx, cy, lo, hi)
        if m is None:
            continue
        r, hole, contrast = m
        if abs(contrast) < min_contrast:
            continue
        if cx - r < 1 or cy - r < 1 or cx + r > gray.shape[1] - 1 or cy + r > gray.shape[0] - 1:
            continue                                            # washer cut by the frame edge
        if best is None or abs(contrast) > abs(best[4]):
            best = (cx, cy, r, hole, contrast)
    if best is None:
        return None
    cx, cy, r, hole, contrast = best
    return Washer(cx / s, cy / s, r / s, hole / s if hole else None, contrast)


def crop(img, washer, angle=0.0, gain=1.0):
    """CROP x CROP RGB crop centred on the washer, rim + MARGIN filling it, optionally rotated / brightened.
    Sized straight to 224: a resize-256 + centre-crop would cut the rim off."""
    half = washer.r * (1 + MARGIN)
    scale = CROP / (2 * half)
    m = cv2.getRotationMatrix2D((washer.cx, washer.cy), angle, scale)
    m[:, 2] += (CROP / 2 - washer.cx, CROP / 2 - washer.cy)
    out = cv2.warpAffine(img, m, (CROP, CROP), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if gain != 1.0:
        rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return rgb


def ring_mask(hole_frac, size=CROP, outer=1.04, inner=0.7, soft=3.0):
    """Float mask on the crop: 1 on the washer ring, 0 outside the rim and in the middle of the hole.
    outer is in rim radii; inner is a fraction of the hole radius (so the hole EDGE is still scored)."""
    r = size / 2 / (1 + MARGIN)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    d = np.hypot(xx - size / 2, yy - size / 2)
    m = np.clip((outer * r - d) / soft + 0.5, 0, 1)
    if hole_frac:
        m *= np.clip((d - inner * hole_frac * r) / soft + 0.5, 0, 1)
    return m.astype(np.float32)
