"""One inspection: frame -> washer -> crop -> patch features -> distance to good memory -> verdict + heatmap."""
import json
import time

import cv2
import numpy as np

from . import CROP, GRID, MODEL_NAME, MODELS, locate
from .features import Backbone
from .images import png_b64
from .memory import Memory


def anomaly_map(dist, sigma):
    """Per-patch distances (GRID*GRID,) -> smooth CROP x CROP map."""
    m = cv2.resize(dist.reshape(GRID, GRID).astype(np.float32), (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), sigma)


def heat_overlay(crop, amap, mask, lo, hi):
    """Colour heatmap over the crop: transparent below lo, full colour at hi (the reject line)."""
    t = np.clip((amap - lo) / max(hi - lo, 1e-6), 0, 1)
    colour = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].astype(np.float32)
    alpha = (0.75 * t)[..., None]                       # fades in smoothly, no hard edge
    base = crop.astype(np.float32)
    dim = 0.45 + 0.55 * mask[..., None]                    # grey out what is not scored
    out = base * dim * (1 - alpha) + colour * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def thumbnail(img, width=320):
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class Inspector:
    def __init__(self, cfg, backbone, bank):
        self.cfg = cfg
        self.backbone = backbone
        self.memory = Memory(bank)
        self.mask = locate.ring_mask(cfg["hole_frac"], outer=cfg["mask_outer"], inner=cfg["mask_inner"])
        self.score_crop(np.full((CROP, CROP, 3), 128, np.uint8))    # warm-up run

    @classmethod
    def load(cls, models=MODELS):
        missing = [n for n in ("backbone.onnx", "memory_bank.npy", "config.json") if not (models / n).is_file()]
        if missing:
            raise FileNotFoundError(f"models/ is missing {', '.join(missing)}: run ./retrain.sh "
                                    "(backbone.onnx comes from the release or nazar_vision.export)")
        cfg = json.loads((models / "config.json").read_text())
        return cls(cfg, Backbone(models / "backbone.onnx"), np.load(models / "memory_bank.npy"))

    hold = property(lambda self: self.cfg["hold"])
    reject = property(lambda self: self.cfg["reject"])

    def find(self, img):
        """Washer, using the radius range and brightness polarity learned from the training photos."""
        side = min(img.shape[:2])
        f = self.cfg.get("radius_frac")
        lo, hi = self.cfg.get("radius_range", [0.6, 1.6])
        rng = (f * side * lo, f * side * hi) if f else None
        w = locate.find(img, rng, min_contrast=self.cfg.get("min_contrast", 30.0))
        pol = self.cfg.get("polarity", 0)
        if w is not None and pol and np.sign(w.contrast) != pol:
            return None                     # a dark round shadow when washers are bright, etc.
        return w

    def score_crop(self, crop):
        feats = self.backbone.embed([crop])[0]
        dist = self.memory.nearest(feats)
        amap = anomaly_map(dist, self.cfg["sigma"]) * self.mask
        return float(amap.max()), amap

    def inspect(self, img, hold=None, reject=None):
        t0 = time.perf_counter()
        hold = self.hold if hold is None else float(hold)
        reject = self.reject if reject is None else float(reject)
        out = {"ok": True, "verdict": "NO_PART", "score": 0.0, "hold": round(hold, 4), "reject": round(reject, 4)}
        w = self.find(img)
        if w is None:
            out.update(crop=png_b64(thumbnail(img)), heatmap="", box=None)
        else:
            crop = locate.crop(img, w)
            score, amap = self.score_crop(crop)
            out["verdict"] = "PASS" if score <= hold else "HOLD" if score < reject else "REJECT"
            out["score"] = round(score, 4)
            out.update(crop=png_b64(crop), heatmap=png_b64(heat_overlay(crop, amap, self.mask, 0.6 * hold, reject)),
                       box=w.box(img.shape))
            f = self.cfg.get("radius_frac")
            out["washer"] = {"cx": round(w.cx, 1), "cy": round(w.cy, 1), "r": round(w.r, 1),
                             "hole": round(w.hole, 1) if w.hole else None,
                             "size_ratio": round(w.r / (f * min(img.shape[:2])), 3) if f else None}
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    def health(self):
        return {"ok": True, "model": self.cfg.get("model", MODEL_NAME), "hold": round(self.hold, 4),
                "reject": round(self.reject, 4), "bank": len(self.memory)}
