"""Turn iPhone photos and videos of washers into training images: one washer per image, sharp, no near-copies.

    python -m nazar_vision.ingest                                   # everything under footage/ (see below)
    python -m nazar_vision.ingest ~/Downloads/washers --label good  # a flat folder, e.g. straight from AirDrop
    python -m nazar_vision.ingest ~/Downloads/bad --label defect --kind scratch
    python -m nazar_vision.ingest --reset                           # forget everything ingested so far

footage/ layout (any of these may be missing):
    footage/good/*                good washers
    footage/defect/<kind>/*       damaged washers; kind = scratch, dent, rust, burr, crack, chipped, wrong-size, other
    footage/empty/*               belt with no washer (tests NO_PART)
Accepted files: photos .heic .heif .jpg .jpeg .png, videos .mov .mp4 .m4v.

PHOTOS (the main path)
  - HEIC is converted with macOS `sips` to JPEG (the primary image, i.e. the SDR base of an HDR photo; the gain map is
    not applied), then EXIF orientation is applied with Pillow. The capture time comes from EXIF DateTimeOriginal +
    SubsecTimeOriginal (file modification time if missing).
  - Each photo is its own placement group, except a burst: photos taken < 2 s apart with the washer centre within
    6 % of its radius are the same placement (one group; near-identical copies inside it are dropped).
VIDEOS
  - ffmpeg extracts ~4 frames/s. ffmpeg applies the rotation metadata itself. HDR (HLG / PQ, e.g. iPhone Dolby
    Vision) is tone-mapped to SDR BT.709 so frames are not washed out. Each clip is one placement group.
  - Inside a clip a frame is dropped if the washer barely moved (< 6 % of its radius) AND looks the same as a frame
    already kept, or if it is blurrier than half the clip's median. At most 12 frames per clip, spread over time.
EVERY IMAGE
  - Scaled so the long side is <= 1920 px (the live camera is 1920 x 1080). The inspection itself is resolution
    independent: it crops around the washer and resizes the crop to 224 x 224.
  - Kept only if exactly one washer is found, fully inside the frame with a margin of 10 % of its radius, and the
    washer crop is sharp enough (variance of the Laplacian on the 224 x 224 crop >= --min-sharp, default 15).
  - A washer whose size (radius / short side) is more than 25 % off the median good washer is dropped as a
    mis-detection, except in defect/wrong-size.
  - The share of washer pixels that are pure white (>= 250) is reported: above ~20 % the photo is overexposed.

Output (the layout nazar_vision.split reads):
    data/footage/good/<group>/<image>.jpg, data/footage/defect/<kind>/<group>/<image>.jpg, data/footage/empty/<group>/...
    data/footage/manifest.csv    one row per photo / video frame: source, time, label, kind, group, sharpness, kept, reason
Work files: data/footage/_records/<source>.json (analysis, so a rerun skips finished sources) and
data/footage/_work/ (the converted images). Outputs are hard links to the work files, so they cost no extra disk.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import DATA, ROOT, locate

FOOTAGE = ROOT / "footage"
OUT = DATA / "footage"
RECORDS = OUT / "_records"
WORK = OUT / "_work"
LABEL_DIRS = ("good", "defect", "empty")
KINDS = ("scratch", "dent", "rust", "burr", "crack", "chipped", "wrong-size", "other")
PHOTO_EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mov", ".mp4", ".m4v"}

DEFAULTS = dict(fps=4.0, max_side=1920, cap=12, min_sharp=15.0, rel_sharp=0.5, dup_move=0.06, dup_look=6.0,
                burst_s=2.0, edge_pad=0.10, radius_tol=0.25, empty_cap=6)


# ------------------------------------------------------------------ tools
def tool(name):
    for cand in (shutil.which(name), str(Path.home() / "Library/Python/3.14/bin" / name), f"/opt/homebrew/bin/{name}"):
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit(f"{name} not found (needed for videos). Put it on PATH or in ~/Library/Python/3.14/bin/")


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "x"


def source_id(path):
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:10]
    return f"{slug(path.stem)}-{h}"


def fit(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


# ------------------------------------------------------------------ photos
def _exif_time(exif):
    sub = exif.get_ifd(0x8769)
    raw = sub.get(0x9003) or exif.get(0x0132)
    if not raw:
        return None
    try:
        t = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None
    frac = str(sub.get(0x9291) or "").strip("\x00 ")
    return t + (float("0." + frac) if frac.isdigit() else 0.0)


def load_photo(path, max_side):
    """-> (BGR image upright and scaled, capture time or None, note). HEIC goes through macOS sips."""
    from PIL import Image, ImageOps
    note = ""
    tmp = None
    try:
        try:
            im = Image.open(path)
            im.load()
        except Exception:
            if not shutil.which("sips"):
                raise ValueError("cannot read this photo (HEIC needs macOS sips)")
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
            r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "95", str(path), "--out", tmp],
                               capture_output=True, text=True)
            if r.returncode or not os.path.getsize(tmp):
                raise ValueError(f"sips could not convert it: {r.stderr.strip()[:120]}")
            im = Image.open(tmp)
            im.load()
            note = "sips"
        exif = im.getexif()
        orient = exif.get(0x0112, 1)
        t = _exif_time(exif)
        im = ImageOps.exif_transpose(im).convert("RGB")
        if orient not in (1, None):
            note = (note + f" orient{orient}").strip()
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        return fit(bgr, max_side), t, note
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ------------------------------------------------------------------ videos
def probe(path):
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                          "-of", "json", str(path)], capture_output=True, text=True)
    if out.returncode:
        raise ValueError(f"ffprobe failed: {out.stderr.strip()[:200]}")
    info = json.loads(out.stdout)
    if not info.get("streams"):
        raise ValueError("no video stream")
    s = info["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int(s.get("tags", {}).get("rotate", 0) or 0)
    return {"codec": s.get("codec_name"), "w": s.get("width"), "h": s.get("height"), "pix_fmt": s.get("pix_fmt"),
            "transfer": s.get("color_transfer"), "primaries": s.get("color_primaries"),
            "matrix": s.get("color_space"), "rotation": rot,
            "duration": float(info.get("format", {}).get("duration") or s.get("duration") or 0)}


def video_filter(p):
    """HDR -> SDR filter chain ('' for SDR). Frame sampling is done with the -r output option, not a filter:
    the ffmpeg that ships with Remotion (the one on this Mac) has no fps/select filters."""
    chain = []
    if p["transfer"] in ("arib-std-b67", "smpte2084"):          # HLG or PQ: HDR -> SDR BT.709
        tin = p["transfer"]
        pin = p["primaries"] if p["primaries"] in ("bt2020", "bt709") else "bt2020"
        mi = "bt2020nc" if p["matrix"] in (None, "bt2020nc", "unknown") else p["matrix"]
        # SDR white = 203 nits (ITU-R BT.2408 HLG reference white); mobius keeps everything below that almost
        # unchanged and rolls brighter highlights off softly (hable darkened SDR-range content by ~35 %, clip
        # loses highlight texture). Tested on a synthetic HLG encode: mean abs difference 13.6 grey levels.
        chain += [f"zscale=tin={tin}:pin={pin}:min={mi}:t=linear:npl=203", "format=gbrpf32le", "zscale=p=bt709",
                  "tonemap=tonemap=mobius:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p"]
    return ",".join(chain)


def extract_frames(path, p, fps, folder):
    vf = video_filter(p)
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path)] + (["-vf", vf] if vf else []) + \
          ["-r", str(fps), "-q:v", "2", str(folder / "f%05d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")
    return sorted(folder.glob("f*.jpg"))


# ------------------------------------------------------------------ analysis
def analyse(img, label, cfg):
    """Find the washer and measure it. Returns a dict; 'reason' is set when the image must be dropped."""
    h, w = img.shape[:2]
    a = {"w": w, "h": h, "reason": ""}
    wsh = locate.find(img, allow_edge=True)
    if label == "empty":
        if wsh is not None:
            a["reason"] = "washer found in an empty-belt image"
        return a
    if wsh is None:
        a["reason"] = "no washer found (or cut by the frame edge)"
        return a
    a.update(cx=round(wsh.cx, 1), cy=round(wsh.cy, 1), r=round(wsh.r, 1), ratio=round(wsh.r / min(h, w), 4),
             hole=round(wsh.hole, 1) if wsh.hole else None)
    if locate.touches_edge((wsh.cx, wsh.cy, wsh.r), img.shape, pad=cfg["edge_pad"] * wsh.r):
        a["reason"] = "washer cut by (or too close to) the frame edge"
        return a
    if locate.count(img, wsh) > 1:
        a["reason"] = "more than one washer"
        return a
    crop = locate.crop(img, wsh)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ring = locate.ring_mask((wsh.hole / wsh.r) if wsh.hole else 0.35) > 0.5
    a["sharp"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    a["clipped"] = round(float((gray[ring] >= 250).mean()), 3)
    a["look"] = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    if a["sharp"] < cfg["min_sharp"]:
        a["reason"] = f"too blurry (sharpness {a['sharp']:.0f} < {cfg['min_sharp']:.0f})"
    return a


def near_copy(a, b, cfg):
    """Same placement and same look: centre moved < dup_move x radius and crops differ < dup_look grey levels."""
    if "look" not in a or "look" not in b:
        return False
    moved = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / max(a["r"], 1)
    return moved < cfg["dup_move"] and float(np.mean(np.abs(a["look"] - b["look"]))) < cfg["dup_look"]


def _save_work(img, rel):
    path = WORK / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return rel


def _public(a):
    return {k: v for k, v in a.items() if k != "look"}


def ingest_photo(path, label, kind, cfg):
    sid = source_id(path)
    rec = {"id": sid, "source": str(path), "type": "photo", "label": label, "kind": kind, "folder": str(path.parent)}
    try:
        img, t, note = load_photo(path, cfg["max_side"])
    except Exception as e:
        rec["items"] = [{"t": 0, "reason": f"cannot read: {e}"}]
        return rec
    try:
        a = analyse(img, label, cfg)
    except Exception as e:                                   # never stop the whole run for one file
        a = {"reason": f"analysis failed: {type(e).__name__}: {e}"}
    a.update(t=0.0, abs_time=t if t is not None else path.stat().st_mtime, time_src="exif" if t else "file",
             note=note)
    if not a["reason"]:
        a["work"] = _save_work(img, f"{sid}.jpg")
        if "look" in a:
            np.save(WORK / f"{sid}.look.npy", a["look"])
    rec["items"] = [_public(a)]
    return rec


def ingest_video(path, label, kind, cfg):
    sid = source_id(path)
    rec = {"id": sid, "source": str(path), "type": "video", "label": label, "kind": kind, "folder": str(path.parent)}
    try:
        p = probe(path)
    except Exception as e:
        rec["items"] = [{"t": 0, "reason": f"cannot read: {e}"}]
        return rec
    rec["probe"] = p
    items = []
    with tempfile.TemporaryDirectory() as td:
        try:
            frames = extract_frames(path, p, cfg["fps"], Path(td))
        except Exception as e:
            rec["items"] = [{"t": 0, "reason": str(e)}]
            return rec
        imgs = {}
        for i, f in enumerate(frames):
            img = fit(cv2.imread(str(f), cv2.IMREAD_COLOR), cfg["max_side"])
            try:
                a = analyse(img, label, cfg)
            except Exception as e:
                a = {"reason": f"analysis failed: {type(e).__name__}: {e}"}
            a["t"] = round(i / cfg["fps"], 2)
            items.append(a)
            if not a["reason"]:
                imgs[i] = img
    rec["frame_size"] = [items[0]["w"], items[0]["h"]] if items else None
    ok = [i for i, a in enumerate(items) if not a["reason"]]
    if label != "empty" and ok:                              # relative blur: moving-hand frames
        med = float(np.median([items[i]["sharp"] for i in ok]))
        for i in ok:
            if items[i]["sharp"] < cfg["rel_sharp"] * med:
                items[i]["reason"] = f"blurrier than the clip (sharpness {items[i]['sharp']:.0f} < half of {med:.0f})"
    kept = []
    for i, a in enumerate(items):
        if a["reason"]:
            continue
        if any(near_copy(a, items[j], cfg) for j in kept):
            a["reason"] = "near-copy of a kept frame (washer not moved)"
            continue
        kept.append(i)
    cap = cfg["empty_cap"] if label == "empty" else cfg["cap"]
    if len(kept) > cap:                                      # spread the cap over the whole clip
        keep = set(kept[int(round(k))] for k in np.linspace(0, len(kept) - 1, cap))
        for i in kept:
            if i not in keep:
                items[i]["reason"] = f"over the {cap}-frame cap for one clip"
        kept = sorted(keep)
    for i in kept:
        items[i]["work"] = _save_work(imgs[i], f"{sid}/t{items[i]['t']:07.2f}.jpg")
    rec["items"] = [_public(a) for a in items]
    return rec


# ------------------------------------------------------------------ build the output tree from all records
def _groups(records, cfg):
    """Placement group per kept item. Videos: the clip. Photos: bursts (< burst_s apart, centre within dup_move*r)."""
    out = {}
    photos = {}
    for rec in records:
        if rec["type"] == "video":
            for a in rec["items"]:
                if a.get("work"):
                    out[(rec["id"], a["t"])] = f"clip-{rec['id']}"
        else:
            a = rec["items"][0]
            if a.get("work"):
                photos.setdefault((rec["label"], rec["kind"], rec["folder"]), []).append((a["abs_time"], rec))
    for key, lst in photos.items():
        lst.sort(key=lambda x: (x[0], x[1]["source"]))
        group, prev = None, None
        for t, rec in lst:
            a = rec["items"][0]
            same = (prev is not None and t - prev[0] < cfg["burst_s"] and "cx" in a and "cx" in prev[1]
                    and np.hypot(a["cx"] - prev[1]["cx"], a["cy"] - prev[1]["cy"]) / max(a["r"], 1) < cfg["dup_move"])
            if not same:
                group = f"photo-{rec['id']}"
            out[(rec["id"], 0.0)] = group
            prev = (t, a)
    return out


def build(cfg, log=print):
    records = [json.loads(p.read_text()) for p in sorted(RECORDS.glob("*.json"))]
    groups = _groups(records, cfg)
    # near-copies inside a photo burst
    seen = {}
    for rec in records:
        if rec["type"] != "photo":
            continue
        a = rec["items"][0]
        g = groups.get((rec["id"], 0.0))
        if not g:
            continue
        lk = WORK / f"{rec['id']}.look.npy"
        a["look"] = np.load(lk) if lk.is_file() else None
        if a["look"] is not None and any(near_copy(a, b, cfg) for b in seen.get(g, [])):
            a["reason"] = "near-copy of another photo of the same placement"
            a.pop("work", None)
        else:
            seen.setdefault(g, []).append(a)
    # size outliers against the median good washer
    good = [a["ratio"] for rec in records if rec["label"] == "good" for a in rec["items"] if a.get("work")]
    med = float(np.median(good)) if len(good) >= 5 else None
    if med:
        for rec in records:
            if rec["label"] == "empty" or rec["kind"] == "wrong-size":
                continue
            for a in rec["items"]:
                if a.get("work") and abs(a["ratio"] / med - 1) > cfg["radius_tol"]:
                    a["reason"] = f"washer size {a['ratio'] / med:.2f} x the median good washer (mis-detection?)"
                    a.pop("work")
    # output tree (hard links), manifest
    for d in LABEL_DIRS:
        shutil.rmtree(OUT / d, ignore_errors=True)
    rows = []
    for rec in records:
        for a in rec["items"]:
            g = groups.get((rec["id"], a.get("t", 0.0)), "")
            out = ""
            if a.get("work"):
                sub = Path(rec["label"]) / (rec["kind"] if rec["label"] == "defect" else "") / g
                dst = OUT / sub / (f"{rec['id']}.jpg" if rec["type"] == "photo" else f"{rec['id']}-t{a['t']:07.2f}.jpg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(WORK / a["work"], dst)
                except OSError:
                    shutil.copy2(WORK / a["work"], dst)
                out = str(dst.relative_to(ROOT))
            rows.append({"source": rec["source"], "type": rec["type"], "time_s": a.get("t", 0.0),
                         "label": rec["label"], "kind": rec["kind"] or "", "group": g if out else "",
                         "width": a.get("w", ""), "height": a.get("h", ""), "radius_ratio": a.get("ratio", ""),
                         "sharpness": a.get("sharp", ""), "clipped_white": a.get("clipped", ""),
                         "kept": "yes" if out else "no", "reason": a.get("reason", ""), "output": out})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["source"])
        wr.writeheader()
        wr.writerows(rows)
    summary(records, rows, med, log)
    return rows


def summary(records, rows, med, log):
    log("")
    log(f"{'source':34s} {'label':16s} {'kept':>9s}  why dropped")
    for rec in records:
        r = [x for x in rows if x["source"] == rec["source"]]
        k = sum(x["kept"] == "yes" for x in r)
        why = {}
        for x in r:
            if x["kept"] == "no":
                key = re.sub(r"\s*\(.*\)", "", x["reason"])
                why[key] = why.get(key, 0) + 1
        lab = rec["label"] + (f"/{rec['kind']}" if rec["kind"] else "")
        extra = ""
        if rec["type"] == "video" and rec.get("probe"):
            p = rec["probe"]
            extra = f"  [{p['codec']} {p['w']}x{p['h']} {p['pix_fmt']}{' HDR->SDR' if p['transfer'] in ('arib-std-b67', 'smpte2084') else ''}{' rot ' + str(p['rotation']) if p['rotation'] else ''}]"
        sharp = [x["sharpness"] for x in r if x["kept"] == "yes" and x["sharpness"] != ""]
        clip = [x["clipped_white"] for x in r if x["kept"] == "yes" and x["clipped_white"] != ""]
        meas = f"  sharp {np.median(sharp):.0f}" if sharp else ""
        meas += f"  white {100 * np.median(clip):.0f}%" if clip else ""
        log(f"{Path(rec['source']).name[:34]:34s} {lab:16s} {k:>4d}/{len(r):<4d}  "
            + (", ".join(f"{v} {kk}" for kk, v in why.items()) or "-") + meas + extra)
    kept = [x for x in rows if x["kept"] == "yes"]
    by = {}
    for x in kept:
        key = x["label"] + (f"/{x['kind']}" if x["kind"] else "")
        by.setdefault(key, set()).add(x["group"])
        by.setdefault(key + " images", set()).add(x["output"])
    log("")
    for key in sorted(k for k in by if not k.endswith(" images")):
        log(f"  {key:22s} {len(by[key + ' images']):4d} images in {len(by[key]):3d} placement groups")
    if med:
        log(f"  median good washer radius: {med:.4f} x short side of the image")
    clip = [x["clipped_white"] for x in kept if x["clipped_white"] != ""]
    if clip and np.median(clip) > 0.2:
        log(f"  WARNING: median {100 * np.median(clip):.0f} % of washer pixels are pure white: lower the exposure, "
            "texture (scratches, rust) is lost")
    g = len(by.get("good", ()))
    if 0 < g < 9:
        log(f"  WARNING: only {g} good placement groups; training needs at least 9 (aim for 30-40)")
    log(f"  manifest: {(OUT / 'manifest.csv').relative_to(ROOT)}")


# ------------------------------------------------------------------ main
def sources_in_footage():
    out = []
    for label in LABEL_DIRS:
        base = FOOTAGE / label
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in PHOTO_EXT | VIDEO_EXT or p.name.startswith("."):
                continue
            kind = ""
            if label == "defect":
                parts = p.relative_to(base).parts
                kind = parts[0].lower() if len(parts) > 1 else "other"
            out.append((p, label, kind))
    return out


def norm_kind(kind, label, log):
    if label != "defect":
        return ""
    k = (kind or "other").lower().replace("_", "-").replace(" ", "-")
    if k not in KINDS:
        log(f"  kind '{kind}' is not one of {', '.join(KINDS)}: using 'other'")
        k = "other"
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="a flat folder of photos/videos (default: the footage/ tree)")
    ap.add_argument("--label", choices=LABEL_DIRS, help="label for every file in FOLDER")
    ap.add_argument("--kind", default="", help=f"defect kind for FOLDER: {', '.join(KINDS)}")
    ap.add_argument("--reset", action="store_true", help="forget everything ingested so far (data/footage/)")
    ap.add_argument("--redo", action="store_true", help="re-analyse sources already ingested")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    if args.reset:
        shutil.rmtree(OUT, ignore_errors=True)
        print("data/footage/ removed")
        if not args.folder and not FOOTAGE.is_dir():
            return
    RECORDS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    if args.folder:
        folder = Path(args.folder).expanduser()
        if not folder.is_dir():
            raise SystemExit(f"{folder} is not a folder")
        if not args.label:
            raise SystemExit("with a folder, say what is in it: --label good | defect (--kind ...) | empty")
        kind = norm_kind(args.kind, args.label, print)
        todo = [(p, args.label, kind) for p in sorted(folder.iterdir())
                if p.suffix.lower() in PHOTO_EXT | VIDEO_EXT and not p.name.startswith(".")]
    else:
        todo = [(p, lab, norm_kind(k, lab, print) if lab == "defect" else "") for p, lab, k in sources_in_footage()]
    if not todo and not any(RECORDS.glob("*.json")):
        print("nothing to ingest (no footage/ and no folder given)")
        return

    settings = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    t0 = time.time()
    for i, (path, label, kind) in enumerate(todo, 1):
        sid = source_id(path)
        recf = RECORDS / f"{sid}.json"
        if recf.is_file() and not args.redo:
            old = json.loads(recf.read_text())
            if old.get("settings") == settings and old.get("label") == label and old.get("kind") == kind:
                continue
        is_video = path.suffix.lower() in VIDEO_EXT
        print(f"[{i}/{len(todo)}] {path.name} ({label}{'/' + kind if kind else ''}, {'video' if is_video else 'photo'})", flush=True)
        rec = (ingest_video if is_video else ingest_photo)(path, label, kind, cfg)
        rec["settings"] = settings
        rec["ingested"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = recf.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(recf)                                    # written whole, so a power cut never leaves half a record
    build(cfg)
    print(f"ingest done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
