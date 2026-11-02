"""FAKE iPhone photos and videos made from the existing washer photos, to TEST nazar_vision.ingest only.
Never train a real model on them.

    python -m nazar_vision.fake_footage photos OUT_DIR      # HEIC/JPG/PNG test set (see PHOTO_CASES)
    python -m nazar_vision.fake_footage videos OUT_DIR      # footage/ tree with H.264, HEVC 10-bit HLG, rotated clips

Photos written to OUT_DIR/good, OUT_DIR/defect/chipped, OUT_DIR/empty with EXIF capture times, including:
a burst (3 shots 0.4 s apart, washer not moved), an EXIF-orientation-6 HEIC stored sideways, a 48 MP HEIC,
a blurred shot, a washer cut by the frame edge, two washers, a PNG without EXIF, an empty belt.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import DATA, images, locate, synth
from .ingest import tool

T0 = 1793600000.0          # 2026-11-02 ~13:03 IST, fake capture clock


def exif_for(t, orientation=1):
    import time as _t
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Apple", "iPhone (FAKE TEST)"
    ex[0x0112] = orientation
    sub = ex.get_ifd(0x8769)
    sub[0x9003] = _t.strftime("%Y:%m:%d %H:%M:%S", _t.localtime(int(t)))
    sub[0x9291] = f"{int(round((t % 1) * 1000)):03d}"
    return ex


def save(bgr, path, t=None, orientation=1, heic=False, transpose=None):
    """Write a JPEG (or HEIC via sips) with EXIF. transpose rotates the stored pixels (orientation tag undoes it)."""
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if transpose is not None:
        im = im.transpose(transpose)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        im.save(path)
        return path
    jpg = path.with_suffix(".jpg") if heic else path
    im.save(jpg, quality=95, exif=exif_for(t, orientation) if t is not None else Image.Exif())
    if heic:
        subprocess.run(["sips", "-s", "format", "heic", str(jpg), "--out", str(path)], check=True, capture_output=True)
        jpg.unlink()
    return path


def shifted(img, dx, dy, angle=0.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)   # reflect would mirror a 2nd washer in


def to_1080(img):
    return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)


def photos(out):
    good = images.list_images(DATA / "good")
    shutil.rmtree(out, ignore_errors=True)
    t = T0
    cases = []
    for i, p in enumerate(good[:12]):                        # 12 placements, washer moved around the belt
        img = to_1080(images.load(p))
        dx, dy = ((i % 4) - 1.5) * 160, ((i // 4) - 1) * 120
        img = shifted(img, dx, dy, angle=(i % 5 - 2) * 5)
        heic = i % 3 != 2
        f = save(img, out / "good" / f"IMG_{4100 + i}.{'HEIC' if heic else 'JPG'}", t, heic=heic)
        cases.append((f.name, "good, own placement"))
        t += 20
    base = to_1080(images.load(good[0]))
    for k in range(3):                                       # burst: same placement 0.4 s apart
        f = save(base, out / "good" / f"IMG_{4200 + k}.HEIC", t + 0.4 * k, heic=True)
        cases.append((f.name, "burst -> one group, near-copies dropped"))
    t += 20
    f = save(shifted(base, 120, -60), out / "good" / "IMG_4300.HEIC", t, orientation=6, heic=True,
             transpose=Image.Transpose.ROTATE_90)
    cases.append((f.name, "stored sideways + EXIF orientation 6"))
    t += 20
    big = cv2.resize(shifted(base, -200, 50), (8064, 4536), interpolation=cv2.INTER_CUBIC)
    f = save(big, out / "good" / "IMG_4301.HEIC", t, heic=True)
    cases.append((f.name, "48 MP class (8064 x 4536)"))
    t += 20
    f = save(cv2.GaussianBlur(shifted(base, 60, 60), (0, 0), 6), out / "good" / "IMG_4302.HEIC", t, heic=True)
    cases.append((f.name, "blurred -> dropped"))
    t += 20
    w = locate.find(base)
    f = save(shifted(base, 1920 - 0.5 * w.r - w.cx, 0), out / "good" / "IMG_4303.HEIC", t, heic=True)
    cases.append((f.name, "washer cut by the frame edge -> dropped"))
    t += 20
    two = shifted(base, 300, 0)
    r = int(w.r * 1.15)
    patch = base[int(w.cy) - r:int(w.cy) + r, int(w.cx) - r:int(w.cx) + r]
    disc = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(disc, (r, r), int(w.r * 1.04), 255, -1)
    roi = two[620:620 + 2 * r, 300:300 + 2 * r]              # second washer on the belt, round cut-out
    roi[disc > 0] = patch[disc > 0]
    f = save(two, out / "good" / "IMG_4304.HEIC", t, heic=True)
    cases.append((f.name, "two washers -> dropped"))
    t += 20
    f = save(shifted(base, -300, 100, 8), out / "good" / "Screenshot-no-exif.png")
    cases.append((f.name, "PNG without EXIF -> own group, file time"))
    for j, p in enumerate(good[12:14]):                     # fake 'chipped' defects (synthetic notch)
        img = to_1080(images.load(p))
        ww = locate.find(img)
        bad = synth.draw(img, ww, "notch", np.random.default_rng(j))
        f = save(shifted(bad, -100 * j, 80 * j, 6 * j), out / "defect" / "chipped" / f"IMG_{4400 + j}.HEIC", t, heic=True)
        cases.append((f.name, "FAKE chipped defect"))
        t += 20
    belt = synth.remove_washer(base, w)
    f = save(belt, out / "empty" / "IMG_4500.HEIC", t, heic=True)
    cases.append((f.name, "empty belt"))
    for name, what in cases:
        print(f"  {name:26s} {what}")


def write_clip(frames, path, fps=30, codec="h264", hlg=False, rotate=0):
    """Encode BGR frames. hlg=True: convert to HLG BT.2020 10-bit HEVC (iPhone HDR style)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            cv2.imwrite(f"{td}/{i:05d}.png", f)
        vf = []
        if hlg:
            vf = ["zscale=tin=bt709:pin=bt709:min=bt709:t=arib-std-b67:p=bt2020:m=bt2020nc:npl=203:r=tv",
                  "format=p010le"]
        # HEVC Main10 with Apple's hardware encoder, like the iPhone (this ffmpeg's libx265 is 8-bit only)
        enc = (["-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-pix_fmt", "p010le", "-b:v", "8M", "-tag:v", "hvc1"]
               + (["-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc"] if hlg else [])
               if codec == "hevc" else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        tmp = path.with_suffix(".tmp" + path.suffix) if rotate else path
        cmd = [tool("ffmpeg"), "-v", "error", "-y", "-framerate", str(fps), "-i", f"{td}/%05d.png"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        subprocess.run(cmd + enc + [str(tmp)], check=True)
        if rotate:
            subprocess.run([tool("ffmpeg"), "-v", "error", "-y", "-display_rotation:v:0", str(rotate), "-i", str(tmp),
                            "-c", "copy", str(path)], check=True)
            tmp.unlink()


def clip_frames(img, seconds=10, fps=30, positions=7, blur_moves=True, rng=None):
    """Washer moved to `positions` spots (and turned), held ~still, motion-blurred while moving."""
    rng = rng or np.random.default_rng(0)
    spots = [(rng.uniform(-260, 260), rng.uniform(-160, 160), rng.uniform(-12, 12)) for _ in range(positions)]
    n = seconds * fps
    per = n // positions
    frames = []
    for k in range(n):
        s, f = divmod(k, per)
        s = min(s, positions - 1)
        dx, dy, a = spots[s]
        moving = blur_moves and f >= per - 6 and s < positions - 1
        if moving:                                           # hand moving it: in between + motion blur
            nx, ny, na = spots[s + 1]
            u = (f - (per - 6)) / 6
            img2 = shifted(img, dx + (nx - dx) * u, dy + (ny - dy) * u, a)
            kern = np.zeros((1, 31), np.float32)
            kern[0, :] = 1 / 31
            frames.append(cv2.filter2D(img2, -1, kern))
        else:
            noise = rng.normal(0, 1.5, img.shape).astype(np.float32)
            frames.append(np.clip(shifted(img, dx, dy, a).astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return frames


def videos(out):
    good = images.list_images(DATA / "good")
    shutil.rmtree(out, ignore_errors=True)
    for i, p in enumerate(good[:10]):
        img = to_1080(images.load(p))
        name = f"IMG_{5100 + i}.MOV"
        if i == 0:
            write_clip(clip_frames(img, rng=np.random.default_rng(i)), out / "good" / name, codec="hevc", hlg=True)
            print(f"  {name}: HEVC 10-bit HLG (iPhone HDR style)")
        elif i == 1:
            frames = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in clip_frames(img, rng=np.random.default_rng(i))]
            write_clip(frames, out / "good" / name, codec="hevc", rotate=90)
            print(f"  {name}: HEVC 10-bit SDR, stored sideways + rotation metadata 90")
        else:
            write_clip(clip_frames(img, rng=np.random.default_rng(i)), out / "good" / name.replace(".MOV", ".mp4"))
            print(f"  {name.replace('.MOV', '.mp4')}: H.264 1080p")
    for j, p in enumerate(good[10:12]):
        img = to_1080(images.load(p))
        bad = synth.draw(img, locate.find(img), "crack", np.random.default_rng(j))
        write_clip(clip_frames(bad, seconds=8, positions=6, rng=np.random.default_rng(50 + j)),
                   out / "defect" / "crack" / f"IMG_{5200 + j}.MOV", codec="hevc", hlg=True)
        print(f"  defect/crack/IMG_{5200 + j}.MOV: FAKE crack, HEVC 10-bit HLG")
    base = to_1080(images.load(good[0]))
    belt = synth.remove_washer(base, locate.find(base))
    write_clip([belt] * 90, out / "empty" / "IMG_5300.mp4")
    print("  empty/IMG_5300.mp4: belt only")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["photos", "videos"])
    ap.add_argument("out")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (photos if args.what == "photos" else videos)(out)
    print(f"FAKE test {args.what} in {out}")


if __name__ == "__main__":
    main()
