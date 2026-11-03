"""HTTP service for the NAZAR console (CONTRACT.md section 4). Standard library only, 127.0.0.1:8765.

    python -m nazar_vision.service [--port 8765] [--test-images DIR]

GET  /health                     model, thresholds, bank size, version
POST /inspect[?hold=&reject=]    JPEG/PNG bytes -> verdict PASS | HOLD | REJECT | NO_PART, score, crop, heatmap, box
POST /locate                     JPEG/PNG bytes (a small frame is enough) -> is a whole washer in view, and where
POST /capture?label=good|defect[&kind=scratch]   JPEG/PNG bytes -> data/capture/<label>/[<kind>/]<timestamp>.jpg

Every response is JSON with Access-Control-Allow-Origin: *. Errors are {"ok": false, "error": "..."} with HTTP 200.
The service keeps running without a trained model: /capture still works, /health and /inspect say what is missing.
--test-images DIR: every posted /inspect frame is replaced by the next photo from DIR (demo without a camera).
"""
import argparse
import itertools
import json
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import DATA, MODELS, ROOT, VERSION, images, locate

MAX_BODY = 30 * 1024 * 1024


class State:
    def __init__(self, test_images=None):
        self.lock = threading.Lock()          # one inspection at a time: steady latency, no thread fights
        self.inspector = None
        self.error = None
        self.test_cycle = None
        self.test_count = 0
        if test_images:
            paths = images.list_images(test_images)
            if not paths:
                raise SystemExit(f"--test-images: no JPEG/PNG photos in {test_images}")
            self.test_cycle = itertools.cycle(paths)
            self.test_count = len(paths)
        self.load()

    def load(self):
        from .inspector import Inspector
        try:
            self.inspector = Inspector.load(MODELS)
            self.error = None
        except Exception as e:                 # missing or broken model: keep serving /capture
            self.inspector, self.error = None, f"model not loaded: {e}"


def _num(q, key):
    v = q.get(key, [""])[0].strip()
    if not v:
        return None
    try:
        x = float(v)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {v!r}")
    if not 0 < x < 1e6:
        raise ValueError(f"{key} must be > 0")
    return x


class Handler(BaseHTTPRequestHandler):
    server_version = "nazar-vision/" + VERSION
    state: State = None
    quiet = False

    # ---------------------------------------------------------------- plumbing
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def _body(self):
        n = self.headers.get("Content-Length")
        if n is None:
            raise ValueError("Content-Length header missing: send the image as the request body")
        n = int(n)
        if n <= 0:
            raise ValueError("empty body: send JPEG or PNG bytes")
        if n > MAX_BODY:
            raise ValueError(f"body too large ({n} bytes, max {MAX_BODY})")
        return self.rfile.read(n)

    def _route(self, method):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            fn = {("GET", "/health"): self.health, ("GET", "/"): self.health,
                  ("POST", "/inspect"): self.inspect, ("POST", "/locate"): self.locate,
                  ("POST", "/capture"): self.capture}.get((method, url.path))
            if fn is None:
                self._send({"ok": False, "error": f"unknown endpoint {method} {url.path}"})
            else:
                self._send(fn(q))
        except ValueError as e:
            self._send({"ok": False, "error": str(e)})
        except Exception as e:                               # never crash, never hang the console
            traceback.print_exc()
            try:
                self._send({"ok": False, "error": f"internal error: {type(e).__name__}: {e}"})
            except Exception:
                pass

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}\n")

    # ---------------------------------------------------------------- endpoints
    def health(self, q):
        st = self.state
        if st.inspector is None:
            st.load()                                  # picks up a model trained after start-up
        if st.inspector is None:
            return {"ok": False, "error": st.error, "version": VERSION}
        out = st.inspector.health()
        out["version"] = VERSION
        if st.test_cycle:
            out["test_images"] = st.test_count
        return out

    def inspect(self, q):
        st = self.state
        hold, reject = _num(q, "hold"), _num(q, "reject")
        data = self._body()
        if st.test_cycle:
            with st.lock:
                path = next(st.test_cycle)
            data = path.read_bytes()
        img = images.decode(data)
        if st.inspector is None:
            st.load()
            if st.inspector is None:
                return {"ok": False, "error": st.error}
        h = st.inspector.hold if hold is None else hold
        r = st.inspector.reject if reject is None else reject
        if r < h:
            raise ValueError(f"reject ({r}) must not be below hold ({h})")
        t0 = time.perf_counter()
        with st.lock:
            out = st.inspector.inspect(img, h, r)
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        if st.test_cycle:
            out["test_image"] = path.name
        if not self.quiet:
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} inspect {out['verdict']:7s} score {out['score']:.3f} "
                             f"{out['ms']} ms{'  ' + path.name if st.test_cycle else ''}\n")
        return out

    def locate(self, q):
        """The console's camera part trigger: only the washer finder, no AI. A washer cut by the frame edge is not
        found, so found = true means a whole washer is in view. Not under the inspection lock: it never waits."""
        img = images.decode(self._body())
        st = self.state
        if st.inspector is None:
            st.load()
        t0 = time.perf_counter()
        w = st.inspector.find(img) if st.inspector else locate.find(img)
        h, wd = img.shape[:2]
        out = {"ok": True, "found": w is not None, "w": wd, "h": h}
        if w is not None:
            out.update(cx=round(w.cx, 1), cy=round(w.cy, 1), r=round(w.r, 1))
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    def capture(self, q):
        label = q.get("label", [""])[0].strip().lower()
        if label not in ("good", "defect"):
            raise ValueError("label must be good or defect")
        kind = q.get("kind", [""])[0].strip().lower()
        kind = re.sub(r"[^a-z0-9_-]+", "-", kind).strip("-")[:40]
        data = self._body()
        img = images.decode(data)                       # refuse anything that is not an image
        folder = DATA / "capture" / label
        if label == "defect" and kind:
            folder = folder / kind
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = folder / f"{stamp}.jpg"
        if data[:3] == b"\xff\xd8\xff":
            path.write_bytes(data)                      # JPEG: keep the original bytes
        else:
            import cv2
            if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("could not write the photo")
        return {"ok": True, "saved": str(path.relative_to(ROOT)), "label": label, "kind": kind or None,
                "count": len(images.list_images(folder))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--test-images", help="replace each posted frame with the next photo from this folder")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    Handler.state = State(args.test_images)
    Handler.quiet = args.quiet
    st = Handler.state
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    if st.inspector:
        h = st.inspector.health()
        print(f"nazar-vision {VERSION} on http://{args.host}:{args.port}  bank {h['bank']}  "
              f"hold {h['hold']}  reject {h['reject']}", flush=True)
    else:
        print(f"nazar-vision {VERSION} on http://{args.host}:{args.port}  WITHOUT a model ({st.error}); "
              "/capture works", flush=True)
    if st.test_cycle:
        print(f"TEST MODE: posted frames are replaced by {st.test_count} photos from {args.test_images}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
