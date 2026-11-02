"""Reading, listing and encoding images."""
import base64
import hashlib
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def decode(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> BGR uint8. Raises ValueError with a readable message."""
    if not data:
        raise ValueError("empty body: send JPEG or PNG bytes")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("body is not a JPEG or PNG image")
    return img


def load(path) -> np.ndarray:
    return decode(Path(path).read_bytes())


def list_images(folder) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("."))


def sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
