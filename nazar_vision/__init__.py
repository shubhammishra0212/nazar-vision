"""NAZAR vision: washer inspection by anomaly detection (PatchCore-style), offline on the CPU."""
from pathlib import Path

VERSION = "2.0.0"
MODEL_NAME = "patchcore-wrn50"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

CROP = 224          # backbone input size (pixels)
GRID = 28           # patch grid the backbone returns (CROP / 8)
