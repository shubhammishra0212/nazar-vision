"""Patch features from the exported WideResNet50 backbone, run by ONNX Runtime on the CPU."""
import numpy as np

from . import MODELS

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255     # ImageNet statistics, in 0..255 units
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class Backbone:
    def __init__(self, path=MODELS / "backbone.onnx"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    @staticmethod
    def prepare(crops):
        """list of RGB uint8 (224, 224, 3) -> (N, 3, 224, 224) float32, ImageNet-normalised."""
        x = (np.asarray(crops, np.float32) - MEAN) / STD
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def embed(self, crops, batch=8):
        """-> (N, grid*grid, dim) float32: one feature vector per patch, row-major over the 28x28 grid."""
        out = []
        for i in range(0, len(crops), batch):
            f = self.session.run(None, {self.input: self.prepare(crops[i:i + batch])})[0]
            n, d, h, w = f.shape
            out.append(f.reshape(n, d, h * w).transpose(0, 2, 1))
        return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)
