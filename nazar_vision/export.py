"""Export the feature extractor to models/backbone.onnx. The ONLY step that needs torch.

    ~/Downloads/NAZAR/10-AI/.venv/bin/python -m nazar_vision.export

Graph: image (N,3,224,224, ImageNet-normalised float32) -> features (N,512,28,28)
  WideResNet50-2 (ImageNet weights) layer2 (512 ch, 28x28) and layer3 (1024 ch, 14x14)
  -> 3x3 average pool on each (every patch also sees its neighbours)
  -> layer3 upsampled to 28x28, concatenated (1536 ch)
  -> fixed random projection to 512 ch (seeded Gaussian matrix baked into the graph).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from . import CROP, MODELS


class PatchEmbedder(torch.nn.Module):
    def __init__(self, dim=512, seed=7):
        super().__init__()
        net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        net.eval()
        self.trunk = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        self.layer3 = net.layer3
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(dim, 512 + 1024, generator=g) / dim ** 0.5
        self.register_buffer("proj", proj[:, :, None, None])     # 1x1 conv weight

    def forward(self, x):
        a = self.trunk(x)
        b = self.layer3(a)
        a = F.avg_pool2d(a, 3, 1, 1)
        b = F.avg_pool2d(b, 3, 1, 1)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return F.conv2d(torch.cat([a, b], 1), self.proj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model = PatchEmbedder(args.dim, args.seed).eval()
    MODELS.mkdir(exist_ok=True)
    out = MODELS / "backbone.onnx"
    x = torch.randn(2, 3, CROP, CROP)
    with torch.no_grad():
        ref = model(x).numpy()
        torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["features"],
                          dynamic_axes={"image": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    meta = {"arch": "wide_resnet50_2", "weights": "IMAGENET1K_V1", "layers": ["layer2", "layer3"],
            "projection": f"gaussian 1536->{args.dim}, seed {args.seed}", "dim": args.dim,
            "grid": int(got.shape[-1]), "onnx_vs_torch_max_diff": diff}
    (MODELS / "backbone.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), features {got.shape}, max diff vs torch {diff:.1e}")


if __name__ == "__main__":
    main()
