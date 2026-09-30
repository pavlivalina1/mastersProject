import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, CLASS_LIST_PATH, IMAGE_SIZE, MANIFEST_OOD_TEST,
    MANIFEST_TEST, NUM_WORKERS, PROJECT_ROOT,
)
from src.data.dataset import CropDiseaseDataset, read_manifest
from src.evaluate import load_trained_model
from src.utils import get_device, load_json

MOBILE_DIR = PROJECT_ROOT / "mobile" / "assets"
MOBILE_MODELS_DIR = MOBILE_DIR / "models"
MOBILE_DEBUG_DIR = MOBILE_DIR / "debug_samples"

LOGIT_ATOL = 1e-3
NUM_DEBUG_SAMPLES = 5

# mobilenet_v2's classifier is a plain Linear directly on globally-pooled
# features, so a class activation map (CAM, Zhou et al. 2016) can be computed
# from the last conv feature map + that Linear layer's weights in a single
# forward pass — no gradients/backprop needed, which on-device ONNX Runtime
# (inference-only) can't do anyway. This isn't "true" Grad-CAM, but it's the
# same idea and visually equivalent for this architecture.
CAM_CAPABLE_MODELS = {"mobilenet_v2"}


class _WithCAM(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.classifier_weight = base_model.classifier[1].weight  # [num_classes, channels]

    def forward(self, x):
        features = self.base_model.features(x)  # [B, C, H, W]
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        logits = self.base_model.classifier(torch.flatten(pooled, 1))
        cam = torch.einsum("bkhw,ck->bchw", features, self.classifier_weight)
        return logits, cam


def export_onnx(model: torch.nn.Module, out_path: Path, image_size: int = IMAGE_SIZE,
                 with_cam: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    if with_cam:
        output_names = ["logits", "cam"]
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}}
    else:
        output_names = ["logits"]
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(out_path))


def _compare_on_manifest(model, session, manifest_path, device, label: str,
                          with_cam: bool = False, classifier_bias=None) -> None:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        print(f"[{label}] manifest missing, skipping parity check")
        return

    ds = CropDiseaseDataset(manifest_path, train=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    input_name = session.get_inputs()[0].name

    max_abs_diff = 0.0
    max_cam_consistency_diff = 0.0
    mismatched_top1 = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            if with_cam:
                torch_logits, _ = outputs
                torch_logits = torch_logits.cpu().numpy()
                onnx_logits, onnx_cam = session.run(None, {input_name: images.numpy()})
                # CAM correctness check: mean(cam[c]) + bias[c] must equal logits[c],
                # since cam[c] = weight[c] . features (pre-pool) and logits[c] =
                # weight[c] . mean(features) + bias[c] (adaptive_avg_pool = spatial mean).
                cam_mean_plus_bias = onnx_cam.mean(axis=(2, 3)) + classifier_bias
                max_cam_consistency_diff = max(
                    max_cam_consistency_diff,
                    float(np.abs(cam_mean_plus_bias - onnx_logits).max()),
                )
            else:
                torch_logits = outputs.cpu().numpy()
                onnx_logits = session.run(None, {input_name: images.numpy()})[0]

            max_abs_diff = max(max_abs_diff, float(np.abs(torch_logits - onnx_logits).max()))
            mismatched_top1 += int((torch_logits.argmax(1) != onnx_logits.argmax(1)).sum())
            total += len(labels)

    print(f"[{label}] n={total} max_abs_logit_diff={max_abs_diff:.6f} "
          f"top1_mismatches={mismatched_top1}"
          + (f" max_cam_consistency_diff={max_cam_consistency_diff:.6f}" if with_cam else ""))
    assert max_abs_diff < LOGIT_ATOL, (
        f"[{label}] ONNX output diverges from PyTorch: max_abs_diff={max_abs_diff} >= {LOGIT_ATOL}"
    )
    assert mismatched_top1 == 0, f"[{label}] {mismatched_top1} top-1 predictions differ from PyTorch"
    if with_cam:
        assert max_cam_consistency_diff < LOGIT_ATOL, (
            f"[{label}] CAM output inconsistent with logits: "
            f"max diff={max_cam_consistency_diff} >= {LOGIT_ATOL}"
        )


def _write_debug_samples(model, session, device, classes, with_cam: bool = False) -> None:
    if MANIFEST_OOD_TEST.exists() and MANIFEST_OOD_TEST.stat().st_size > 0:
        rows = read_manifest(MANIFEST_OOD_TEST)
    else:
        rows = read_manifest(MANIFEST_TEST)
    rows = rows[:NUM_DEBUG_SAMPLES]

    from src.data.dataset import base_transform
    from PIL import Image

    transform = base_transform(train=False)
    input_name = session.get_inputs()[0].name

    MOBILE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, (filepath, label_idx) in enumerate(rows):
        image = Image.open(filepath).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor.to(device))
        if with_cam:
            torch_logits = outputs[0].cpu().numpy()[0]
            onnx_logits, onnx_cam = session.run(None, {input_name: tensor.numpy()})
            onnx_logits = onnx_logits[0]
        else:
            torch_logits = outputs.cpu().numpy()[0]
            onnx_logits = session.run(None, {input_name: tensor.numpy()})[0][0]
            onnx_cam = None

        dest_name = f"sample_{i}{Path(filepath).suffix.lower()}"
        shutil.copy(filepath, MOBILE_DEBUG_DIR / dest_name)

        top3_idx = np.argsort(onnx_logits)[::-1][:3]
        probs = np.exp(onnx_logits - onnx_logits.max())
        probs = probs / probs.sum()
        entry = {
            "image_file": dest_name,
            "true_label_idx": label_idx,
            "true_label": classes[label_idx],
            "expected_top3": [
                {"label_idx": int(idx), "label": classes[idx], "confidence": float(probs[idx])}
                for idx in top3_idx
            ],
            "onnx_vs_pytorch_max_abs_diff": float(np.abs(torch_logits - onnx_logits).max()),
        }
        if onnx_cam is not None:
            top1_idx = int(top3_idx[0])
            entry["expected_cam_top1"] = onnx_cam[0][top1_idx].tolist()
        samples.append(entry)

    with open(MOBILE_DEBUG_DIR / "expected.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(samples)} debug samples + expected outputs to {MOBILE_DEBUG_DIR}")


def main(model_name: str = "mobilenet_v2", tag: str = "da") -> Path:
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    num_classes = len(classes)

    run_name = f"{model_name}_{tag}" if tag else model_name
    checkpoint = CHECKPOINTS_DIR / f"{run_name}.pt"
    base_model = load_trained_model(model_name, num_classes, checkpoint, device)

    with_cam = model_name in CAM_CAPABLE_MODELS
    model = _WithCAM(base_model).to(device) if with_cam else base_model
    model.eval()
    classifier_bias = (
        base_model.classifier[1].bias.detach().cpu().numpy() if with_cam else None
    )

    out_path = MOBILE_MODELS_DIR / f"{run_name}.onnx"
    export_onnx(model, out_path, with_cam=with_cam)
    print(f"Exported {run_name} -> {out_path} (with_cam={with_cam})")

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    _compare_on_manifest(model, session, MANIFEST_TEST, device, label="PlantVillage test",
                          with_cam=with_cam, classifier_bias=classifier_bias)
    _compare_on_manifest(model, session, MANIFEST_OOD_TEST, device, label="PlantDoc OOD test",
                          with_cam=with_cam, classifier_bias=classifier_bias)
    _write_debug_samples(model, session, device, classes, with_cam=with_cam)

    print("Parity check passed: ONNX export matches PyTorch model.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2")
    parser.add_argument("--tag", default="da")
    args = parser.parse_args()
    main(args.model, args.tag)
