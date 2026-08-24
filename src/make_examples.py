import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import (
    CLASS_LIST_PATH, IMAGENET_MEAN, IMAGENET_STD, MANIFEST_OOD_TEST,
    MANIFEST_TEST, REPORTS_DIR,
)
from src.data.dataset import CropDiseaseDataset
from src.evaluate import load_trained_model
from src.utils import get_device, load_json
from src.xai.gradcam_explain import explain_gradcam

MODEL_NAME = "efficientnet_b0"


def make_panel(model, model_name, dataset, indices, classes, device, title, out_path):
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for col, idx in enumerate(indices):
        image_tensor, label = dataset[idx]
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            confidence = probs[pred].item()

        rgb_image, _, overlay = explain_gradcam(
            model, model_name, image_tensor, IMAGENET_MEAN, IMAGENET_STD,
            method="gradcam++", target_class=pred)

        correct = pred == label
        color = "green" if correct else "red"

        axes[0, col].imshow(rgb_image)
        axes[0, col].axis("off")
        axes[0, col].set_title(f"Правильний клас:\n{classes[label]}", fontsize=9)

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")
        mark = "✓" if correct else "✗"
        axes[1, col].set_title(
            f"{mark} Передбачення: {classes[pred]}\n(впевненість {confidence:.1%})",
            fontsize=9, color=color,
        )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    model = load_trained_model(MODEL_NAME, len(classes),
                                f"experiments/checkpoints/{MODEL_NAME}.pt", device)
    model_da = load_trained_model(MODEL_NAME, len(classes),
                                   f"experiments/checkpoints/{MODEL_NAME}_da.pt", device)

    pv_ds = CropDiseaseDataset(MANIFEST_TEST, train=False)
    pd_ds = CropDiseaseDataset(MANIFEST_OOD_TEST, train=False)

    rng = random.Random(201)
    pv_indices = rng.sample(range(len(pv_ds)), 4)
    make_panel(model, MODEL_NAME, pv_ds, pv_indices, classes, device,
               f"PlantVillage - приклади передбачень моделі {MODEL_NAME} (Grad-CAM++)",
               REPORTS_DIR / "examples_plantvillage_gradcampp.png")

    rng = random.Random(203)
    pd_indices = rng.sample(range(len(pd_ds)), 4)
    make_panel(model_da, MODEL_NAME, pd_ds, pd_indices, classes, device,
               f"PlantDoc (після доменної адаптації) - приклади передбачень {MODEL_NAME} (Grad-CAM++)",
               REPORTS_DIR / "examples_plantdoc_gradcampp.png")


if __name__ == "__main__":
    main()
