import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import (
    CHECKPOINTS_DIR, CLASS_LIST_PATH, IMAGENET_MEAN, IMAGENET_STD, MANIFEST_TEST,
    MANIFEST_TRAIN, MODEL_NAMES, SEED, XAI_DIR,
)
from src.data.dataset import CropDiseaseDataset, base_transform, read_manifest
from src.evaluate import load_trained_model
from src.utils import get_device, load_json, save_json, seed_everything
from src.xai.faithfulness import insertion_deletion_auc
from src.xai.gradcam_explain import explain_gradcam
from src.xai.lime_explain import explain_lime
from src.xai.shap_explain import explain_shap

NUM_SAMPLE_IMAGES = 5


def pick_sample_indices(manifest_path, n, seed=SEED):
    rows = read_manifest(manifest_path)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return idx[:n]


def run_xai_for_model(model_name, sample_indices, background_tensors, classes, device):
    checkpoint = CHECKPOINTS_DIR / f"{model_name}.pt"
    if not checkpoint.exists():
        print(f"No checkpoint for {model_name}, skipping XAI")
        return {}

    model = load_trained_model(model_name, len(classes), checkpoint, device)
    test_ds = CropDiseaseDataset(MANIFEST_TEST, train=False)

    faithfulness_scores = {"gradcam": [], "gradcam++": [], "lime": [], "shap": []}

    for sample_i, idx in enumerate(sample_indices):
        image_tensor, label = test_ds[idx]
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0)).argmax(1).item()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        rgb_image, cam_map, cam_overlay = explain_gradcam(
            model, model_name, image_tensor, IMAGENET_MEAN, IMAGENET_STD,
            method="gradcam", target_class=pred)
        axes[0].imshow(rgb_image); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cam_overlay); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        score = insertion_deletion_auc(model, image_tensor, cam_map, pred, device)
        faithfulness_scores["gradcam"].append(score)

        _, campp_map, campp_overlay = explain_gradcam(
            model, model_name, image_tensor, IMAGENET_MEAN, IMAGENET_STD,
            method="gradcam++", target_class=pred)
        axes[2].imshow(campp_overlay); axes[2].set_title("Grad-CAM++"); axes[2].axis("off")
        score = insertion_deletion_auc(model, image_tensor, campp_map, pred, device)
        faithfulness_scores["gradcam++"].append(score)

        try:
            lime_overlay, _, _ = explain_lime(model, rgb_image, device, num_samples=300)
            axes[3].imshow(lime_overlay); axes[3].set_title("LIME"); axes[3].axis("off")
        except Exception as exc:
            print(f"LIME failed for {model_name} sample {sample_i}: {exc}")
            axes[3].axis("off"); axes[3].set_title("LIME (failed)")

        fig.suptitle(f"{model_name} — true={classes[label]} pred={classes[pred]}")
        fig.tight_layout()
        fig.savefig(XAI_DIR / f"{model_name}_sample{sample_i}.png", dpi=130)
        plt.close(fig)

        try:
            shap_map = explain_shap(model, background_tensors, image_tensor, target_class=pred)
            score = insertion_deletion_auc(model, image_tensor, shap_map, pred, device)
            faithfulness_scores["shap"].append(score)
        except Exception as exc:
            print(f"SHAP failed for {model_name} sample {sample_i}: {exc}")

    summary = {}
    for method, scores in faithfulness_scores.items():
        if not scores:
            continue
        summary[method] = {
            "mean_deletion_auc": sum(s["deletion_auc"] for s in scores) / len(scores),
            "mean_insertion_auc": sum(s["insertion_auc"] for s in scores) / len(scores),
        }
    return summary


def main():
    seed_everything(SEED)
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)

    sample_indices = pick_sample_indices(MANIFEST_TEST, NUM_SAMPLE_IMAGES)

    train_ds = CropDiseaseDataset(MANIFEST_TRAIN, train=False)
    bg_indices = random.Random(SEED).sample(range(len(train_ds)), min(20, len(train_ds)))
    background_tensors = torch.stack([train_ds[i][0] for i in bg_indices]).to(device)

    all_results = {}
    for model_name in MODEL_NAMES:
        print(f"Running XAI for {model_name}...")
        all_results[model_name] = run_xai_for_model(
            model_name, sample_indices, background_tensors, classes, device)

    save_json(all_results, XAI_DIR / "faithfulness_scores.json")


if __name__ == "__main__":
    main()
