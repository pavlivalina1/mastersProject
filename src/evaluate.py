import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, CLASS_LIST_PATH, EXPERIMENTS_DIR, MANIFEST_TEST,
    NUM_WORKERS,
)
from src.data.dataset import CropDiseaseDataset
from src.models.factory import build_model
from src.utils import get_device, load_json, save_json, count_parameters


def load_trained_model(model_name: str, num_classes: int, checkpoint_path=None, device=None):
    device = device or get_device()
    model = build_model(model_name, num_classes).to(device)
    checkpoint_path = checkpoint_path or (CHECKPOINTS_DIR / f"{model_name}.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def measure_inference_latency(model, device, image_size=128, num_runs=30) -> float:
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        start = time.time()
        for _ in range(num_runs):
            model(dummy)
        elapsed = time.time() - start
    return (elapsed / num_runs) * 1000


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.4), max(6, len(class_names) * 0.35)))
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(model_name: str, manifest_path=MANIFEST_TEST, tag: str = "") -> dict:
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    num_classes = len(classes)

    run_name = f"{model_name}{('_' + tag) if tag else ''}"
    model = load_trained_model(model_name, num_classes, CHECKPOINTS_DIR / f"{run_name}.pt", device)

    test_ds = CropDiseaseDataset(manifest_path, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds, labels = predict_all(model, test_loader, device)
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)

    latency_ms = measure_inference_latency(model, device)

    result = {
        "model_name": model_name,
        "run_name": run_name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "per_class_report": report,
        "num_params": count_parameters(model),
        "inference_latency_ms": latency_ms,
        "num_test_samples": len(test_ds),
    }
    save_json(result, EXPERIMENTS_DIR / f"{run_name}_metrics.json")
    plot_confusion_matrix(cm, classes, EXPERIMENTS_DIR / f"{run_name}_confusion.png")

    train_result_path = EXPERIMENTS_DIR / f"{run_name}_train.json"
    if train_result_path.exists():
        train_result = load_json(train_result_path)
        plot_training_curves(train_result["history"], EXPERIMENTS_DIR / f"{run_name}_curves.png", run_name)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    res = evaluate_model(args.model)
    print(f"{args.model}: accuracy={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f} "
          f"latency={res['inference_latency_ms']:.2f}ms params={res['num_params']}")
