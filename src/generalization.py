import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINTS_DIR, CLASS_LIST_PATH, CV_FOLDS, EXPERIMENTS_DIR, MANIFEST_OOD_TEST,
    MANIFEST_TEST, MANIFEST_TRAIN, MANIFEST_VAL, MODEL_NAMES, ROBUSTNESS_LEVELS,
    SEED,
)
from src.data.dataset import CropDiseaseDataset, apply_perturbation, base_transform, to_tensor_normalized
from src.evaluate import load_trained_model, predict_all
from src.train import train_model
from src.utils import get_device, load_json, save_json, seed_everything


def _read_all_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["filepath"], row["label"], int(row["label_idx"])))
    return rows


def _write_fold_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "label_idx"])
        writer.writerows(rows)


def run_cross_validation(model_names, k=CV_FOLDS, epochs=8) -> dict:
    all_rows = _read_all_rows([MANIFEST_TRAIN, MANIFEST_VAL, MANIFEST_TEST])
    labels = [r[2] for r in all_rows]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(np.zeros(len(labels)), labels))

    cv_dir = EXPERIMENTS_DIR / "cv_manifests"
    results = {}
    from src.evaluate import evaluate_model

    for model_name in model_names:
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            train_rows = [all_rows[i] for i in train_idx]
            val_rows = [all_rows[i] for i in val_idx]
            train_path = cv_dir / f"fold{fold_idx}_train.csv"
            val_path = cv_dir / f"fold{fold_idx}_val.csv"
            _write_fold_manifest(train_path, train_rows)
            _write_fold_manifest(val_path, val_rows)

            tag = f"cv{fold_idx}"
            train_model(model_name, manifest_train=train_path, manifest_val=val_path,
                        max_epochs=epochs, patience=3, tag=tag)
            metrics = evaluate_model(model_name, manifest_path=val_path, tag=tag)
            fold_metrics.append({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            print(f"[CV] {model_name} fold {fold_idx}: acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['macro_f1']:.4f}")

        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["macro_f1"] for m in fold_metrics]
        results[model_name] = {
            "fold_metrics": fold_metrics,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }

    save_json(results, EXPERIMENTS_DIR / "cv_results.json")
    return results


def run_ood_evaluation(model_names, manifest_path=MANIFEST_OOD_TEST, checkpoint_tag: str = "") -> dict:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        print("No OOD manifest found, skipping OOD evaluation")
        return {}

    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    ood_ds = CropDiseaseDataset(manifest_path, train=False)
    ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=0)

    results = {}
    for model_name in model_names:
        run_name = f"{model_name}_{checkpoint_tag}" if checkpoint_tag else model_name
        checkpoint = CHECKPOINTS_DIR / f"{run_name}.pt"
        if not checkpoint.exists():
            continue
        model = load_trained_model(model_name, len(classes), checkpoint, device)
        preds, labels = predict_all(model, ood_loader, device)
        accuracy = float((preds == labels).mean())

        in_dist_metrics_path = EXPERIMENTS_DIR / f"{model_name}_metrics.json"
        in_dist_accuracy = None
        if in_dist_metrics_path.exists():
            in_dist_accuracy = load_json(in_dist_metrics_path)["accuracy"]

        results[model_name] = {
            "ood_accuracy": accuracy,
            "in_distribution_accuracy": in_dist_accuracy,
            "accuracy_drop": (in_dist_accuracy - accuracy) if in_dist_accuracy is not None else None,
            "num_ood_samples": len(ood_ds),
        }
        print(f"[OOD] {model_name}: ood_acc={accuracy:.4f} "
              f"(in-dist={in_dist_accuracy})")

    out_name = f"ood_results_{checkpoint_tag}.json" if checkpoint_tag else "ood_results.json"
    save_json(results, EXPERIMENTS_DIR / out_name)
    return results


class _PerturbedTestSet(torch.utils.data.Dataset):
    def __init__(self, manifest_path, kind, level):
        import csv as _csv
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                self.samples.append((row["filepath"], int(row["label_idx"])))
        self.kind = kind
        self.level = level

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image_bgr = cv2.imread(filepath)
        image_bgr = apply_perturbation(image_bgr, self.kind, self.level)
        tensor = to_tensor_normalized(image_bgr)
        return tensor, label


def run_robustness_evaluation(model_names, perturbation_kinds=("noise", "blur", "brightness", "rotation"),
                               levels=ROBUSTNESS_LEVELS) -> dict:
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    results = {}

    for model_name in model_names:
        checkpoint = CHECKPOINTS_DIR / f"{model_name}.pt"
        if not checkpoint.exists():
            continue
        model = load_trained_model(model_name, len(classes), checkpoint, device)
        model_result = {}
        for kind in perturbation_kinds:
            level_accuracies = []
            for level in levels:
                ds = _PerturbedTestSet(MANIFEST_TEST, kind, level)
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
                preds, labels = predict_all(model, loader, device)
                acc = float((preds == labels).mean())
                level_accuracies.append(acc)
                print(f"[Robustness] {model_name} {kind} level={level}: acc={acc:.4f}")
            model_result[kind] = {"levels": list(levels), "accuracy": level_accuracies}
        results[model_name] = model_result

    save_json(results, EXPERIMENTS_DIR / "robustness_results.json")
    return results


def main():
    seed_everything(SEED)
    run_cross_validation(MODEL_NAMES)
    run_ood_evaluation(MODEL_NAMES)
    run_robustness_evaluation(MODEL_NAMES)


if __name__ == "__main__":
    main()
