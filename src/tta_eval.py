import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import (
    CHECKPOINTS_DIR, CLASS_LIST_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MANIFEST_OOD_TEST, MODEL_NAMES,
)
from src.data.dataset import read_manifest
from src.evaluate import load_trained_model, predict_all
from src.data.dataset import CropDiseaseDataset
from src.utils import get_device, load_json, save_json
from src.config import EXPERIMENTS_DIR
from torch.utils.data import DataLoader


def _tta_views():
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    to_tensor = transforms.ToTensor()

    return [
        transforms.Compose([resize, to_tensor, normalize]),
        transforms.Compose([resize, transforms.RandomHorizontalFlip(p=1.0), to_tensor, normalize]),
        transforms.Compose([resize, transforms.RandomRotation((15, 15)), to_tensor, normalize]),
        transforms.Compose([resize, transforms.RandomRotation((-15, -15)), to_tensor, normalize]),
    ]


@torch.no_grad()
def evaluate_with_tta(model, manifest_path, device):
    rows = read_manifest(manifest_path)
    views = _tta_views()
    correct = 0
    for filepath, label in rows:
        image = Image.open(filepath).convert("RGB")
        batch = torch.stack([view(image) for view in views]).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).mean(dim=0)
        pred = probs.argmax().item()
        correct += int(pred == label)
    return correct / len(rows)


@torch.no_grad()
def evaluate_single_view(model, manifest_path, device):
    ds = CropDiseaseDataset(manifest_path, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    preds, labels = predict_all(model, loader, device)
    return float((preds == labels).mean())


def main(model_names=MODEL_NAMES, tag="da"):
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    results = {}
    for model_name in model_names:
        run_name = f"{model_name}_{tag}" if tag else model_name
        checkpoint = CHECKPOINTS_DIR / f"{run_name}.pt"
        if not checkpoint.exists():
            print(f"No checkpoint for {run_name}, skipping")
            continue
        model = load_trained_model(model_name, len(classes), checkpoint, device)
        single = evaluate_single_view(model, MANIFEST_OOD_TEST, device)
        tta = evaluate_with_tta(model, MANIFEST_OOD_TEST, device)
        results[model_name] = {"single_view_accuracy": single, "tta_accuracy": tta}
        print(f"[TTA] {model_name}: single-view={single:.4f} -> TTA={tta:.4f}")

    save_json(results, EXPERIMENTS_DIR / "tta_results.json")
    return results


if __name__ == "__main__":
    main()
