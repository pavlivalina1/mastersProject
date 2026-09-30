import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOPPING_PATIENCE, EXPERIMENTS_DIR,
    LEARNING_RATE, MANIFEST_TRAIN, MANIFEST_VAL, MAX_EPOCHS, NUM_WORKERS, SEED,
    WEIGHT_DECAY, CLASS_LIST_PATH,
)
from src.data.dataset import CropDiseaseDataset
from src.models.factory import build_model, trainable_parameters
from src.utils import get_device, load_json, save_json, seed_everything, count_parameters


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_model(model_name: str, manifest_train=MANIFEST_TRAIN, manifest_val=MANIFEST_VAL,
                 max_epochs: int = MAX_EPOCHS, patience: int = EARLY_STOPPING_PATIENCE,
                 tag: str = "", learning_rate: float = LEARNING_RATE,
                 init_checkpoint=None, unfreeze_all: bool = False) -> dict:
    seed_everything(SEED)
    device = get_device()
    classes = load_json(CLASS_LIST_PATH)
    num_classes = len(classes)

    train_ds = CropDiseaseDataset(manifest_train, train=True)
    val_ds = CropDiseaseDataset(manifest_val, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(model_name, num_classes).to(device)
    if init_checkpoint is not None:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device))
    if unfreeze_all:
        # Domain adaptation benefits from adapting low-level filters too, not
        # just the last few blocks used for plain transfer learning — the
        # domain gap (studio photos vs. real-world field photos) shows up in
        # low-level texture/color statistics, not just high-level features.
        for param in model.parameters():
            param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    run_name = f"{model_name}{('_' + tag) if tag else ''}"
    checkpoint_path = CHECKPOINTS_DIR / f"{run_name}.pt"

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{run_name}] epoch {epoch}/{max_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    training_time_sec = time.time() - start_time
    result = {
        "model_name": model_name,
        "run_name": run_name,
        "num_classes": num_classes,
        "num_trainable_params": count_parameters(model),
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "training_time_sec": training_time_sec,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(result, EXPERIMENTS_DIR / f"{run_name}_train.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    train_model(args.model, max_epochs=args.epochs)
