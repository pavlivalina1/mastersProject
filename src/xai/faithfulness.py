import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _trapezoid_auc(values, dx: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(dx * (values.sum() - 0.5 * (values[0] + values[-1])))


def _ranked_pixel_order(saliency: np.ndarray) -> np.ndarray:
    flat = saliency.flatten()
    return np.argsort(-flat)


def _blurred_baseline(image_chw: torch.Tensor) -> torch.Tensor:
    image_np = image_chw.cpu().numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(image_np, (31, 31), 0)
    return torch.from_numpy(blurred.transpose(2, 0, 1)).to(image_chw.device)


@torch.no_grad()
def insertion_deletion_auc(model, image_chw: torch.Tensor, saliency: np.ndarray,
                            target_class: int, device, steps: int = 20):
    model.eval()
    h, w = saliency.shape
    order = _ranked_pixel_order(saliency)
    n_pixels = h * w
    step_size = max(1, n_pixels // steps)

    baseline = _blurred_baseline(image_chw)
    original = image_chw.to(device)

    deletion_scores = []
    insertion_scores = []

    deleted = original.clone()
    inserted = baseline.clone()

    mask_flat_shape = (h, w)
    for step in range(steps + 1):
        n_reveal = min(step * step_size, n_pixels)
        idx = order[:n_reveal]
        mask = np.zeros(mask_flat_shape, dtype=np.float32).flatten()
        mask[idx] = 1.0
        mask = torch.from_numpy(mask.reshape(mask_flat_shape)).to(device)

        deleted_img = original * (1 - mask) + baseline * mask
        inserted_img = baseline * (1 - mask) + original * mask

        del_logits = model(deleted_img.unsqueeze(0))
        ins_logits = model(inserted_img.unsqueeze(0))
        del_prob = F.softmax(del_logits, dim=1)[0, target_class].item()
        ins_prob = F.softmax(ins_logits, dim=1)[0, target_class].item()

        deletion_scores.append(del_prob)
        insertion_scores.append(ins_prob)

    deletion_auc = _trapezoid_auc(deletion_scores, dx=1.0 / steps)
    insertion_auc = _trapezoid_auc(insertion_scores, dx=1.0 / steps)
    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
    }
