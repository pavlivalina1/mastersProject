import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.config import IMAGENET_MEAN, IMAGENET_STD


def _make_classifier_fn(model, device):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def classifier_fn(images_uint8: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(images_uint8).float().permute(0, 3, 1, 2) / 255.0
        batch = (batch - mean) / std
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return classifier_fn


def explain_lime(model, rgb_float_image: np.ndarray, device, num_samples=500, num_features=8):
    image_uint8 = (rgb_float_image * 255).astype(np.uint8)
    classifier_fn = _make_classifier_fn(model, device)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_uint8,
        classifier_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=num_features, hide_rest=False,
    )
    overlay = mark_boundaries(temp / 255.0, mask)
    return overlay, explanation, top_label
