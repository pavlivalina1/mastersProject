import numpy as np
import shap
import torch


def explain_shap(model, background_tensors: torch.Tensor, image_tensor: torch.Tensor, target_class=None):
    model.eval()
    explainer = shap.GradientExplainer(model, background_tensors)
    input_batch = image_tensor.unsqueeze(0)
    shap_values = explainer.shap_values(input_batch, nsamples=50)

    if isinstance(shap_values, list):
        class_idx = target_class if target_class is not None else 0
        values = shap_values[class_idx][0]
    else:
        class_idx = target_class if target_class is not None else 0
        values = shap_values[0, ..., class_idx]

    saliency = np.abs(values).sum(axis=0)
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency
