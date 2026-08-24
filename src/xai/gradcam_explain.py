import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.factory import GRADCAM_TARGET_LAYER


def _denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    image = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(std) + np.array(mean)
    return np.clip(image, 0, 1)


def explain_gradcam(model, model_name, image_tensor, mean, std, method="gradcam", target_class=None):
    target_layer = GRADCAM_TARGET_LAYER[model_name](model)
    cam_cls = GradCAMPlusPlus if method == "gradcam++" else GradCAM
    cam = cam_cls(model=model, target_layers=[target_layer])

    input_tensor = image_tensor.unsqueeze(0)
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    rgb_image = _denormalize(image_tensor, mean, std)
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return rgb_image, grayscale_cam, overlay
