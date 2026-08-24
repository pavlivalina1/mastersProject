import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    _freeze_backbone(model)
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    _freeze_backbone(model)
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    _freeze_backbone(model)
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_shufflenet_v2(num_classes: int) -> nn.Module:
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    _freeze_backbone(model)
    for param in model.conv5.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "custom_cnn": lambda num_classes: CustomCNN(num_classes),
    "resnet18": build_resnet18,
    "mobilenet_v2": build_mobilenet_v2,
    "efficientnet_b0": build_efficientnet_b0,
    "shufflenet_v2": build_shufflenet_v2,
}

GRADCAM_TARGET_LAYER = {
    "custom_cnn": lambda model: model.features[-2],
    "resnet18": lambda model: model.layer4[-1],
    "mobilenet_v2": lambda model: model.features[-1],
    "efficientnet_b0": lambda model: model.features[-1],
    "shufflenet_v2": lambda model: model.conv5,
}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model name: {name}. Choices: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_classes)


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)
