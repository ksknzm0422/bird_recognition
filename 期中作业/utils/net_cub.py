from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
from utils.weight_init import weight_init_kaiming


def init_resnet50(n_class=200, pretrained=True):
    if pretrained:
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    else:
        resnet = resnet50(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    return resnet


