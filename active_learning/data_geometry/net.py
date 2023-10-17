import torch.nn as nn
from torchvision.models import resnet18, resnet50


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, inchans=3):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        if inchans == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, inchans=3):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        if inchans == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        return x


def resnet50(pretrained=False, inchans=3):
    model = ResNet18(pretrained=pretrained, inchans=inchans)
    return model


def resnet18(pretrained=False, inchans=3):
    model = ResNet50(pretrained=pretrained, inchans=inchans)
    return model