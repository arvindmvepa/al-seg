import torch.nn as nn
from torchvision.models import resnet18 as resnet18_
from torchvision.models import resnet50 as resnet50_


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, inchans=3):
        super().__init__()
        self.resnet = resnet18_(pretrained=pretrained)
        if inchans == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = Identity(self.resnet.fc.in_features)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, inchans=3):
        super().__init__()
        self.resnet = resnet50_(pretrained=pretrained)
        if inchans == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = Identity(self.resnet.fc.in_features)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Identity(nn.Module):
    def __init__(self, in_features=None):
        super(Identity, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        return x


def resnet18(pretrained=False, inchans=3):
    model = ResNet18(pretrained=pretrained, inchans=inchans)
    return model


def resnet50(pretrained=False, inchans=3):
    model = ResNet50(pretrained=pretrained, inchans=inchans)
    return model