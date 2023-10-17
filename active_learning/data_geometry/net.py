from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import model_urls
from torch.utils import model_zoo


def _load_pretrained(model, url, inchans=3):
    state_dict = model_zoo.load_url(url)
    if inchans == 1:
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif inchans != 3:
        assert False, "Invalid number of inchans for pretrained weights"
    model.load_state_dict(state_dict)


def resnet50(pretrained=False, inchans=3):
    model = resnet50(pretrained=pretrained)
    if pretrained:
        _load_pretrained(model, model_urls['resnet50'], inchans=inchans)
    return model


def resnet18(pretrained=False, inchans=3):
    model = resnet18(pretrained=pretrained)
    if pretrained:
        _load_pretrained(model, model_urls['resnet18'], inchans=inchans)
    return model