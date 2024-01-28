import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.models as models

def get_pretrained_vgg(device):
    vgg = models.vgg19(pretrained=True)
    return vgg.to(device)

def get_vgg_feature_extractor(device):
    vgg = get_pretrained_vgg(device)

    nodes = {
        "features.1": "convblock1",
        "features.6": "convblock2",
        "features.13": "convblock3",
        "features.22": "convblock4",
        "features.29": "convblock5",
        "features.35": "contentblock",
    }

    layers = ["convblock1", "convblock2", "convblock3", "convblock4", "convblock5", "contentblock"]
    return create_feature_extractor(vgg, return_nodes=nodes), layers