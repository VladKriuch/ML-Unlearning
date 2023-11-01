import torch
import requests
import os

from torchvision.models import resnet18


def get_model(pretrained=False, device=None):
    # Returns pretrained resnet18 model
    # download pre-trained weights
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)

    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_pretrained = torch.load(local_path, map_location=torch.device(device))

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)

    return model