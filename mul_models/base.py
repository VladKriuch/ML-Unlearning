import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


class UnlearnAbstract:
    def __init__(self, model, DEVICE='cpu'):
        self.__model = model
        self.__device = DEVICE

    def unlearn(self, retain_set, forget_set, validation=None):
        pass