import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import numpy as np

NUM_FINETUNE_CLASSES = 219  # classes : 0 ~ 218

#
class beit_model(nn.Module):
    def __init__(self, classes=219):
        super(beit_model, self).__init__()
        self.model = timm.create_model('beit_base_patch16_384', pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)
    
class convnext_model(nn.Module):
    def __init__(self, classes=219):
        super(convnext_model, self).__init__()
        self.model = timm.create_model('convnext_base_384_in22ft1k', pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)

class swin_model(nn.Module):
    def __init__(self, classes=219):
        super(swin_model, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)

class vit_model(nn.Module):
    def __init__(self, classes=219):
        super(vit_model, self).__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    # input
    x = torch.randn(1, 3, 384, 384)

    # beit
    model_1 = timm.create_model('beit_base_patch16_384', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    # convext
    model_2 = timm.create_model('convnext_base_384_in22ft1k', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    # swin
    model_3 = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    # vit
    model_4 = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    print(model_1(x).shape)
