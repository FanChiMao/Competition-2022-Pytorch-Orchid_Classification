import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
import numpy as np
from torchvision import transforms

NUM_FINETUNE_CLASSES = 219  # classes : 0 ~ 218


#
class beit_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(beit_model, self).__init__()
        self.model = timm.create_model('beit_base_patch16_384', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class convnext_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(convnext_model, self).__init__()
        self.model = timm.create_model('convnext_base_384_in22ft1k', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class swin_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(swin_model, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class vit_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(vit_model, self).__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class volo_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(volo_model, self).__init__()
        self.model = timm.create_model('volo_d2_384', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class resmlp_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(resmlp_model, self).__init__()
        self.model = timm.create_model('resmlp_big_24_224_in22ft1k', pretrained=pretrained, num_classes=classes)


class xcittiny_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(xcittiny_model, self).__init__()
        self.model = timm.create_model('xcit_tiny_12_p8_384_dist', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class ecaresnet269_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(ecaresnet269_model, self).__init__()
        self.model = timm.create_model('ecaresnet269d', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class dmnfnet_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(dmnfnet_model, self).__init__()
        self.model = timm.create_model('dm_nfnet_f0', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class ecaresnet50_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(ecaresnet50_model, self).__init__()
        self.model = timm.create_model('ecaresnet50t', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class regnetz_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(regnetz_model, self).__init__()
        self.model = timm.create_model('regnetz_e8', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class efficientnet_model(nn.Module):
    def __init__(self, classes=219, pretrained=True):
        super(efficientnet_model, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=pretrained, num_classes=classes)

    def forward(self, x):
        return self.model(x)


class traditional_EnsembleModel(nn.Module):
    def __init__(self, model_list: dict, hidden_layer=512, output_layer=219, act=nn.ReLU()):
        super(traditional_EnsembleModel, self).__init__()

        # freeze all layers in all models
        for each_model in model_list:
            (model, transformation, size) = each_model
            for param in model.parameters():
                param.requires_grad_(False)

        self.model_list = model_list

        # traditional MLP
        self.ensemble_mlp = nn.Sequential(
            nn.Linear(len(model_list) * 219, hidden_layer),
            act,
            nn.Linear(hidden_layer, output_layer),
        )

    def forward(self, x_224, x_256, x_320, x_352, x_384, x_480):
        result = torch.empty((0)).cuda()
        # each preds
        for i, (each_model, transformation, size) in enumerate(self.model_list):
            if size == 224:
                pred = each_model(x_224)
            elif size == 256:
                pred = each_model(x_256)
            elif size == 320:
                pred = each_model(x_320)
            elif size == 352:
                pred = each_model(x_352)
            elif size == 384:
                pred = each_model(x_384)
            elif size == 480:
                pred = each_model(x_480)
            result = torch.cat((result, pred), dim=1)

        x = self.ensemble_mlp(result)

        return x


def transform_size(size: int):
    # mean & std for different sizes
    mean = {224: (0.5446, 0.4137, 0.3847),
            256: (0.5364, 0.4142, 0.3821),
            320: (0.5188, 0.4166, 0.3773),
            352: (0.5100, 0.4183, 0.3750),
            384: (0.5015, 0.4198, 0.3728),
            480: (0.4806, 0.4232, 0.3675)}

    std = {224: (0.2329, 0.2484, 0.2500),
           256: (0.2354, 0.2470, 0.2490),
           320: (0.2403, 0.2442, 0.2479),
           352: (0.2423, 0.2431, 0.2479),
           384: (0.2440, 0.2424, 0.2481),
           480: (0.2478, 0.2423, 0.2500)}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=480, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=mean[size], std=std[size])
    ])
    return transform


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)


def build_model(model: str, pretrained: bool = True):
    models = ['vit', 'beit', 'swin', 'convnext', 'volo', 'resmlp', 'xcit', 'ecaresnet50', 'ecaresnet269',
              'dmnfnet', 'regnet', 'efficient']
    if model == 'vit':
        return vit_model(pretrained=pretrained)
    elif model == 'beit':
        return beit_model(pretrained=pretrained)
    elif model == 'swin':
        return swin_model(pretrained=pretrained)
    elif model == 'convnext':
        return convnext_model(pretrained=pretrained)
    elif model == 'volo':
        return volo_model(pretrained=pretrained)
    elif model == 'resmlp':
        return resmlp_model(pretrained=pretrained)
    elif model == 'xcit':
        return xcittiny_model(pretrained=pretrained)
    elif model == 'ecaresnet50':
        return ecaresnet50_model(pretrained=pretrained)
    elif model == 'ecaresnet269':
        return ecaresnet269_model(pretrained=pretrained)
    elif model == 'dmnfnet':
        return dmnfnet_model(pretrained=pretrained)
    elif model == 'regnet':
        return regnetz_model(pretrained=pretrained)
    elif model == 'efficient':
        return efficientnet_model(pretrained=pretrained)
    else:
        raise Exception(
            "\nNo corresponding model! \nPlease enter the supported model: \n\n{}".format('\n'.join(models)))


def build_ensemble_model(model: dict, pretrained: bool):
    """
    Args:
        model:
        {
        CLASSIFIER1: ['vit', 384, 'pretrained/vit_testmodel.pth'],
        CLASSIFIER2: ['beit', 384, 'pretrained/beit_testmodel.pth'],
        CLASSIFIER3: ['swin', 384, 'pretrained/swin_testmodel.pth'],
        CLASSIFIER4: ['convnext', 384, 'pretrained/convnext_fold1_best_acc.pth']
        }

    Returns: [[finish loading pretrained model, corresponding transform], ...]

    """

    print('==> Build and load the ensemble models')
    ensemble_model = []
    for i, key in enumerate(tqdm(model)):
        #print(model[key][0])
        value = model[key]
        each_model = build_model(model=value[0], pretrained=pretrained).cuda()
        load_checkpoint(each_model, value[2])
        each_model.eval()
        ensemble_model.append([each_model, transform_size(value[1]), value[1]])

    return ensemble_model


if __name__ == "__main__":
    # input
    x = torch.randn(1, 3, 384, 384)

    # # beit
    # model_1 = timm.create_model('beit_base_patch16_384', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
    #
    # # convext
    # model_2 = timm.create_model('convnext_base_384_in22ft1k', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
    #
    # # swin
    # model_3 = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)

    # vit
    model_4 = timm.create_model('resmlp_big_24_224_in22ft1k', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
