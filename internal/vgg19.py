import torch
from torchvision import models
from collections import namedtuple

def gram_matrix(x):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features

        offset = 1

        self.slice1 = vgg_pretrained_features[0:1 + offset]
        self.slice2 = vgg_pretrained_features[1 + offset:6 + offset]
        self.slice3 = vgg_pretrained_features[6 + offset:11 + offset]
        self.slice4 = vgg_pretrained_features[11 + offset:20 + offset]
        self.slice5 = vgg_pretrained_features[20 + offset:22 + offset]
        self.slice6 = vgg_pretrained_features[22 + offset:29 + offset]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x

        content_part = conv4_2
        style_part = [gram_matrix(x) for x in (layer1_1, layer2_1, layer3_1, layer4_1, layer5_1)]
        return content_part, style_part
