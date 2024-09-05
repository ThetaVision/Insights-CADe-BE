"""IMPORT PACKAGES"""
import torch.nn as nn
from models.VisionMamba import vim_small, vim_small_stride8, vim_tiny, vim_tiny_stride8

"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
"""""" """""" """""" """""" """""" """"""

"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS ONLY MODEL"""
"""""" """""" """""" """""" """""" """"""


class Model_CLS(nn.Module):
    def __init__(self, opt):
        super(Model_CLS, self).__init__()

        # Define the variations that can occur in forward
        self.multi_features = False
        self.features = False

        # Define SSM-based backbone architectures
        if opt.backbone == 'ViM-T':
            self.backbone = vim_tiny(opt, pretrained=True)
        elif opt.backbone == 'ViM-T-S8':
            self.backbone = vim_tiny_stride8(opt, pretrained=True)
        elif opt.backbone == 'ViM-S':
            self.backbone = vim_small(opt, pretrained=True)
        elif opt.backbone == 'ViM-S-S8':
            self.backbone = vim_small_stride8(opt, pretrained=True)

        # Define Exception for unexpected backbones
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))

    def forward(self, img):
        if self.multi_features:
            # Output of model contains multiple features
            cls, _, _ = self.backbone(img)
        elif self.features:
            # Output of model contains features
            cls, _ = self.backbone(img)
        else:
            # Output of model only contains classification
            cls = self.backbone(img)

        return cls
