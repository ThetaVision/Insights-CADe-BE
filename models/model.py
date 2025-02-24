"""IMPORT PACKAGES"""
import torch.nn as nn

# Import helper functions from other files
from models.MetaFormer import MetaFormerFPN, MetaFormerUperNet, MetaFormerDeepLabV3p, MetaFormerUNetpp
from models.MetaFormer import convformer_s18, caformer_s18



"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
"""""" """""" """""" """""" """""" """"""


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        # Define Backbone architecture
        if 'MetaFormer' in opt.backbone:
            if 'FPN' in opt.backbone:
                self.backbone = MetaFormerFPN(opt=opt)
            elif 'UperNet' in opt.backbone:
                self.backbone = MetaFormerUperNet(opt=opt)
            elif 'DeepLabV3p' in opt.backbone:
                self.backbone = MetaFormerDeepLabV3p(opt=opt)
            elif 'UNetpp' in opt.backbone:
                self.backbone = MetaFormerUNetpp(opt=opt)
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))

        # Define segmentation branch architecture
        if opt.seg_branch is None and 'MetaFormer' in opt.backbone:
            self.single_model = True
        else:
            raise Exception('Unexpected Segmentation Branch {}'.format(opt.seg_branch))

    def forward(self, img):
        if self.single_model:
            # Output of single model
            cls, seg = self.backbone(img)

        else:
            # Backbone output
            cls, low_level, high_level = self.backbone(img)

            # Segmentation output
            seg = self.seg_branch(img, low_level, high_level)

        return cls, seg


"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS ONLY MODEL"""
"""""" """""" """""" """""" """""" """"""


class Model_CLS(nn.Module):
    def __init__(self, opt):
        super(Model_CLS, self).__init__()

        # Define the variations that can occur in forward
        self.multi_features = False
        self.features = False

        # Define hybrid backbone architectures
        if opt.backbone == 'CaFormer-S18':
            self.backbone = caformer_s18(opt, pretrained=True)
            self.features = True
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
