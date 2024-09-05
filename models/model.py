"""IMPORT PACKAGES"""
import torch.nn as nn

# Import helper functions from other files
from models.ResNet import ResNet50
from models.ConvNeXt import convnext_tiny
from models.FCBFormer import pvt_v2_b2
from models.SwinUperNet import swin_v2_tiny
from models.MetaFormer import MetaFormerFPN, MetaFormerUperNet, MetaFormerDeepLabV3p, MetaFormerUNetpp
from models.MetaFormer import convformer_s18, caformer_s18
from models.UniFormer import uniformer_small_plus
from models.iFormer import iformer_small
from models.vision_transformer import vit_small

from models.Putten import GastroNet_ResNet18
from models.Hussein import FCNResNet101
from models.Ebigbo import ResNet101DeepLabV3
from models.Abdelrahim import VGG16SegNet
from models.Meinikheim import ResNet50DeepLabV3


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
        elif 'Putten' in opt.backbone:
            url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            self.backbone = GastroNet_ResNet18(
                n_channels=3,
                n_classes=opt.num_classes,
                mode='segmentation',
                url=url,
            )
        elif 'Hussein' in opt.backbone:
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = FCNResNet101(n_channels=3, n_classes=opt.num_classes, url=url, mode='segmentation')
        elif 'Ebigbo' in opt.backbone:
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = ResNet101DeepLabV3(n_channels=3, n_classes=opt.num_classes, url=url)
        elif 'Abdelrahim' in opt.backbone:
            url = "https://download.pytorch.org/models/vgg16-397923af.pth"
            self.backbone = VGG16SegNet(n_classes=opt.num_classes, pretrained=True, url=url)
        elif 'Meinikheim' in opt.backbone:
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = ResNet50DeepLabV3(n_channels=3, n_classes=opt.num_classes, url=url)
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))

        # Define segmentation branch architecture
        if opt.seg_branch is None and 'MetaFormer' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Putten' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Hussein' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Ebigbo' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Abdelrahim' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Meinikheim' in opt.backbone:
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

        # Define CNN-based backbone architectures
        if opt.backbone == 'ResNet-50':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = ResNet50(
                num_classes=opt.num_classes,
                channels=3,
                pretrained='ImageNet',
                url=url,
            )
            self.multi_features = True
        elif opt.backbone == 'ConvNeXt-T':
            self.backbone = convnext_tiny(pretrained=True, num_classes=opt.num_classes)
            self.multi_features = True
        elif opt.backbone == 'ConvFormer-S18':
            self.backbone = convformer_s18(opt, pretrained=True)
            self.features = True

        # Define hybrid backbone architectures
        elif opt.backbone == 'CaFormer-S18':
            self.backbone = caformer_s18(opt, pretrained=True)
            self.features = True
        elif opt.backbone == 'UniFormer-S+':
            self.backbone = uniformer_small_plus(opt, pretrained=True)
        elif opt.backbone == 'iFormer-S':
            self.backbone = iformer_small(opt, pretrained=True)

        # Define transformer-based backbone architectures
        elif opt.backbone == 'ViT-S':
            self.backbone = vit_small(opt, pretrained=True)
        elif opt.backbone == 'SwinV2-T':
            self.backbone = swin_v2_tiny(opt, pretrained=True)
        elif opt.backbone == 'PVTv2-S':
            self.backbone = pvt_v2_b2(opt, pretrained=True)

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
