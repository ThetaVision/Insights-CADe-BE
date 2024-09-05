"""IMPORT PACKAGES"""
import torch.nn as nn

# Import helper functions from other files
from models.ResNet import ResNet50, ResNet101, ResNet152, DeepLabv3_plus
from models.ConvNeXt import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
)
from models.ConvNeXt import (
    DeepLabv3_plus_ConvNeXt_TS,
    DeepLabv3_plus_ConvNeXt_B,
    DeepLabv3_plus_ConvNeXt_L,
)
from models.FCBFormer import FCBFormer, pvt_v2_b2
from models.ESFPNet import ESFPNetStructure
from models.SwinUperNet import SwinUperNet, swin_v2_tiny
from models.TransNetR import TransNetR
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
from models.Kusters import EfficientNetB4


"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
"""""" """""" """""" """""" """""" """"""


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        # Define Backbone architecture
        if opt.backbone == 'ResNet-50-ImageNet':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = ResNet50(
                num_classes=opt.num_classes,
                channels=3,
                pretrained='ImageNet',
                url=url,
            )
        elif opt.backbone == 'ResNet-50-GastroNet':
            self.backbone = ResNet50(
                num_classes=opt.num_classes,
                channels=3,
                pretrained='GastroNet',
                url='',
            )
        elif opt.backbone == 'ResNet-101':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = ResNet101(
                num_classes=opt.num_classes,
                channels=3,
                pretrained='ImageNet',
                url=url,
            )
        elif opt.backbone == 'ResNet-152':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = ResNet152(
                num_classes=opt.num_classes,
                channels=3,
                pretrained='ImageNet',
                url=url,
            )
        elif opt.backbone == 'ConvNeXt-T':
            self.backbone = convnext_tiny(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-S':
            self.backbone = convnext_small(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-B':
            self.backbone = convnext_base(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-L':
            self.backbone = convnext_large(pretrained=True, num_classes=opt.num_classes)
        elif 'FCBFormer' in opt.backbone:
            self.backbone = FCBFormer(opt=opt)
        elif 'ESFPNet' in opt.backbone:
            self.backbone = ESFPNetStructure(opt=opt)
        elif 'Swin' in opt.backbone and 'UperNet' in opt.backbone:
            self.backbone = SwinUperNet(opt=opt)
        elif 'TransNetR' in opt.backbone:
            self.backbone = TransNetR(opt=opt, weights='ImageNet')
        elif 'MetaFormer' in opt.backbone:
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
        if opt.seg_branch == 'DeepLabV3p':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus(n_classes=1)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-TS':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_TS(n_classes=1)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-B':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_B(n_classes=1)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-L':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_L(n_classes=1)
        elif opt.seg_branch is None and 'UNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'FCBFormer' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'ESFPNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Swin' in opt.backbone and 'UperNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'TransNetR' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'MetaFormer' in opt.backbone:
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
            if opt.weights == 'ImageNet':
                url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
                self.backbone = ResNet50(
                    num_classes=opt.num_classes,
                    channels=3,
                    pretrained='ImageNet',
                    url=url,
                )
                self.multi_features = True
            elif opt.weights == 'GastroNet':
                self.backbone = ResNet50(
                    num_classes=opt.num_classes,
                    channels=3,
                    pretrained='GastroNet',
                    url='',
                )
                self.multi_features = True
            else:
                raise Exception(f'Unexpected Weights {opt.weights} for backbone {opt.backbone}')
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

        # Define SOTA BE characterization models
        elif opt.backbone == 'Putten':
            url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            self.backbone = GastroNet_ResNet18(
                n_channels=3,
                n_classes=opt.num_classes,
                mode='classification',
                url=url,
            )
        elif opt.backbone == 'Kusters':
            self.backbone = EfficientNetB4(
                n_classes=opt.num_classes,
                n_channels=1792,
                pretrained=True,
            )
        elif opt.backbone == 'Hussein':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = FCNResNet101(n_channels=3, n_classes=opt.num_classes, url=url, mode='classification')

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


def fps_check(opt, use_cuda=True):
    # Set parameters
    num_samples = 1000
    reps = 5

    # Test Device
    if use_cuda:
        device = check_cuda()

    # Construct Model and load weights
    model = Model(opt=opt)
    # weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    # model.load_state_dict(weights, strict=True)

    # Create random dummy input
    if use_cuda:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize).cuda()
    else:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize)

    # Push model to GPU
    if use_cuda:
        model.cuda()

    # Do model summary
    summary(model=model, input_size=(1, 3, opt.imagesize, opt.imagesize))

    # Set Model in evaluation mode and do fps check
    times = []
    model.eval()
    for j in range(reps):
        with torch.no_grad():
            starttime = time.time()
            for i in range(num_samples):
                cls_pred, seg_pred = model(dummy)
            stoptime = time.time()

        inference_time = stoptime - starttime
        avg_inference_sample = inference_time / num_samples
        avg_inference_fps = 1 / avg_inference_sample
        times.append(avg_inference_fps)

        print('Average Inference Time per sample: {} sec.'.format(avg_inference_sample))
        print('Average fps: {}'.format(avg_inference_fps))

    print('Average fps over {} reps: {}'.format(reps, sum(times) / len(times)))


if __name__ == "__main__":
    import torch
    import argparse
    from torchinfo import summary
    import time
    from train import check_cuda

    def get_params():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # DEFINE MODEL
        parser.add_argument('--backbone', type=str, default='MetaFormer-CAS18-UNetpp')
        parser.add_argument('--seg_branch', type=str, default=None)
        parser.add_argument('--weights', type=str, default='ImageNet')

        # AUGMENTATION PARAMS
        parser.add_argument('--imagesize', type=int, default=256)  # Should be 256, but for ViT 224
        parser.add_argument('--batchsize', type=int, default=8)
        parser.add_argument('--num_classes', type=int, default=1)

        args = parser.parse_args()

        return args

    opt = get_params()

    # model = Model(opt=opt).cuda()
    # model = Model_CLS(opt=opt)
    # summary(model, input_size=(1, 3, 256, 256))
    # dummy = torch.zeros([12, 3, 256, 256]).cuda()
    # cls = model(dummy)
    # print(cls.shape)
    # cls, seg = model(dummy)
    # print(cls.shape, seg.shape)
    fps_check(opt=opt, use_cuda=True)
