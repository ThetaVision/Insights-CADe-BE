"""
ResNet-101 + FCN (Fully Convolutional Network) network implementation for the CADe paper:
Hussein et al. - A new artificial intelligence system successfully detects and localises early neoplasia in Barrett's
                 esophagus by using convolutional neural networks (UEG Journal, 2022)
                 https://onlinelibrary.wiley.com/doi/full/10.1002/ueg2.12233

ResNet-101 network implementation for the CADx paper:
Hussein et al. - Computer-aided characterization of early cancer in Barrett's esophagus on i-scan magnification imaging:
                 a multicenter international study (Gastrointestinal Endoscopy, 2023)
                 https://www.giejournal.org/article/S0016-5107(22)02809-1/fulltext

"""
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import ResNet101


"""""" """""" """""" """"""
"""" FCN DEFINITION """
"""""" """""" """""" """"""
# Partly adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class FCNResNet101(nn.Module):
    def __init__(self, n_channels, n_classes, url='', mode='segmentation'):
        super(FCNResNet101, self).__init__()

        # Define ResNet-101 architecture
        self.encoder = ResNet101(
            num_classes=n_classes,
            channels=n_channels,
            pretrained='ImageNet',
            url=url,
            replace_stride_with_dilation=[False, True, True],
        )

        self.mode = mode
        if mode == 'segmentation':
            # Define FCN Head architecture
            self.head = FCNHead(in_channels=2048, channels=n_classes)

    def forward(self, x):
        # Find input shape
        input_shape = x.shape[-2:]

        # Forward pass through ResNet-101
        cls, _, feat = self.encoder(x)

        if self.mode == 'classification':
            return cls
        elif self.mode == 'segmentation':
            # Forward pass through FCN Head
            seg = self.head(feat)

            # Perform up-sampling to match input image size
            seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=False)

            return cls, seg

        else:
            print('unrecognized mode type, only "classification" and "segmentation" are supported')
