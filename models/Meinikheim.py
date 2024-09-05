"""
ResNet-50 + DeepLabV3+ network implementation for the paper:
- Meinikheim et al. - Effect of AI on performance of endoscopists to detect Barrett neoplasia: A Randomized Tandem Trial
                     (Endoscopy, 2024)
                     https://www.thieme-connect.com/products/ejournals/abstract/10.1055/a-2296-5696
"""
import torch.nn as nn
from models.ResNet import ResNet50, DeepLabv3_plus


"""""" """""" """""" """"""
"""" MODEL DEFINITION """
"""""" """""" """""" """"""


class ResNet50DeepLabV3(nn.Module):
    def __init__(self, n_channels, n_classes, url=''):
        super(ResNet50DeepLabV3, self).__init__()

        # Define ResNet-101 architecture
        self.encoder = ResNet50(
            num_classes=n_classes,
            channels=n_channels,
            pretrained='ImageNet',
            url=url,
        )

        # Define DeepLabV3+ Head architecture
        self.decoder = DeepLabv3_plus(n_classes=n_classes)

    def forward(self, x):
        # Forward pass through ResNet-101
        cls, ll_feat, hl_feat = self.encoder(x)

        # Forward pass through DeepLabV3+ Head
        seg = self.decoder(x, ll_feat, hl_feat)

        return cls, seg
