"""
ResNet-101 + DeepLabV3+ network implementation for the paper:
- Ebigbo et al. - Computer-aided diagnosis using deep learning in the evaluation of early oesophageal adenocarcinoma
                  (Gut, 2019)
                  https://gut.bmj.com/content/68/7/1143
- Ebigbo et al. - Real-time use of artificial intelligence in the evaluation of cancer in Barrett's oesophagus
                  (Gut, 2020)
                  https://gut.bmj.com/content/69/4/615
"""
import torch.nn as nn
from models.ResNet import ResNet101, DeepLabv3_plus


"""""" """""" """""" """"""
"""" FCN DEFINITION """
"""""" """""" """""" """"""


class ResNet101DeepLabV3(nn.Module):
    def __init__(self, n_channels, n_classes, url=''):
        super(ResNet101DeepLabV3, self).__init__()

        # Define ResNet-101 architecture
        self.encoder = ResNet101(
            num_classes=n_classes,
            channels=n_channels,
            replace_stride_with_dilation=[False, False, True],  # Last layer uses concept of dilated convolutions
            extra_fc_layers=True,
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
