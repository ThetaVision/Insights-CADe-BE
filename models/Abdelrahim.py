"""
VGG-16 + SegNet network implementation for the paper:
Abdelrahim et al. - Development and validation of artificial neural networks for detection of Barrett's neoplasia:
                    a multicenter pragmatic nonrandomized trial (GIE, 2023)
                    https://www.sciencedirect.com/science/article/pii/S0016510722020843
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from typing import cast, Dict, List, Union


"""""" """""" """""" """"""
"""" VGG16-SEGNET DEFINITION """
"""""" """""" """""" """"""


class VGG16SegNet(nn.Module):
    def __init__(self, n_classes, pretrained=False, url=''):
        super(VGG16SegNet, self).__init__()

        # Define VGG-16 architecture
        self.encoder = vgg16(n_classes=n_classes, pretrained=pretrained, url=url)

        # Define SegNet Decoder architecture
        self.decoder = SegNetDecoder(n_classes=n_classes)

    def forward(self, x):
        # Forward pass through VGG-16
        cls, feat, indices, out_sizes = self.encoder(x)

        # Forward pass through SegNet Decoder
        seg = self.decoder(feat, indices, out_sizes)

        return cls, seg


"""""" """""" """""" """"""
"""" VGG16 DEFINITION """
"""""" """""" """""" """"""
# Partly adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D_custom": [[64, 64, "M"], [128, 128, "M"], [256, 256, 256, "M"], [512, 512, 512, "M"], [512, 512, 512, "M"]],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=1000, init_weights=True, dropout=0.5, pretrained=False, url=''):
        super().__init__()
        self.block1 = make_layers(cfg[0], batch_norm=False, in_channels=3)
        self.block2 = make_layers(cfg[1], batch_norm=False, in_channels=64)
        self.block3 = make_layers(cfg[2], batch_norm=False, in_channels=128)
        self.block4 = make_layers(cfg[3], batch_norm=False, in_channels=256)
        self.block5 = make_layers(cfg[4], batch_norm=False, in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        self.url = url
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        size_1 = x.size()
        x, ind1 = self.block1(x)
        size_2 = x.size()
        x, ind2 = self.block2(x)
        size_3 = x.size()
        x, ind3 = self.block3(x)
        size_4 = x.size()
        x, ind4 = self.block4(x)
        size_5 = x.size()
        x, ind5 = self.block5(x)
        cls = self.avgpool(x)
        cls = torch.flatten(cls, 1)
        cls = self.classifier(cls)
        return cls, x, [ind1, ind2, ind3, ind4, ind5], [size_1, size_2, size_3, size_4, size_5]

    def _load_pretrained_model(self):
        # Extract the keys from the pre-trained model
        pretrain_dict = model_zoo.load_url(self.url)
        pretrain_keys = list(pretrain_dict.keys())

        # Extract the keys from the current model
        model_keys = list(self.state_dict().keys())

        # Replace the keys in the pre-trained model with the keys in the current model
        for i in range(len(pretrain_keys)):
            if 'features' in pretrain_keys[i]:
                pretrain_dict[model_keys[i]] = pretrain_dict[pretrain_keys[i]]
                del pretrain_dict[pretrain_keys[i]]

        # Update the state dictionary of the current model
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'classifier.6.weight' not in k and 'classifier.6.bias' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def make_layers(cfg, in_channels, batch_norm=False):
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(n_classes, pretrained=False, url='', **kwargs):
    return VGG(cfg=cfgs["D_custom"], num_classes=n_classes, pretrained=pretrained, url=url)


"""""" """""" """""" """"""
"""" SEGNET DEFINITION """
"""""" """""" """""" """"""
# Partly adapted from:
# https://github.com/alejandrodebus/SegNet/blob/master/Segmentation%20-%20Endocardium%20and%20Epicardium.ipynb


class SegNetDecoder(nn.Module):
    def __init__(self, n_classes=1):
        super(SegNetDecoder, self).__init__()

        # Unpooling layers
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        # Deconvolution layers
        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(64, n_classes, kernel_size=3, stride=1, padding=1)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, indices, out_sizes):
        # Up-sampling block 5
        x = self.unpool5(x, indices[4], output_size=out_sizes[4])
        x = self.deconv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)

        # Up-sampling block 4
        x = self.unpool4(x, indices[3], output_size=out_sizes[3])
        x = self.deconv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        # Up-sampling block 3
        x = self.unpool3(x, indices[2], output_size=out_sizes[2])
        x = self.deconv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_3(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        # Up-sampling block 2
        x = self.unpool2(x, indices[1], output_size=out_sizes[1])
        x = self.deconv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        # Up-sampling block 1
        x = self.unpool1(x, indices[0], output_size=out_sizes[0])
        x = self.deconv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1_2(x)

        return x
