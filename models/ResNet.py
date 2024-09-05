"""IMPORT PACKAGES"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch import Tensor
from typing import Callable, List, Optional, Type, Union


"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


# Function for 3x3 convolution
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


# Function for 1x1 convolution
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Class definition for BasicBlock
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Class definition for Bottleneck
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""""" """""" """""" """"""
"""" RESNET DEFINITION """
"""""" """""" """""" """"""


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        num_channels=3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        extra_fc_layers: bool = False,
        pretrained=None,
        url='',
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.extra_fc_layers = extra_fc_layers

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if extra_fc_layers:
            self.fc1 = nn.Linear(512 * block.expansion, 512)
            self.fc2 = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        # Define URL for pretrained weights
        self.url = url

        # Load pretrained weights if pretrained is True
        if pretrained:
            self._load_pretrained_model(pretrained)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_level_feat = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.extra_fc_layers:
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)

        return x, low_level_feat, high_level_feat

    def forward_features(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_level_feat = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, low_level_feat, high_level_feat

    def forward_features_all(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)

        return x0, x1, x2, x3, x

    def _load_pretrained_model(self, pretrained):
        # Define initialization
        if pretrained == 'ImageNet':
            pretrain_dict = model_zoo.load_url(self.url)
        elif pretrained == 'GastroNet':
            pretrain_dict = torch.load(
                os.path.join(
                    os.getcwd(),
                    '..',
                    'pretrained',
                    'checkpoint_200ep_teacher_adapted.pth',
                )
            )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# # Functions to create various different versions of ResNet
def ResNet18(
    num_classes, channels=3, pretrained=None, url='', replace_stride_with_dilation=None, extra_fc_layers=False
):
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        num_channels=channels,
        pretrained=pretrained,
        url=url,
        replace_stride_with_dilation=replace_stride_with_dilation,
        extra_fc_layers=extra_fc_layers,
    )


def ResNet34(
    num_classes, channels=3, pretrained=None, url='', replace_stride_with_dilation=None, extra_fc_layers=False
):
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        num_channels=channels,
        pretrained=pretrained,
        url=url,
        replace_stride_with_dilation=replace_stride_with_dilation,
        extra_fc_layers=extra_fc_layers,
    )


def ResNet50(
    num_classes, channels=3, pretrained=None, url='', replace_stride_with_dilation=None, extra_fc_layers=False
):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        num_channels=channels,
        pretrained=pretrained,
        url=url,
        replace_stride_with_dilation=replace_stride_with_dilation,
        extra_fc_layers=extra_fc_layers,
    )


def ResNet101(
    num_classes, channels=3, pretrained=None, url='', replace_stride_with_dilation=None, extra_fc_layers=False
):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        num_channels=channels,
        pretrained=pretrained,
        url=url,
        replace_stride_with_dilation=replace_stride_with_dilation,
        extra_fc_layers=extra_fc_layers,
    )


def ResNet152(
    num_classes, channels=3, pretrained=None, url='', replace_stride_with_dilation=None, extra_fc_layers=False
):
    return ResNet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        num_classes=num_classes,
        num_channels=channels,
        pretrained=pretrained,
        url=url,
        replace_stride_with_dilation=replace_stride_with_dilation,
        extra_fc_layers=extra_fc_layers,
    )


"""""" """""" """""" """""" """"""
"""" DEFINE SEGMENTATION MODELS"""
"""""" """""" """""" """""" """"""
# https://github.com/MLearing/Pytorch-DeepLab-v3-plus


# Class for ASPP Module
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=padding,
            dilation=rate,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, n_classes=1, os=16):
        super(DeepLabv3_plus, self).__init__()

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48, track_running_stats=True)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                304,
                256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(
                256,
                256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)),
        )

        # Apply initialization of weights
        self._init_weight()

    def forward(self, img, low_level_features, high_level_features):
        x1 = self.aspp1(high_level_features)
        x2 = self.aspp2(high_level_features)
        x3 = self.aspp3(high_level_features)
        x4 = self.aspp4(high_level_features)
        x5 = self.global_avg_pool(high_level_features)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.interpolate(
            x,
            size=(
                int(math.ceil(img.size()[-2] / 4)),
                int(math.ceil(img.size()[-1] / 4)),
            ),
            mode='bilinear',
            align_corners=True,
        )

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
