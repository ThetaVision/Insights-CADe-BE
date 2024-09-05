# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
import os
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

"""""" """""" """""" """""" """""" """"""
"""" DEFINE METAFORMER-FPN MODEL"""
"""""" """""" """""" """""" """""" """"""


class MetaFormerFPN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Implement and initialize backbone (Identity)
        if "identity" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = identityformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = identityformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = identityformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = identityformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "identity" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = identityformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "rand" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = randformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = randformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = randformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = randformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "rand" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = randformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Pool)
        elif "pool" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = poolformerv2_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = poolformerv2_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = poolformerv2_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = poolformerv2_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "pool" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = poolformerv2_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "conv" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "waveletdown" in opt.backbone.lower():
                self.metaformer = convformer_waveletdown_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            elif "waveletfuse" in opt.backbone.lower():
                self.metaformer = convformer_waveletfuse_s18(opt=opt, pretrained=True)
                # feature_channels = (73, 137, 329, 521)
                feature_channels = (76, 140, 332, 524)
            else:
                self.metaformer = convformer_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = convformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = convformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "conv" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = convformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Implement and initialize backbone (Random)
        elif "ca" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "swin" in opt.backbone.lower():
                self.metaformer = caformer_s18_swin(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            elif "pvt" in opt.backbone.lower():
                self.metaformer = caformer_s18_pvt(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            else:
                if "waveletdown" in opt.backbone.lower():
                    self.metaformer = caformer_waveletdown_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)
                elif "waveletfuse" in opt.backbone.lower():
                    self.metaformer = caformer_waveletfuse_s18(opt=opt, pretrained=True)
                    # feature_channels = (73, 137, 329, 521)
                    feature_channels = (76, 140, 332, 524)
                else:
                    self.metaformer = caformer_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = caformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = caformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "ca" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = caformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Implement and initialize backbone (DWT)
        elif "dwt" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "cdwt" in opt.backbone.lower():
                if "waveletfuse" in opt.backbone.lower():
                    self.metaformer = cdwtformer_waveletfuse_s18(opt=opt, pretrained=True)
                    # feature_channels = (73, 137, 329, 521)
                    feature_channels = (76, 140, 332, 524)
                else:
                    self.metaformer = cdwtformer_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)
            elif "dwtc" in opt.backbone.lower():
                if "waveletfuse" in opt.backbone.lower():
                    self.metaformer = dwtcformer_waveletfuse_s18(opt=opt, pretrained=True)
                    # feature_channels = (73, 137, 329, 521)
                    feature_channels = (76, 140, 332, 524)
                else:
                    self.metaformer = dwtcformer_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)
            elif "dwta" in opt.backbone.lower():
                if "waveletfuse" in opt.backbone.lower():
                    self.metaformer = dwtaformer_waveletfuse_s18(opt=opt, pretrained=True)
                    # feature_channels = (73, 137, 329, 521)
                    feature_channels = (76, 140, 332, 524)
                else:
                    self.metaformer = dwtaformer_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)
            else:
                if "waveletfuse" in opt.backbone.lower():
                    self.metaformer = dwtformer_waveletfuse_s18(opt=opt, pretrained=True)
                    # feature_channels = (73, 137, 329, 521)
                    feature_channels = (76, 140, 332, 524)
                else:
                    self.metaformer = dwtformer_s18(opt=opt, pretrained=True)
                    feature_channels = (64, 128, 320, 512)

        # Define Exception
        else:
            raise Exception("Unrecognized MetaFormer version...")

        # Define FPN Decoder
        self.FPN = FPN(
            encoder_channels=feature_channels,
            encoder_depth=3,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.0,
            merge_policy="cat",
            num_classes=opt.num_classes,
            interpolation=4,
        )

    def forward(self, x):
        # Produce encoder output
        cls, features = self.metaformer(x)

        # Produce decoder output
        seg = self.FPN(*features)

        return cls, seg

    def forward_features_list(self, x):
        # Produce encoder output
        x, features = self.metaformer.forward_features(x)

        return features


"""""" """""" """""" """""" """""" """"""
"""" DEFINE METAFORMER-UPERNET MODEL"""
"""""" """""" """""" """""" """""" """"""


class MetaFormerUperNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Implement and initialize backbone (Identity)
        if "identity" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = identityformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = identityformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = identityformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = identityformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "identity" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = identityformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "rand" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = randformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = randformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = randformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = randformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "rand" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = randformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Pool)
        elif "pool" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = poolformerv2_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = poolformerv2_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = poolformerv2_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = poolformerv2_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "pool" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = poolformerv2_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "conv" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "waveletdown" in opt.backbone.lower():
                self.metaformer = convformer_waveletdown_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            elif "waveletfuse" in opt.backbone.lower():
                self.metaformer = convformer_waveletfuse_s18(opt=opt, pretrained=True)
                # feature_channels = (73, 137, 329, 521)
                feature_channels = (76, 140, 332, 524)
            else:
                self.metaformer = convformer_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = convformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = convformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "conv" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = convformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Implement and initialize backbone (Random)
        elif "ca" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            self.metaformer = caformer_s18(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = caformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = caformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "ca" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = caformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Define Exception
        else:
            raise Exception("Unrecognized MetaFormer version...")

        # Define UperNet blocks
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], opt.num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Find input size
        input_size = (x.size()[2], x.size()[3])

        # Produce encoder output
        cls, features = self.metaformer(x)

        # UperNet
        features[-1] = self.PPN(features[-1])
        features_seg = self.FPN(features)
        features_seg = self.head(features_seg)

        # Segmentation
        seg = F.interpolate(features_seg, size=input_size, mode='bilinear')

        return cls, seg

    def forward_features_list(self, x):
        # Produce encoder output
        x, features = self.metaformer.forward_features(x)

        return features


"""""" """""" """""" """""" """""" """"""
"""" DEFINE METAFORMER-DEEPLABV3+ MODEL"""
"""""" """""" """""" """""" """""" """"""


class MetaFormerDeepLabV3p(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Implement and initialize backbone (Identity)
        if "identity" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = identityformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = identityformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = identityformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = identityformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "identity" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = identityformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "rand" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = randformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = randformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = randformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = randformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "rand" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = randformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Pool)
        elif "pool" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = poolformerv2_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = poolformerv2_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = poolformerv2_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = poolformerv2_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "pool" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = poolformerv2_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "conv" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "waveletdown" in opt.backbone.lower():
                self.metaformer = convformer_waveletdown_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            elif "waveletfuse" in opt.backbone.lower():
                self.metaformer = convformer_waveletfuse_s18(opt=opt, pretrained=True)
                # feature_channels = (73, 137, 329, 521)
                feature_channels = (76, 140, 332, 524)
            else:
                self.metaformer = convformer_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = convformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = convformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "conv" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = convformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Implement and initialize backbone (Random)
        elif "ca" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            self.metaformer = caformer_s18(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = caformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = caformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "ca" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = caformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Define Exception
        else:
            raise Exception("Unrecognized MetaFormer version...")

        # Define DeepLabV3+ Decoder
        self.deeplabv3p = DeepLabv3_plus(
            in_planes_low=feature_channels[0], in_planes_high=feature_channels[-1], n_classes=opt.num_classes
        )

    def forward(self, x):
        # Produce encoder output
        cls, features = self.metaformer(x)

        # Produce DeepLabV3+ output
        seg = self.deeplabv3p(x, features[0], features[-1])

        return cls, seg

    def forward_features_list(self, x):
        # Produce encoder output
        x, features = self.metaformer.forward_features(x)

        return features


"""""" """""" """""" """""" """""" """"""
"""" DEFINE METAFORMER-UNET MODEL"""
"""""" """""" """""" """""" """""" """"""


class MetaFormerUNetpp(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Implement and initialize backbone (Identity)
        if "identity" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = identityformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = identityformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = identityformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "identity" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = identityformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "identity" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = identityformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "rand" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = randformer_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = randformer_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = randformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "rand" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = randformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "rand" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = randformer_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Pool)
        elif "pool" in opt.backbone.lower() and "s12" in opt.backbone.lower():
            self.metaformer = poolformerv2_s12(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s24" in opt.backbone.lower():
            self.metaformer = poolformerv2_s24(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = poolformerv2_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "pool" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = poolformerv2_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)
        elif "pool" in opt.backbone.lower() and "m48" in opt.backbone.lower():
            self.metaformer = poolformerv2_m48(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 768)

        # Implement and initialize backbone (Random)
        elif "conv" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            if "waveletdown" in opt.backbone.lower():
                self.metaformer = convformer_waveletdown_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
            elif "waveletfuse" in opt.backbone.lower():
                self.metaformer = convformer_waveletfuse_s18(opt=opt, pretrained=True)
                # feature_channels = (73, 137, 329, 521)
                feature_channels = (76, 140, 332, 524)
            else:
                self.metaformer = convformer_s18(opt=opt, pretrained=True)
                feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = convformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "conv" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = convformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "conv" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = convformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Implement and initialize backbone (Random)
        elif "ca" in opt.backbone.lower() and "s18" in opt.backbone.lower():
            self.metaformer = caformer_s18(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "s36" in opt.backbone.lower():
            self.metaformer = caformer_s36(opt=opt, pretrained=True)
            feature_channels = (64, 128, 320, 512)
        elif "ca" in opt.backbone.lower() and "m36" in opt.backbone.lower():
            self.metaformer = caformer_m36(opt=opt, pretrained=True)
            feature_channels = (96, 192, 384, 576)
        elif "ca" in opt.backbone.lower() and "b36" in opt.backbone.lower():
            self.metaformer = caformer_b36(opt=opt, pretrained=True)
            feature_channels = (128, 256, 512, 768)

        # Define Exception
        else:
            raise Exception("Unrecognized MetaFormer version...")

        # Define UNet Decoder
        encoder_channels = [3] + list(feature_channels)
        decoder_channels = list(feature_channels)[::-1][1:] + [32]

        self.UNet = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels, decoder_channels=decoder_channels, n_blocks=len(decoder_channels)
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=opt.num_classes,
            activation=None,
            upsampling=2,
        )

    def forward(self, x):
        # Produce encoder output
        cls, features = self.metaformer(x)

        # Repeat first feature block
        features = [features[0], *features]

        # Produce UNet output
        seg = self.UNet(*features)
        seg = self.seg_head(seg)

        return cls, seg

    def forward_features_list(self, x):
        # Produce encoder output
        x, features = self.metaformer.forward_features(x)

        return features


"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# Default Downsampling function of MetaFormer
class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_norm=None,
        post_norm=None,
        pre_permute=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


# Basic DWT function
class Basic_DWT(nn.Module):
    def __init__(
        self,
        undecimated=True,
        mode="reflect",
        pre_permute=False,
        post_permute=True,
    ):
        super(Basic_DWT, self).__init__()

        # Define the mode
        self.mode = mode

        # Define the stride based on undecimated argument
        if undecimated:
            self.stride = 1
        else:
            self.stride = 2

        # Define 2D Haar DWT kernels
        LL = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Low Frequency content
        LH = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=torch.float32) * 0.5  # Horizontal detail
        HL = torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Vertical detail
        HH = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Diagonal detail

        # Define parameters and set requires_grad to false
        self.LL = nn.Parameter(data=LL, requires_grad=False).cuda()
        self.LH = nn.Parameter(data=LH, requires_grad=False).cuda()
        self.HL = nn.Parameter(data=HL, requires_grad=False).cuda()
        self.HH = nn.Parameter(data=HH, requires_grad=False).cuda()

        # Define whether to permute input prior/after to DWT
        self.pre_permute = pre_permute
        self.post_permute = post_permute

    def forward(self, x):
        # Permute to [BS, C, H, W] based on pre_permute argument
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)

        # Pad input
        xlc = F.pad(x, (1, 0, 1, 0), mode=self.mode, value=0)

        # Reshaping for easy convolutions to [BS*C, 1, H, W]
        xlcr = xlc.view(xlc.shape[0] * xlc.shape[1], 1, xlc.shape[2], xlc.shape[3])

        # Perform Wavelet Transform
        LL = F.conv2d(xlcr, self.LL, bias=None, stride=self.stride)
        LH = F.conv2d(xlcr, self.LH, bias=None, stride=self.stride)
        HL = F.conv2d(xlcr, self.HL, bias=None, stride=self.stride)
        HH = F.conv2d(xlcr, self.HH, bias=None, stride=self.stride)

        # Reshape back to [BS, C, H, W] and permute to output [BS, H, W, C] for MetaFormer if post_permute
        osi = [xlc.shape[0], xlc.shape[1], LL.shape[2], LL.shape[3]]
        if self.post_permute:
            out = [
                LL.view(*osi).permute(0, 2, 3, 1),
                LH.view(*osi).permute(0, 2, 3, 1),
                HL.view(*osi).permute(0, 2, 3, 1),
                HH.view(*osi).permute(0, 2, 3, 1),
            ]
        else:
            out = [LL.view(*osi), LH.view(*osi), HL.view(*osi), HH.view(*osi)]

        return out


# DWT Downsampling function
class DWTDownsampling(nn.Module):
    """ "
    Downsampling implemented by DWT
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pre_norm=None,
        post_norm=None,
        double=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()
        self.double = double

        # Define dwt based on double argument
        if self.double:
            self.dwt = Basic_DWT(undecimated=False, pre_permute=False)
            self.dwt2 = Basic_DWT(undecimated=False, pre_permute=True)
        else:
            self.dwt = Basic_DWT(undecimated=False, pre_permute=True)

        # # Option 1: Define point-wise convolution on 3*dim (HL, LH, HH)
        # self.pwconv = nn.Conv2d(in_channels=3*in_channels, out_channels=out_channels, kernel_size=(1, 1))

        # Option 2: Define point-wise convolution on 4*dim (LL, HL, LH, HH)
        self.pwconv = nn.Conv2d(
            in_channels=4 * in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )

    def forward(self, x):
        # Apply pre_norm
        x = self.pre_norm(x)

        # Apply DWT and receive components in [BS, H, W, C] format
        dwt_out = self.dwt(x)
        LL, LH, HL, HH = dwt_out[0], dwt_out[1], dwt_out[2], dwt_out[3]

        # Apply second DWT based on double, receive components in [BS, H, W, C] format
        if self.double:
            dwt_out = self.dwt2(LL)
            LL, LH, HL, HH = dwt_out[0], dwt_out[1], dwt_out[2], dwt_out[3]

        # # Option 1: Concatenate [LH, HL, HH] and apply point-wise convolution
        # x = torch.cat([LH.permute(0, 3, 1, 2), HL.permute(0, 3, 1, 2), HH.permute(0, 3, 1, 2)], dim=1)
        # x = self.pwconv(x).permute(0, 2, 3, 1)

        # Option 2: Concatenate [LL, LH, HL, HH] and apply point-wise convolution
        x = torch.cat(
            [
                LL.permute(0, 3, 1, 2),
                LH.permute(0, 3, 1, 2),
                HL.permute(0, 3, 1, 2),
                HH.permute(0, 3, 1, 2),
            ],
            dim=1,
        )
        x = self.pwconv(x).permute(0, 2, 3, 1)

        # Apply post normalization
        x = self.post_norm(x)

        return x


# Default Scale function of MetaFormer
class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


# Default SquaredReLU function of MetaFormer
class SquaredReLU(nn.Module):
    """
    Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


# Default StarReLU function of MetaFormer
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


# Default Attention function of MetaFormer
class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        **kwargs,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Custom MiT/PvT Attention function
class Attention_MiT_PvT(nn.Module):
    """
    MiT self-attention from SegFormer: https://arxiv.org/abs/2105.15203.
    Modified from https://github.com/dumyCq/ESFPNet.
    Modifications include addition of head_dim argument, Replacement of commented lines in init,
    Additional reshaping in forward and deletion of H, W inputs for forward function
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        drop=0,
    ):
        super().__init__()

        # Commented from original implementation
        # self.dim = dim
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5

        # Modifications towards Vanilla Attention implementation
        self.dim = dim
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.num_heads = num_heads if num_heads else dim // head_dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Additional reshaping
        H, W = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Additional reshaping
        x = x.reshape(B, H, W, C)

        return x


# Custom PvT Linear Attention function
class Attention_PvT_Li(nn.Module):
    """
    PvT Linear self-attention from PvT V2: https://arxiv.org/abs/2106.13797.
    Modified from https://github.com/whai362/PVT.
    Modifications include addition of head_dim argument, Replacement of commented lines in init,
    Additional reshaping in forward and deletion of H, W inputs for forward function
    Removed Linear and sr_ratio arguments, and follow up statements in init and forward
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop=0,
    ):
        super().__init__()

        # Commented from original implementation
        # self.dim = dim
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5

        # Modifications towards Vanilla Attention implementation
        self.dim = dim
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.num_heads = num_heads if num_heads else dim // head_dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Additional reshaping
        H, W = x.shape[1], x.shape[2]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Additional reshaping
        x = x.reshape(B, H, W, C)

        return x


# Custom Windowed Attention function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size=(8, 8),
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        drop=0,
    ):
        super().__init__()

        # Commented from original implementation
        # self.dim = dim
        # self.window_size = window_size  # Wh, Ww
        # self.pretrained_window_size = pretrained_window_size
        # self.num_heads = num_heads

        # Modification
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads if num_heads else dim // head_dim

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((self.num_heads, 1, 1))),
            requires_grad=True,
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=torch.float32,
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=torch.float32,
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # Additional window partition and reshaping
        B, H, W, C = x.shape
        x = window_partition(x=x, window_size=self.window_size[0])
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01)).cuda()).exp()
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Additional window_reverse
        x = window_reverse(windows=x, window_size=self.window_size[0], H=H, W=W)

        return x


# Default RandomMixing function of MetaFormer
class RandomMixing(nn.Module):
    def __init__(self, num_tokens=256, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            # data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            data=torch.softmax(torch.rand((int(num_tokens), int(num_tokens))), dim=-1),
            requires_grad=False,
        )

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum("mn, bnc -> bmc", self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


# Default LayerNorm functions of MetaFormer
class LayerNormGeneral(nn.Module):
    r"""General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(
        self,
        affine_shape=None,
        normalized_dim=(-1,),
        scale=True,
        bias=True,
        eps=1e-5,
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


# Default Separable Convolution function of MetaFormer
class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_layer=StarReLU,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


# Default Pooling function of MetaFormer
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


# Custom 2D DWT Haar Kernel function for TokenMixer
class DWT_2D_Haar_TokenMixer(nn.Module):
    def __init__(self, dim, undecimated=True, mode="reflect", drop=0):
        super(DWT_2D_Haar_TokenMixer, self).__init__()

        # Define the mode
        self.mode = mode

        # Define the stride based on undecimated argument
        if undecimated:
            self.stride = 1
        else:
            self.stride = 2

        # Define 2D Haar DWT kernels
        LL = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Low Frequency content
        LH = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=torch.float32) * 0.5  # Horizontal detail
        HL = torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Vertical detail
        HH = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=torch.float32) * 0.5  # Diagonal detail

        # Define parameters and set requires_grad to false
        self.LL = nn.Parameter(data=LL, requires_grad=False)
        self.LH = nn.Parameter(data=LH, requires_grad=False)
        self.HL = nn.Parameter(data=HL, requires_grad=False)
        self.HH = nn.Parameter(data=HH, requires_grad=False)

        # # Option 1: Define 1x1 depthwise convolution for concatenation of LH, HL, HH (3*dim)
        # self.pwconv = nn.Conv2d(in_channels=3*dim, out_channels=dim, kernel_size=(1, 1))

        # Option 2: Define 1x1 depthwise convolution for concatenation of LL, LH, HL, HH (4*dim)
        self.pwconv = nn.Conv2d(in_channels=4 * dim, out_channels=dim, kernel_size=(1, 1))

        # Define activation function
        # self.act = StarReLU()
        # self.act = SquaredReLU()
        # self.act = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        # Permute to [BS, C, H, W] for internal operation and pad
        xlc = F.pad(x.permute(0, 3, 1, 2), (1, 0, 1, 0), mode=self.mode, value=0)

        # Reshaping for easy convolutions to [BS*C, 1, H, W]
        xlcr = xlc.view(xlc.shape[0] * xlc.shape[1], 1, xlc.shape[2], xlc.shape[3])

        # Perform Wavelet Transform
        LL = F.conv2d(xlcr, self.LL, bias=None, stride=self.stride)
        LH = F.conv2d(xlcr, self.LH, bias=None, stride=self.stride)
        HL = F.conv2d(xlcr, self.HL, bias=None, stride=self.stride)
        HH = F.conv2d(xlcr, self.HH, bias=None, stride=self.stride)

        # # Option 1: reshape back to [BS, C, H, W], concatenate, depthwise convolution and permute to [BS, H, W, C]
        # osi = [xlc.shape[0], xlc.shape[1], LL.shape[2], LL.shape[3]]
        # DWT = torch.cat([LH.view(*osi), HL.view(*osi), HH.view(*osi)], dim=1)
        # out = self.pwconv(DWT).permute(0, 2, 3, 1)
        # # out = self.act(out)

        # Option 2: reshape back to [BS, C, H, W], concatenate, depthwise convolution and permute to [BS, H, W, C]
        osi = [xlc.shape[0], xlc.shape[1], LL.shape[2], LL.shape[3]]
        DWT = torch.cat([LL.view(*osi), LH.view(*osi), HL.view(*osi), HH.view(*osi)], dim=1)
        out = self.pwconv(DWT).permute(0, 2, 3, 1)
        # out = self.act(out)

        # # Option 3: reshape back to [BS, C, H, W], add up, and permute to [BS, H, W, C]
        # osi = [xlc.shape[0], xlc.shape[1], LL.shape[2], LL.shape[3]]
        # DWT = LH.view(*osi) + HL.view(*osi) + HH.view(*osi)
        # out = DWT.permute(0, 2, 3, 1)
        # # out = self.tanh(out)

        return out


# Default MLP functions of MetaFormer
class Mlp(nn.Module):
    """MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """MLP classification head"""

    def __init__(
        self,
        dim,
        num_classes=1000,
        mlp_ratio=4,
        act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm,
        head_dropout=0.0,
        bias=True,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


# Default MetaFormerBlock function of MetaFormer
class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        mlp=Mlp,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = (
            Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        )
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = (
            Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        )
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [
    partial(
        Downsampling,
        kernel_size=7,
        stride=4,
        padding=2,
        post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
    )
] + [
    partial(
        Downsampling,
        kernel_size=3,
        stride=2,
        padding=1,
        pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        pre_permute=True,
    )
] * 3


DOWNSAMPLE_DWT_FOUR_STAGES = [
    partial(
        DWTDownsampling,
        post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        double=True,
    )
] + [
    partial(
        DWTDownsampling,
        pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
    )
] * 3


"""""" """""" """""" """"""
"""" METAFORMER DEFINITIONS """
"""""" """""" """""" """"""
# Adapted from: https://github.com/sail-sg/metaformer/tree/main

urls = {
    "identityformer_s12": "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth",
    "identityformer_s24": "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth",
    "identityformer_s36": "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth",
    "identityformer_m36": "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth",
    "identityformer_m48": "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth",
    "randformer_s12": "https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth",
    "randformer_s24": "https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth",
    "randformer_s36": "https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth",
    "randformer_m36": "https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth",
    "randformer_m48": "https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth",
    "poolformerv2_s12": "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s12.pth",
    "poolformerv2_s24": "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s24.pth",
    "poolformerv2_s36": "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth",
    "poolformerv2_m36": "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m36.pth",
    "poolformerv2_m48": "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m48.pth",
    "convformer_s18": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth",
    "convformer_s18_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21ft1k.pth",
    "convformer_s18_in21k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21k.pth",
    "convformer_s36": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth",
    "convformer_s36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21ft1k.pth",
    "convformer_s36_in21k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21k.pth",
    "convformer_m36": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth",
    "convformer_m36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21ft1k.pth",
    "convformer_m36_in21k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21k.pth",
    "convformer_b36": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth",
    "convformer_b36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth",
    "convformer_b36_in21k": "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth",
    "caformer_s18": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",
    "caformer_s18_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth",
    "caformer_s18_in21k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth",
    "caformer_s36": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth",
    "caformer_s36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth",
    "caformer_s36_in21k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth",
    "caformer_m36": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth",
    "caformer_m36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth",
    "caformer_m36_in21k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth",
    "caformer_b36": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth",
    "caformer_b36_in21ft1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth",
    "caformer_b36_in21k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth",
}


# Basic MetaFormer implementation
class MetaFormer(nn.Module):
    r"""MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        # partial(LayerNormGeneral, eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        feature_list = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_list.append(x.permute(0, 3, 1, 2))
        return (
            self.norm(x.mean([1, 2])),
            feature_list,
        )  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x, features = self.forward_features(x)
        x = self.head(x)
        return x, features


# MetaFormer with infusion of DWT components of input image
class MetaFormer_DWT(nn.Module):
    r"""MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        # partial(LayerNormGeneral, eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

        # Define DWT extractor
        self.DWT = Basic_DWT(undecimated=False, pre_permute=False, post_permute=False)

        # # Optional: Define additional point-wise convolutions for fusion after downsampling
        # self.pwconv0 = nn.Conv2d(in_channels=dims[0]+9, out_channels=dims[0], kernel_size=(1, 1))
        # self.pwconv1 = nn.Conv2d(in_channels=dims[1]+9, out_channels=dims[1], kernel_size=(1, 1))
        # self.pwconv2 = nn.Conv2d(in_channels=dims[2]+9, out_channels=dims[2], kernel_size=(1, 1))
        # self.pwconv3 = nn.Conv2d(in_channels=dims[3]+9, out_channels=dims[3], kernel_size=(1, 1))
        # self.pwconv_layers = [self.pwconv0, self.pwconv1, self.pwconv2, self.pwconv3]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        # Additional DWT components from input image
        out_dwt2 = self.DWT(x)
        out_dwt4 = self.DWT(out_dwt2[0])
        out_dwt8 = self.DWT(out_dwt4[0])
        out_dwt16 = self.DWT(out_dwt8[0])
        out_dwt32 = self.DWT(out_dwt16[0])
        dwt_list = [out_dwt4, out_dwt8, out_dwt16, out_dwt32]

        feature_list = []
        for i in range(self.num_stage):
            # Downsampling
            x = self.downsample_layers[i](x)

            # # Optional: Fusion of DWT components with point-wise convolutions
            # x = self.pwconv_layers[i](torch.cat([x.permute(0, 3, 2, 1), dwt_list[i][1], dwt_list[i][2], dwt_list[i][3]], dim=1)).permute(0, 2, 3, 1)

            # Stages
            x = self.stages[i](x)

            # # Option 1: Fusion of 3 DWT components for Decoder
            # feature_list.append(
            #     torch.cat(
            #         [
            #             x.permute(0, 3, 1, 2),
            #             dwt_list[i][1],
            #             dwt_list[i][2],
            #             dwt_list[i][3],
            #         ],
            #         dim=1,
            #     )
            # )

            # Option 2: Fusion of all DWT components for Decoder
            feature_list.append(
                torch.cat(
                    [x.permute(0, 3, 1, 2), dwt_list[i][0], dwt_list[i][1], dwt_list[i][2], dwt_list[i][3]], dim=1
                )
            )

        return (
            self.norm(x.mean([1, 2])),
            feature_list,
        )  # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x, features = self.forward_features(x)
        x = self.head(x)
        return x, features


"""IDENTITY MAPPING AS TOKEN MIXER"""


@register_model
def identityformer_s12(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["identityformer_s12"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def identityformer_s24(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["identityformer_s24"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def identityformer_s36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["identityformer_s36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def identityformer_m36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["identityformer_m36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def identityformer_m48(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["identityformer_m48"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


"""RANDOM MIXING AS TOKEN MIXER"""


@register_model
def randformer_s12(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            nn.Identity,
            nn.Identity,
            partial(RandomMixing, num_tokens=(opt.imagesize / 16) ** 2),
            partial(RandomMixing, num_tokens=(opt.imagesize / 32) ** 2),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["randformer_s12"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        for i in range(6):
            del state_dict[f"stages.2.{i}.token_mixer.random_matrix"]
            if i < 2:
                del state_dict[f"stages.3.{i}.token_mixer.random_matrix"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def randformer_s24(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            nn.Identity,
            nn.Identity,
            partial(RandomMixing, num_tokens=(opt.imagesize / 16) ** 2),
            partial(RandomMixing, num_tokens=(opt.imagesize / 32) ** 2),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["randformer_s24"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        for i in range(12):
            del state_dict[f"stages.2.{i}.token_mixer.random_matrix"]
            if i < 4:
                del state_dict[f"stages.3.{i}.token_mixer.random_matrix"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def randformer_s36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            nn.Identity,
            nn.Identity,
            partial(RandomMixing, num_tokens=(opt.imagesize / 16) ** 2),
            partial(RandomMixing, num_tokens=(opt.imagesize / 32) ** 2),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["randformer_s36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        for i in range(18):
            del state_dict[f"stages.2.{i}.token_mixer.random_matrix"]
            if i < 6:
                del state_dict[f"stages.3.{i}.token_mixer.random_matrix"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def randformer_m36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            nn.Identity,
            nn.Identity,
            partial(RandomMixing, num_tokens=(opt.imagesize / 16) ** 2),
            partial(RandomMixing, num_tokens=(opt.imagesize / 32) ** 2),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["randformer_m36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        for i in range(18):
            del state_dict[f"stages.2.{i}.token_mixer.random_matrix"]
            if i < 6:
                del state_dict[f"stages.3.{i}.token_mixer.random_matrix"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def randformer_m48(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            nn.Identity,
            nn.Identity,
            partial(RandomMixing, num_tokens=(opt.imagesize / 16) ** 2),
            partial(RandomMixing, num_tokens=(opt.imagesize / 32) ** 2),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["randformer_m48"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        for i in range(24):
            del state_dict[f"stages.2.{i}.token_mixer.random_matrix"]
            if i < 8:
                del state_dict[f"stages.3.{i}.token_mixer.random_matrix"]
        model.load_state_dict(state_dict, False)
    return model


"""AVERAGE POOLING AS TOKEN MIXER"""


@register_model
def poolformerv2_s12(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=Pooling,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["poolformerv2_s12"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def poolformerv2_s24(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=Pooling,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["poolformerv2_s24"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def poolformerv2_s36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=Pooling,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["poolformerv2_s36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def poolformerv2_m36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=Pooling,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["poolformerv2_m36"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def poolformerv2_m48(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=Pooling,
        mlps=Mlp,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        **kwargs,
    )

    url = urls["poolformerv2_m48"]
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        model.load_state_dict(state_dict, False)
    return model


"""DEPTHWISE SEPARABLE CONVOLUTION AS TOKEN MIXER"""


@register_model
def convformer_s18(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s18"]
    # url = urls['convformer_s18_in21ft1k']
    # url = urls['convformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def convformer_s36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s36"]
    # url = urls['convformer_s36_in21ft1k']
    # url = urls['convformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def convformer_m36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_m36"]
    # url = urls['convformer_m36_in21ft1k']
    # url = urls['convformer_m36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def convformer_b36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_b36"]
    # url = urls['convformer_b36_in21ft1k']
    # url = urls['convformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


"""DEPTHWISE SEPARABLE CONVOLUTION AND ATTENTION AS TOKEN MIXER"""


@register_model
# def caformer_s18(opt, pretrained=False, **kwargs):
#     model = MetaFormer(
#         in_chans=3,
#         num_classes=opt.num_classes,
#         depths=[3, 3, 9, 3],
#         dims=[64, 128, 320, 512],
#         downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
#         token_mixers=[SepConv, SepConv, Attention, Attention],
#         mlps=Mlp,
#         norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
#         drop_path_rate=0.0,
#         head_dropout=0.0,
#         layer_scale_init_values=None,
#         res_scale_init_values=[None, None, 1.0, 1.0],
#         output_norm=partial(nn.LayerNorm, eps=1e-6),
#         head_fn=MlpHead,
#         **kwargs,
#     )
#
#     url = urls["caformer_s18"]
#     # url = urls['caformer_s18_in21ft1k']
#     # url = urls['caformer_s18_in21k']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         del state_dict["head.fc2.weight"]
#         del state_dict["head.fc2.bias"]
#         model.load_state_dict(state_dict, False)
#
#         # print('Loading DINO GastroNet Weights...')
#         # state_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'checkpoint0010_teacher.pth'))
#         # model.load_state_dict(state_dict, False)
#
#     return model


@register_model
def caformer_s18(opt, pretrained=False, **kwargs):
    if opt.weights == 'ImageNet':
        model = MetaFormer(
            in_chans=3,
            num_classes=opt.num_classes,
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
            token_mixers=[SepConv, SepConv, Attention, Attention],
            mlps=Mlp,
            norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
            drop_path_rate=0.0,
            head_dropout=0.0,
            layer_scale_init_values=None,
            res_scale_init_values=[None, None, 1.0, 1.0],
            output_norm=partial(nn.LayerNorm, eps=1e-6),
            head_fn=MlpHead,
            **kwargs,
        )

        url = urls["caformer_s18"]
        # url = urls['caformer_s18_in21ft1k']
        # url = urls['caformer_s18_in21k']
        if pretrained:
            print('Loading ImageNet Weights in CAFormer with StarReLU...')
            state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
            del state_dict["head.fc2.weight"]
            del state_dict["head.fc2.bias"]
            model.load_state_dict(state_dict, False)

    else:
        model = MetaFormer(
            in_chans=3,
            num_classes=opt.num_classes,
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
            token_mixers=[
                partial(SepConv, act1_layer=nn.ReLU),
                partial(SepConv, act1_layer=nn.ReLU),
                Attention,
                Attention,
            ],
            mlps=partial(Mlp, act_layer=nn.ReLU),
            norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
            drop_path_rate=0.0,
            head_dropout=0.0,
            layer_scale_init_values=None,
            res_scale_init_values=[None, None, 1.0, 1.0],
            output_norm=partial(nn.LayerNorm, eps=1e-6),
            head_fn=MlpHead,
            **kwargs,
        )

        if pretrained:
            if opt.weights == 'GastroNet':
                # GastroNet pretrained weights
                print('Loading DINO Default GastroNet Weights in CAFormer with ReLU...')
                state_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'checkpoint0100_teacher.pth'))
                model.load_state_dict(state_dict, False)

            elif opt.weights == 'GastroNet-DSA':
                # GastroNet pretrained weights
                print('Loading DINO DSA GastroNet Weights in CAFormer with ReLU...')
                state_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'checkpoint0100_teacher_custom.pth'))
                model.load_state_dict(state_dict, False)

    return model


@register_model
def caformer_s36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s36"]
    # url = urls['caformer_s36_in21ft1k']
    # url = urls['caformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def caformer_m36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_m36"]
    # url = urls['caformer_m36_in21ft1k']
    # url = urls['caformer_m36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


@register_model
def caformer_b36(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_b36"]
    # url = urls['caformer_b36_in21ft1k']
    # url = urls['caformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


"""DEPTHWISE SEPARABLE CONVOLUTION AND CUSTOM ATTENTION AS TOKEN MIXER"""


# Mix Vision Transformer Attention Block
@register_model
def caformer_s18_mit(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention_MiT_PvT, Attention_MiT_PvT],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    # url = urls['caformer_s18_in21ft1k']
    # url = urls['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# Pyramid Vision Transformer Attention Block
@register_model
def caformer_s18_pvt(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention_PvT_Li, Attention_PvT_Li],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    # url = urls['caformer_s18_in21ft1k']
    # url = urls['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# Swin Vision Transformer Attention Block
@register_model
def caformer_s18_swin(opt, pretrained=False, **kwargs):
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, WindowAttention, WindowAttention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    # url = urls['caformer_s18_in21ft1k']
    # url = urls['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


"""DWT AS TOKEN MIXER"""


# Use configuration with [DWT, DWT, DWT, DWT]
@register_model
def dwtformer_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtformer_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            partial(DWT_2D_Haar_TokenMixer, dim=320),
            partial(DWT_2D_Haar_TokenMixer, dim=512),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "qkv" in key or "proj" in key or "act1" in key or "pwconv" in key or "dwconv" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# Use configuration with [SepConv, SepConv, DWT, DWT]
@register_model
def cdwtformer_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: cdwtformer_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            SepConv,
            SepConv,
            partial(DWT_2D_Haar_TokenMixer, dim=320),
            partial(DWT_2D_Haar_TokenMixer, dim=512),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "qkv" in key or "proj" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# Use configuration with [DWT, DWT, SepConv, SepConv]
@register_model
def dwtcformer_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtcformer_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            SepConv,
            SepConv,
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            for i in range(2):
                for j in range(3):
                    if f"stages.{i}.{j}.token_mixer" in key:
                        delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# Use configuration with [DWT, DWT, Attention, Attention]
@register_model
def dwtaformer_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtaformer_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            Attention,
            Attention,
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            for i in range(2):
                for j in range(3):
                    if f"stages.{i}.{j}.token_mixer" in key:
                        delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


"""DWT AS DOWNSAMPLING OPERATOR"""


# ConvFormer with DWT as downsampling operator
@register_model
def convformer_waveletdown_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: convformer_waveletdown_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_DWT_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s18"]
    # url = urls['convformer_s18_in21ft1k']
    # url = urls['convformer_s18_in21k']
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "downsample_layers" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


# CaFormer with DWT as downsampling operator
@register_model
def caformer_waveletdown_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: convformer_waveletdown_s18")
    model = MetaFormer(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_DWT_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    # url = urls['caformer_s18_in21ft1k']
    # url = urls['caformer_s18_in21k']
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "downsample_layers" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


"""IMAGE DWT COMPONENTS FUSED WITH FEATURES"""


# ConvFormer with Image DWT Components fused with features
@register_model
def convformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: convformer_waveletfuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=SepConv,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s18"]
    # url = urls['convformer_s18_in21ft1k']
    # url = urls['convformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


# CaFormer with Image DWT Components fused with features
@register_model
def caformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: caformer_waveletfuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    # url = urls['caformer_s18_in21ft1k']
    # url = urls['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, False)
    return model


# DWTFormer with Image DWT Components fused with features
@register_model
def dwtformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtformer_wavelet_fuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            partial(DWT_2D_Haar_TokenMixer, dim=320),
            partial(DWT_2D_Haar_TokenMixer, dim=512),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "qkv" in key or "proj" in key or "act1" in key or "pwconv" in key or "dwconv" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# CDWTFormer with Image DWT Components fused with features
@register_model
def cdwtformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: cdwtformer_waveletfuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            SepConv,
            SepConv,
            partial(DWT_2D_Haar_TokenMixer, dim=320),
            partial(DWT_2D_Haar_TokenMixer, dim=512),
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            if "qkv" in key or "proj" in key:
                delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# DWTCFormer with Image DWT Components fused with features
@register_model
def dwtcformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtcformer_waveletfuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            SepConv,
            SepConv,
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["convformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            for i in range(2):
                for j in range(3):
                    if f"stages.{i}.{j}.token_mixer" in key:
                        delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


# DWTAFormer with Image DWT Components fused with features
@register_model
def dwtaformer_waveletfuse_s18(opt, pretrained=False, **kwargs):
    print("Loaded backbone model: dwtaformer_waveletfuse_s18")
    model = MetaFormer_DWT(
        in_chans=3,
        num_classes=opt.num_classes,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=[
            partial(DWT_2D_Haar_TokenMixer, dim=64),
            partial(DWT_2D_Haar_TokenMixer, dim=128),
            Attention,
            Attention,
        ],
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=MlpHead,
        **kwargs,
    )

    url = urls["caformer_s18"]
    if pretrained:
        delete_keys = []
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        for key in state_dict:
            for i in range(2):
                for j in range(3):
                    if f"stages.{i}.{j}.token_mixer" in key:
                        delete_keys.append(key)
        for del_key in delete_keys:
            del state_dict[del_key]
        del state_dict["head.fc2.weight"]
        del state_dict["head.fc2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return model


"""""" """""" """""" """""" """"""
"""" SEMANTIC FPN DEFINITIONS """
"""""" """""" """""" """""" """"""
# Code adapted from:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/fpn/decoder.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/modules.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/heads.py


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=None,
        upsampling=1,
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class FPN(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        num_classes=1,
        interpolation=4,
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(
                    pyramid_channels,
                    segmentation_channels,
                    n_upsamples=n_upsamples,
                )
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.segmentation_head = SegmentationHead(
            in_channels=self.out_channels,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=interpolation,
        )

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        seg = self.segmentation_head(x)

        return seg


"""""" """""" """""" """""" """"""
"""" UPERNET DEFINITIONS """
"""""" """""" """""" """""" """"""


class PSPModule(nn.Module):
    # In the original implementation they use precise RoI pooling
    # Instead of using adaptive average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels * len(bin_sizes)),
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.0)
            nn.Dropout2d(0.1),
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        # Input (BS, 768, 8, 8) --> (BS, 768, bin_sz, bin_sz)
        if bin_sz == 1:
            prior = nn.AvgPool2d(kernel_size=(8, 8))
        elif bin_sz == 2:
            prior = nn.AvgPool2d(kernel_size=(7, 7))
        elif bin_sz == 4:
            prior = nn.AvgPool2d(kernel_size=(5, 5))
        elif bin_sz == 6:
            prior = nn.AvgPool2d(kernel_size=(3, 3))

        # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend(
            [
                F.interpolate(
                    stage(features),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True,
                )
                for stage in self.stages
            ]
        )

        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList(
            [nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels) - 1)
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(
                len(feature_channels) * fpn_out,
                fpn_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]  ##
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)

        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]
        x = self.conv_fusion(torch.cat((P), dim=1))

        return x


"""""" """""" """""" """""" """"""
"""" DEEPLABV3+ DEFINITIONS """
"""""" """""" """""" """""" """"""


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
    def __init__(self, in_planes_low, in_planes_high, n_classes=1, os=16):
        super(DeepLabv3_plus, self).__init__()

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(in_planes_high, 256, rate=rates[0])
        self.aspp2 = ASPP_module(in_planes_high, 256, rate=rates[1])
        self.aspp3 = ASPP_module(in_planes_high, 256, rate=rates[2])
        self.aspp4 = ASPP_module(in_planes_high, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_planes_high, 256, (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(in_planes_low, 48, (1, 1), bias=False)
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

        # x = F.interpolate(
        #     x,
        #     size=(
        #         int(math.ceil(img.size()[-2] / 4)),
        #         int(math.ceil(img.size()[-1] / 4)),
        #     ),
        #     mode='bilinear',
        #     align_corners=True,
        # )

        # Compatibility TensorRT
        x = F.interpolate(
            x,
            size=(64, 64),
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


if __name__ == "__main__":
    import argparse
    from torchinfo import summary

    def get_params():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # DEFINE MODEL
        parser.add_argument("--backbone", type=str, default="MetaFormer-CAS18-UperNet")
        parser.add_argument("--seg_branch", type=str, default=None)
        parser.add_argument('--weights', type=str, default='ImageNet')

        # AUGMENTATION PARAMS
        parser.add_argument("--imagesize", type=int, default=256)
        parser.add_argument("--batchsize", type=int, default=16)
        parser.add_argument("--num_classes", type=int, default=1)

        args = parser.parse_args()

        return args

    opt = get_params()

    # For Full Segmentation model
    # model = MetaFormerUperNet(opt=opt).cuda()
    # model = MetaFormerFPN(opt=opt).cuda()
    # model = MetaFormerDeepLabV3p(opt=opt).cuda()
    model = MetaFormerUNetpp(opt=opt).cuda()
    dummy = torch.zeros([12, 3, 256, 256]).cuda()
    out, seg = model(dummy)
    print(out.shape, seg.shape)
    summary(model, input_size=(1, 3, 256, 256))
