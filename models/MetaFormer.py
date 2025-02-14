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
            if 'GastroNet-DSA' in opt.weights:
                # GastroNet pretrained weights
                print('Loading DINO DSA GastroNet Weights in CAFormer with ReLU...')
                state_dict = torch.load(opt.weights)
                model.load_state_dict(state_dict, False)

            elif 'GastroNet' in opt.weights:
                # GastroNet pretrained weights
                print('Loading DINO Default GastroNet Weights in CAFormer with ReLU...')
                state_dict = torch.load(opt.weights)
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
