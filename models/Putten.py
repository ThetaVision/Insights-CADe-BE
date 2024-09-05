"""
GastroNet Implementation that corresponds to the original implementation in papers:
    - van der Putten et al. - Pseudo-labeled bootstrapping and multi-stage transfer learning for the classification and
                              localization of dysplasia in Barrett's esophagus (MLMI 2019)
                              https://link.springer.com/chapter/10.1007/978-3-030-32692-0_20
    - van der Putten et al. - Multi-stage domain-specific pretraining for improved detection and localization of
                              Barrett's neoplasia: a comprehensive clinically validated study (AI Med 2020)
                              https://www.sciencedirect.com/science/article/pii/S0933365720300488?via%3Dihub
    - de Groof et al. - Deep-learning system detects neoplasia in patients with Barrett's esophagus with higher accuracy
                        than endoscopists in a multistep training and validation study with benchmarking (Gastro 2020)
                        https://www.sciencedirect.com/science/article/pii/S0016508519415862

    - van der Putten et al. - Endoscopy-driven pretraining for classification of dysplasia in Barrett's esophagus with
                              endoscopic narrow-band imaging zoom videos (Appl. Sci. 2020)
                              https://www.mdpi.com/2076-3417/10/10/3407
    - Struyvenberg et al. - A computer-assisted algorithm for narrow-band imaging-based tissue characterization in
                            Barrett's esophagus (GIE 2021)
                            https://www.sciencedirect.com/science/article/pii/S001651072034400X
"""
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import ResNet18


"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class classifier(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class classifier_ftrs(nn.Module):
    def __init__(self, inplanes, h_dim, num_classes):
        super(classifier_ftrs, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.ftrs = nn.Linear(256, h_dim)
        self.fc_end = nn.Linear(h_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, ftrs=False):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.ftrs(x)
        x = self.relu(x)
        if ftrs:
            return x
        else:
            x = self.fc_end(x)
            return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.expand = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.conv1 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.expand(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), BasicBlock(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = BasicBlock(out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))

        x = x1 + x2
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Conv2d(out_ch, 1, 3, padding=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


"""""" """""" """""" """"""
"""" GASTRONET DEFINITION """
"""""" """""" """""" """"""


class GastroNet(nn.Module):
    def __init__(self, opt, n_channels, n_classes, init_channels=32, h_dim=128, mode='classification'):
        super(GastroNet, self).__init__()

        # Define Mode
        self.mode = mode

        # Define part of the network
        self.init_channels = init_channels
        self.inc = inconv(n_channels, init_channels)
        self.down1 = down(init_channels, 2 * init_channels)
        self.down2 = down(2 * init_channels, 4 * init_channels)
        self.down3 = down(4 * init_channels, 8 * init_channels)
        self.down4 = down(8 * init_channels, 16 * init_channels)
        self.classifier = classifier_ftrs(16 * init_channels, h_dim, n_classes)
        self.up1 = up(16 * init_channels, 8 * init_channels)
        self.up2 = up(8 * init_channels, 4 * init_channels)
        self.up3 = up(4 * init_channels, 2 * init_channels)
        self.up4 = up(2 * init_channels, init_channels)
        self.outconv = outconv(init_channels, n_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_mode(self, mode):
        if mode in ['segmentation', 'classification', 'features']:
            self.mode = mode
            print('changed mode to ' + mode)
        else:
            print('mode type not supported, only "classification", "segmentation", and "features" are supported')

    def update_classifier(self, n_classes, hdim=128):
        self.classifier = classifier_ftrs(16 * self.init_channels, hdim, n_classes)
        self.outconv = outconv(self.init_channels, n_classes)
        print('changed classifier output to {} classes'.format(n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

        if self.mode == 'classification':
            out = self.classifier(x5)
            return out

        elif self.mode == 'features':
            out = self.classifier(x5, ftrs=True)
            return out

        elif self.mode == 'segmentation':
            out = self.classifier(x5)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outconv(x)
            return out, x
        else:
            print('unrecognized mode type, only "classification", "segmentation", and "features" are supported')


class GastroNet_ResNet18(nn.Module):
    def __init__(self, n_channels, n_classes, init_channels=32, h_dim=32, mode='classification', url=''):
        super(GastroNet_ResNet18, self).__init__()

        # Define Mode
        self.mode = mode

        # Define parts of the network besides backbone encoder
        self.x1down = nn.Conv2d(64, init_channels, kernel_size=1, stride=1, bias=False)
        self.classifier = classifier_ftrs(16 * init_channels, h_dim, n_classes)
        self.up1 = up(16 * init_channels, 8 * init_channels)
        self.up2 = up(8 * init_channels, 4 * init_channels)
        self.up3 = up(4 * init_channels, 2 * init_channels)
        self.up4 = up(2 * init_channels, init_channels)
        self.outconv = outconv(init_channels, n_classes)

        # Initialize weights for the parts of the network excluding the backbone
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Define Backbone model
        self.encoder = ResNet18(num_classes=n_classes, channels=n_channels, pretrained='ImageNet', url=url)

    def update_mode(self, mode):
        if mode in ['segmentation', 'classification', 'features']:
            self.mode = mode
            print('changed mode to ' + mode)
        else:
            print('mode type not supported, only "classification", "segmentation", and "features" are supported')

    def update_classifier(self, n_classes, hdim=128):
        self.classifier = classifier_ftrs(16 * self.init_channels, hdim, n_classes)
        self.outconv = outconv(self.init_channels, n_classes)
        print('changed classifier output to {} classes'.format(n_classes))

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder.forward_features_all(x)

        if self.mode == 'classification':
            out = self.classifier(x5)
            return out

        elif self.mode == 'features':
            out = self.classifier(x5, ftrs=True)
            return out

        elif self.mode == 'segmentation':
            out = self.classifier(x5)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, self.x1down(x1))
            x = self.outconv(x)
            return out, x
        else:
            print('unrecognized mode type, only "classification", "segmentation", and "features" are supported')
