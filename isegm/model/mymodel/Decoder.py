import torch
import torch.nn as nn
from isegm.model.is_model import ISModel
from isegm.utils.serialization import serialize
from isegm.model.point_flow.ASPP import ASPP

class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)

class decoderlayer(nn.Module):
    def __init__(self, inpchannel, oupchannel):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inpchannel, oupchannel, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(oupchannel),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(oupchannel, oupchannel, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(oupchannel),
                                   nn.ReLU())

    def forward(self, skipfeat, hfeat):
        feat = self.conv1(torch.cat([skipfeat, hfeat], dim=1))
        skipfeat = skipfeat + feat
        feat = self.conv2(feat)
        return skipfeat, feat


class Decoder(nn.Module):

    def __init__(self, asppchannel):
        super(Decoder, self).__init__()
        self.up1 = decoderlayer(asppchannel + 64, 64)
        self.up2 = decoderlayer(64 + 64, 64)
        self.relu = nn.ReLU()

    def forward(self, lfeat, hfeat):
        skipout = []
        hfeat = nn.functional.interpolate(hfeat, scale_factor=2, mode='bilinear', align_corners=False)
        skipfeat, feat = self.up1(lfeat[0], hfeat)
        skipout.append(skipfeat)
        feat = nn.functional.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        skipfeat, feat = self.up2(lfeat[1], feat)
        skipout.append(skipfeat)
        return skipout, feat

    # def forward(self, lfeat, hfeat):
    #     skipout = []
    #     hfeat = nn.functional.interpolate(hfeat, scale_factor=2, mode='bilinear', align_corners=False)  # up为1/4
    #     skipfeat, feat1 = self.up1(lfeat[0], hfeat)
    #     skipout.append(skipfeat)
    #     feat2 = nn.functional.interpolate(feat1, scale_factor=2, mode='bilinear', align_corners=False)  # up为1/2
    #     skipfeat, feat3 = self.up2(lfeat[1], feat2)
    #     skipout.append(skipfeat)
    #     return skipout, feat1