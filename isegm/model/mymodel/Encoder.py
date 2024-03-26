import torch
import torch.nn as nn
from isegm.model.is_model import ISModel
from isegm.utils.serialization import serialize
from isegm.model.point_flow.ASPP import ASPP
from isegm.model.mymodel.Resnet import conv1x1, conv3x3, Bottleneck
from torch import nn, Tensor
from collections import OrderedDict
from typing import Dict, List
from torchvision import models
from isegm.model.point_flow.ASPP import ASPP

BN_MOMENTUM = 0.1

class fixBlock(nn.Module):

    def __init__(self, inp_channel, midchannel):
        super(fixBlock, self).__init__()
        self.conv = nn.Conv2d(2 * inp_channel, midchannel, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(midchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inp_channel, midchannel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midchannel)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        y = self.conv1(y)
        y = self.bn1(y)
        out = torch.cat([x, y], dim=1)
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class encoderlayer(nn.Module):

    def __init__(self, inp_channel, midchannel, oup_channel, down):
        super(encoderlayer, self).__init__()
        if down:
            downsample = nn.Sequential(
                nn.Conv2d(inp_channel, oup_channel, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(oup_channel, momentum=BN_MOMENTUM)
            )
            self.down = BasicBlock(inp_channel, oup_channel, stride=2, downsample=downsample)
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inp_channel, oup_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(oup_channel, momentum=BN_MOMENTUM)
            )
            self.down = BasicBlock(inp_channel, oup_channel, stride=1, downsample=downsample)
        self.fix = fixBlock(oup_channel, midchannel)
        self.layer = nn.Sequential(
            BasicBlock(oup_channel, oup_channel, stride=1),
            BasicBlock(oup_channel, oup_channel, stride=1),
            BasicBlock(oup_channel, oup_channel, stride=1),
            BasicBlock(oup_channel, oup_channel, stride=1)
        )

    def forward(self, x, y):
        y = self.down(y)
        feat = self.fix(x, y)
        feat = self.layer(feat)
        feat = feat + x
        return feat


class OtherEncoder(nn.Module):

    def __init__(self, inpchannel, ch):
        super(OtherEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inpchannel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.fix0 = fixBlock(64, midchannel=32)
        self.layer1 = encoderlayer(inp_channel=64, oup_channel=64, midchannel=32, down=True)
        self.layer2 = encoderlayer(inp_channel=64, oup_channel=128, midchannel=64, down=True)
        self.layer3 = encoderlayer(inp_channel=128, oup_channel=256, midchannel=128, down=False)
        self.layer4 = encoderlayer(inp_channel=256, oup_channel=512, midchannel=256, down=False)
        self.aspp = ASPP(in_channels=512, atrous_rates=[12, 24, 36], out_channels=ch)

    def forward(self, x, y: list):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fix0(y[0], x) + y[0]
        output = [x]
        x = self.layer1(y[1], x)
        output.append(x)
        x = self.layer2(y[2], x)
        output.append(x)
        x = self.layer3(y[3], x)
        output.append(x)
        x = self.layer4(y[4], x)
        output.append(x)
        x = self.aspp(x)

        return output, x
