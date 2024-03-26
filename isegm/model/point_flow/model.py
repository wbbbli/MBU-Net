import torch
import torch.nn as nn
from isegm.model.is_model import ISModel
from isegm.utils.serialization import serialize
from isegm.model.point_flow.ASPP import ASPP


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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

    def forward(self, x):
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


class fixblock(nn.Module):

    def __init__(self, inp_channel):
        super(fixblock, self).__init__()
        self.conv = nn.Conv2d(2 * inp_channel, inp_channel, kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.bn = nn.BatchNorm2d(inp_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class upblock(nn.Module):

    def __init__(self, inp_channel: list, oup_channel: list):
        super(upblock, self).__init__()
        self.hconv1 = nn.Conv2d(inp_channel[0], oup_channel[0], kernel_size=3, stride=1,
                                padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(oup_channel[0])
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(oup_channel[0] + inp_channel[1], oup_channel[1], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(oup_channel[1])

    def forward(self, lfeat, hfeat):
        hfeat = self.hconv1(hfeat)
        hfeat = self.bn1(hfeat)
        hfeat = self.relu(hfeat)
        hfeat = nn.functional.interpolate(hfeat, scale_factor=2, mode='bilinear', align_corners=False)
        feat = torch.cat([lfeat, hfeat], dim=1)
        feat = self.conv2(feat)
        feat = self.bn2(feat)
        feat = self.relu(feat)
        return feat


class ResNet1(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.aspp = ASPP(in_channels=1024, atrous_rates=[12, 24, 36], out_channels=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output = [x]

        x = self.maxpool(x)
        x = self.layer1(x)
        output.append(x)
        x = self.layer2(x)
        output.append(x)
        x = self.layer3(x)
        output.append(x)
        x = self.layer4(x)
        x = self.aspp(x)
        output.append(x)

        return output


class ResNet2(nn.Module):

    def __init__(self, block, layers, inp_channel, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inp_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.fix0 = fixblock(64)
        self.fix1 = fixblock(128)
        self.fix2 = fixblock(256)
        self.fix3 = fixblock(512)

        self.aspp = ASPP(in_channels=1024, atrous_rates=[12, 24, 36], out_channels=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, aux_inp):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fix0(x, aux_inp[0])
        output = [x]

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.fix1(x, aux_inp[1])
        output.append(x)
        x = self.layer2(x)
        x = self.fix2(x, aux_inp[2])
        output.append(x)
        x = self.layer3(x)
        x = self.fix3(x, aux_inp[3])
        output.append(x)
        x = self.layer4(x)
        x = self.aspp(x)
        output.append(x)

        return output


class Attention(nn.Module):
    def __init__(self, inp_channel, key_channel):
        super(Attention, self).__init__()
        self.inp_channel = inp_channel
        self.key_channels = key_channel
        self.keymar = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_channel, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.querymar = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_channel, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.valuemar = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_channel, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        query = self.querymar(x1).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.keymar(x2).view(batch_size, self.key_channels, -1)
        value = self.valuemar(x2).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = nn.functional.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x1.size()[2:])

        return context

class PtNet(ISModel):
    @serialize
    def __init__(self, block=Bottleneck, layers=[2, 3, 5, 2], replace_stride_with_dilation=None, **kwargs):
        super().__init__(norm_layer=nn.BatchNorm2d, **kwargs)

        self.img_encoder = ResNet1(block, layers, replace_stride_with_dilation=replace_stride_with_dilation)
        self.space_encoder = ResNet2(block, layers, inp_channel=2,
                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.semantics_attention = Attention(inp_channel=512, key_channel=256)

        self.up1 = upblock(inp_channel=[256, 256], oup_channel=[256, 256])  # hfeat, lfeat, oup_channel
        self.up2 = upblock(inp_channel=[256, 64], oup_channel=[128, 128])
        self.up3 = upblock(inp_channel=[128, 5], oup_channel=[64, 64])
        self.seg = nn.Sequential(
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),

            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )

    def backbone_forward(self, img, point):
        imgfeat = self.img_encoder(img)
        spacefeat = self.space_encoder(point, imgfeat)
        feat = self.semantics_attention(imgfeat[-1], spacefeat[-1])

        # up
        feat = self.up1(spacefeat[2], feat)
        feat = self.up2(imgfeat[0], feat)
        feat = self.up3(torch.concat([img, point], dim=1), feat)
        result = self.seg(feat)

        return {'instances': result}


if __name__ == '__main__':
    device = torch.device('cuda:1')
    model = PtNet(replace_stride_with_dilation=[True, False, True]).to(device)
    x = torch.ones((2, 3, 480, 480)).to(device)
    point = torch.ones((2, 2, 480, 480)).to(device)
    mask = torch.ones((2, 1, 480, 480)).to(device)
    output = model(x, point)
    print(output.shape)