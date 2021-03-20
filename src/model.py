"""
This file is partially from https://github.com/GlassyWu/KTDN.
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import res2net as Pre_Res2Net

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # out_channel=512
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)  # x16

        return x, x_layer1, x_layer2


######################
# decoder
######################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res

class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)

        #self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = F.interpolate(self.relu(self.conv1010(x101)), size=shape_out, mode='nearest')
        x1020 = F.interpolate(self.relu(self.conv1020(x102)), size=shape_out, mode='nearest')
        x1030 = F.interpolate(self.relu(self.conv1030(x103)), size=shape_out, mode='nearest')
        x1040 = F.interpolate(self.relu(self.conv1040(x104)), size=shape_out, mode='nearest')

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = torch.tanh(self.refine3(dehaze))

        return dehaze

class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

        self.encoder = Res2Net(Bottle2neck,
                               [3, 4, 23, 3],
                               baseWidth=26,
                               scale=4)
        res2net101 = Pre_Res2Net.Res2Net(Bottle2neck,
                                         [3, 4, 23, 3],
                                         baseWidth=26,
                                         scale=4)
        res2net101.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

        self.mid_conv = DehazeBlock(default_conv, 1024, 3)

        self.up_block1 = nn.PixelShuffle(2)
        self.up_block2 = nn.PixelShuffle(2)
        self.up_block3 = nn.PixelShuffle(2)
        self.up_block4 = nn.PixelShuffle(2)
        self.attention1 = DehazeBlock(default_conv, 256, 3)
        self.attention2 = DehazeBlock(default_conv, 192, 3)
        self.enhancer = Enhancer(28, 28)
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(28, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        x, x_layer1, x_layer2 = self.encoder(input)

        x_mid = self.mid_conv(x)

        x = self.up_block1(x_mid)
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)
        x = self.up_block2(x)
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), 1)
        x = self.up_block3(x)
        x = self.up_block4(x)

        x = self.enhancer(x)
        out = self.tail(x)

        return out, x_mid

import pretrainedmodels

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = pretrainedmodels.resnet18()
        self.a = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.b = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
        )
        self.c = resnet.layer2
        self.d = resnet.layer3

    def forward(self, input):
        x = self.a(input)
        layer1 = self.b(x) # 64
        layer2 = self.c(layer1) # 128
        out = self.d(layer2)
        return out, layer1, layer2

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, feature_channels=None, norm=nn.InstanceNorm2d) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = output_channels

        if norm is None:
            norm1 = nn.Sequential()
            norm2 = nn.Sequential()
        else:
            norm1 = norm(feature_channels)
            norm2 = norm(output_channels)

        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, input_channels, kernel_size=3, padding=1),
            norm2,
        )

        self.bottom = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.seq(input)
        return self.bottom(input + out)

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ResNet18()
        self.bottom = ResNetBlock(256, 1024, 512)

        self.mid_conv = DehazeBlock(default_conv, 1024, 3)

        self.up_block1 = nn.Sequential(
            #general.ResNetBlock(1024, 1024),
            nn.PixelShuffle(2),
        )
        self.up_block2 = nn.Sequential(
            #general.ResNetBlock(768, 768),
            nn.PixelShuffle(2),
        )
        self.up_block3 = nn.Sequential(
            # nn.Conv2d(448, 448, 1),
            nn.PixelShuffle(2),
        )
        self.up_block4 = nn.Sequential(
            #nn.Conv2d(112, 112, 1),
            nn.PixelShuffle(2),
        )
        self.attention1 = DehazeBlock(default_conv, 256, 3)
        self.attention2 = DehazeBlock(default_conv, 96, 3)
        self.enhancer = Enhancer(10, 28)
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(28, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        x, x_layer1, x_layer2 = self.encoder(input)

        x_mid = self.mid_conv(self.bottom(x))

        x = self.up_block1(x_mid)
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)
        x = self.up_block2(x)
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), 1)
        x = self.up_block3(x)
        x = self.up_block4(x)

        x = self.enhancer(x)
        out = self.tail(x)

        return out, x_mid

class WAB(nn.Module):
    def __init__(self,n_feats,expand=4):
        super(WAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * expand,3,1,1, bias=True),
            nn.BatchNorm2d(n_feats * expand),
            nn.ReLU(True),
            nn.Conv2d(n_feats* expand, n_feats , 3, 1, 1, bias=True),
            nn.BatchNorm2d(n_feats)
        )

    def forward(self, x):
        res = self.body(x).mul(0.2)+x
        return res

class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

def DFN(output_channels=13):
    return nn.Sequential(
            nn.Conv2d(3,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            invPixelShuffle(4),
            nn.Conv2d(256,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            nn.Sequential(*[WAB(16) for _ in range(3)]),
            nn.Conv2d(16, 256, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, output_channels, 3, 1, 1, bias=True)
        )

class DehazeSR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        layer1_channel = 256
        layer2_channel = 512
        out_channel = 1024

        res2net101 = Pre_Res2Net.Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

        self.mid_conv = DehazeBlock(default_conv, out_channel, 3)

        self.up_block1 = nn.PixelShuffle(2)
        self.up_block2 = nn.PixelShuffle(2)
        self.up_block3 = nn.PixelShuffle(2)
        self.up_block4 = nn.PixelShuffle(2)

        a1_channel = out_channel // 4
        self.attention1 = DehazeBlock(default_conv, a1_channel, 3)

        a2_channel = (a1_channel + layer2_channel) // 4

        self.attention2 = DehazeBlock(default_conv, a2_channel, 3)

        sr = 4
        self.dfn = DFN(sr)

        enhancer_channel = (a2_channel + layer1_channel) // 16
        enhancer_out = 28
        self.enhancer = Enhancer(enhancer_channel, enhancer_out)

        tail_in = enhancer_out + sr

        kernel_size = 7
        tail_channel = 3

        self.tail = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(tail_in, tail_channel, kernel_size=kernel_size, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        x16, x_layer1, x_layer2 = self.encoder(input)

        x16 = self.mid_conv(x16)

        x8 = self.up_block1(x16)
        x8 = self.attention1(x8)

        x8 = torch.cat((x8, x_layer2), 1)
        x4 = self.up_block2(x8)
        x4 = self.attention2(x4)

        x4 = torch.cat((x4, x_layer1), 1)
        x2 = self.up_block3(x4)

        x = self.up_block4(x2)

        x = self.enhancer(x)
        x = torch.cat([x, self.dfn(input)], dim=1)

        x = self.tail(x)

        return x, x16


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)[0]
