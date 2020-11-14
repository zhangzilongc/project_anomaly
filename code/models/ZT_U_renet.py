#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class MultiPrmSequential(nn.Sequential):
    # sequential中需要的层forward的时候只有一个参数输入input时是不需要的，但是如果有多个参数放入sequential后还要forward的时候
    # 就需要写这个函数，重写forward,因为默认的sequential forward是只有一个Input的
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, cat_feature):
        # self.modules会返回底层的模块,children's的模块,而self._modules和self.children一样返回浅层模块
        # self._modules的layer,modeules[layer]表示的是具体的某个值，所代表的的模块
        for module in self._modules.values():
            input = module(input, cat_feature)
        return input


def make_secat_layer(block, inplanes, planes, block_count, stride=1, senet=False):
    outplanes = planes * block.expansion
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False)),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, 16, stride, downsample, senet=senet))
    for i in range(1, block_count):
        layers.append(block(outplanes, planes, 16, senet=senet))

    return nn.Sequential(*layers)


class Selayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Selayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SECatBottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality=16, stride=1, downsample=None, senet=False):
        super(SECatBottleneckX, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False))
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if senet:
            self.selayer = Selayer(planes * self.expansion)

        self.senet = senet
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

        if self.senet:
            out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, block_num, senet=False):
        super(DecoderBlock, self).__init__()
        self.secat_layer = make_secat_layer(SECatBottleneckX, inplanes, planes // 4, block_num, senet=senet)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.secat_layer(x)
        return self.ps(out)


class Generator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                 layers=[12, 8, 5], bins=32, senet=False):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cardinality = 16

        self.conv1 = self._make_encoder_block_first(self.input_dim, 16)
        self.conv2 = self._make_encoder_block(16, 32)
        self.conv3 = self._make_encoder_block(32, 64)
        self.conv4 = self._make_encoder_block(64, 128)

        self.deconv1 = DecoderBlock(128, 4 * 128, layers[0], senet=senet)  # 16 -- 32
        self.deconv2 = DecoderBlock(128 + 64, 4 * 64, layers[1], senet=senet)  # 32 -- 64
        self.deconv3 = DecoderBlock(64 + 32, 4 * 32, layers[2], senet=senet)
        self.deconv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(32 + 16, bins, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(bins, bins, 3, 1, 1)),
        )
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inplanes, planes, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(planes, planes, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(inplanes, planes, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(planes, planes, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        out4_prime = self.deconv1(out4)
        concat_tensor = torch.cat([out4_prime, out3], 1)

        out3_prime = self.deconv2(concat_tensor)
        concat_tensor = torch.cat([out3_prime, out2], 1)

        out2_prime = self.deconv3(concat_tensor)

        concat_tensor = torch.cat([out2_prime, out1], 1)
        full_output = self.deconv4(concat_tensor)
        out_reg = self.softmax(full_output)

        return out_reg
