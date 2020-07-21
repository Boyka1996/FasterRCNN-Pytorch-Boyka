#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 下午3:58
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
from torch import nn


class BasicBlock(nn.Module):
    """
    用于ResNet18和ResNet34，这里是两个3*3卷积
    ResNet50、ResNet101和ResNet152残差块是1*1，3*3,1*1的三卷积结构
    """
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 shortcut_down_sample=None):
        """
        残差结构的基础模块
        :param in_channels: 输入的通道数
        :param out_channels: 输出通道数，即卷积核的数量
        :param stride: 步长，默认为1
        :param shortcut_down_sample: 捷径是否进行下采样
        """
        super(BasicBlock, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.shortcut_down_sample = shortcut_down_sample

    def forward(self, x):
        if self.shortcut_down_sample is not None:
            identity = self.shortcut_down_sample(x)
        else:
            identity = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        output = x + identity
        output = self.relu(output)
        return output
