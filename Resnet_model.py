from __future__ import print_function
# 使得我们能够手动输入命令行参数
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import os
import torch.utils.data as Data
import numpy as np


class ResidualBlock(nn.Module):
    """
    每一个ResidualBlock,需要保证输入和输出的维度不变
    所以卷积核的通道数都设置成一样
    """
    def __init__(self, in_features):
        super().__init__()

        '''
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        '''

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        """
        ResidualBlock中有跳跃连接;
        在得到第二次卷积结果时,需要加上该残差块的输入,
        再将结果进行激活,实现跳跃连接 ==> 可以避免梯度消失
        在求导时,因为有加上原始的输入x,所以梯度为: dy + 1,在1附近
        """
        # y = F.relu(self.conv1(x))
        # y = self.conv2(y)
        # return F.relu(x + y)

        return x + self.block(x)


class Net(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
    # def __init__(self):
        super().__init__()

        channels = input_shape[0]

        # Initial convolution block 初始化卷积块
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),  # 使用输入边界的反射填充输入张量。填充长度为channels。
            nn.Conv2d(channels, out_features, 7),  # 7为 kernal_size 卷积核大小
            nn.InstanceNorm2d(out_features),  # 归一化
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer  nn.Tanh()激活函数
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7)]
        self.fc1 = nn.Linear(17664, 2)
        self.tanh = nn.Tanh()
        self.model = nn.Sequential(*model)


        '''
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  #输入为RGB 3通道图像
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res_block_1 = ResidualBlock(16)
        self.res_block_2 = ResidualBlock(32)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(15488, 2)
        self.fc1 = nn.Linear(33088, 2)  # concat(a, b, 1)
        # self.fc1 = nn.Linear(244000, 2)  # concat(a, b, 1) 图片为3通道 256*256
        '''


    def forward(self, x):
        # in_size = x.size(0)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = self.res_block_1(x)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = self.res_block_2(x)
        # x = x.view(in_size, -1)
        # x = self.fc1(x)
        # return F.log_softmax(x, dim=1)

        x = self.model(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x)

'''
# 输出torch模型每一层的输出
model = Net()
print('model_summary', summary(model, input_size=(1, 512, 256), batch_size=64, device='cpu'))
'''
