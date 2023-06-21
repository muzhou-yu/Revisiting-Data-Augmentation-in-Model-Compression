"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class MainClassifier(nn.Module):
    def __init__(self, channel, num_classes=100):
        super(MainClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AuxiliaryClassifier(nn.Module):
    #   Auxiliary classifier, including first an attention layer, then a bottlecneck layer,
    #   and final a fully connected layer

    def __init__(self, channel, num_classes=100):
        super(AuxiliaryClassifier, self).__init__()
        self.bottleneck_layer = self._make_bottleneck_layer(channel)
        self.fc = nn.Linear(channel, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_bottleneck_layer(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=2, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.bottleneck_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepwiseAuxiliaryClassifier(nn.Module):
    #   Auxiliary classifier, including first an attention layer, then a bottlecneck layer,
    #   and final a fully connected layer

    def __init__(self, channel, num_classes=100, downsample=0):
        super(DeepwiseAuxiliaryClassifier, self).__init__()
        self.fc = nn.Linear(1024, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.downsample = downsample
        self.layer = self._make_conv_layer(channel)

    def _make_conv_layer(self, channel):
        layer_list = []
        for i in range(self.downsample):
            layer_list.append(SepConv(channel, channel*2))
            channel *= 2
        layer_list.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):


    def __init__(self, width_multiplier=1, class_num=100):
       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )
       self.deepwise1 = DeepwiseAuxiliaryClassifier(channel=128, downsample=3)
       self.deepwise2 = DeepwiseAuxiliaryClassifier(channel=256, downsample=2)
       self.deepwise3 = DeepwiseAuxiliaryClassifier(channel=512, downsample=1)
       self.deepwise4 = DeepwiseAuxiliaryClassifier(channel=1024, downsample=0)

       self.bn_means, self.bn_vars = [], []

    def load_bn(self):
        index = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.bn_means[index].clone()
                m.running_var.data = self.bn_vars[index].clone()
                index += 1
        self.bn_vars = []
        self.bn_means = []

    def record_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_means.append(m.running_mean.clone())
                self.bn_vars.append(m.running_var.clone())

    def forward(self, x):
        feature_list = []
        x = self.stem(x)
        x = self.conv1(x)
        feature_list.append(x)
        x = self.conv2(x)
        feature_list.append(x)
        x = self.conv3(x)
        feature_list.append(x)
        x = self.conv4(x)
        feature_list.append(x)
        x1 = self.deepwise1(feature_list[-4])
        x2 = self.deepwise2(feature_list[-3])
        x3 = self.deepwise3(feature_list[-2])
        x4 = self.deepwise4(feature_list[-1])

        feature = [x4, x3, x2, x1]
        x = self.deepwise4.fc(x4)
        for index in range(len(feature)):
            feature[index] = F.normalize(feature[index], dim=1)

        if self.training:
            return x, feature
        else:
            return x

def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)

