"""shufflenetv2 in pytorch



[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
    )


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class AuxiliaryClassifier(nn.Module):
    #   Auxiliary classifier, including first an attention layer, then a bottlecneck layer,
    #   and final a fully connected layer

    def __init__(self, channel, num_classes=100):
        super(AuxiliaryClassifier, self).__init__()
        self.bottleneck_layer = nn.Sequential(
            nn.Conv2d(464, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(channel, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_bottleneck_layer(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.bottleneck_layer(x)
        x = self.pool(x)
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
        self.conv = nn.Sequential(
            nn.Conv2d(464, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def _make_conv_layer(self, channel):
        layer_list = []
        for i in range(self.downsample):
            layer_list.append(SepConv(channel, channel * 2))
            channel *= 2
        layer_list.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layer(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)

def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x

class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, class_num=100):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)

        self.deepwise2 = DeepwiseAuxiliaryClassifier(channel=116, downsample=2)
        self.deepwise3 = DeepwiseAuxiliaryClassifier(channel=232, downsample=1)
        self.deepwise4 = DeepwiseAuxiliaryClassifier(channel=464, downsample=0)



    def forward(self, x):
        feature_list = []
        x = self.pre(x)
        x = self.stage2(x)
        feature_list.append(x)
        x = self.stage3(x)
        feature_list.append(x)
        x = self.stage4(x)
        feature_list.append(x)

        x1 = self.deepwise2(feature_list[-3])
        x2 = self.deepwise3(feature_list[-2])
        x3 = self.deepwise4(feature_list[-1])

        feature = [x3, x2, x1]
        for index in range(len(feature)):
            feature[index] = F.normalize(feature[index], dim=1)

        x = self.deepwise4.fc(x3)
        if self.training:
            # return x, feature
            return x
        else:
            return x


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

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1
        
        return nn.Sequential(*layers)

def shufflenetv2():
    return ShuffleNetV2()





