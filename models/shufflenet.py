"""shufflenet in pytorch



[1] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.

    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    https://arxiv.org/abs/1707.01083v2
"""

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn


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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepwiseAuxiliaryClassifier(nn.Module):
    #   Auxiliary classifier, including first an attention layer, then a bottlecneck layer,
    #   and final a fully connected layer

    def __init__(self, channel, num_classes=100, downsample=0):
        super(DeepwiseAuxiliaryClassifier, self).__init__()
        self.fc = nn.Linear(960, num_classes)
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

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv2d(
                input_channels,
                int(output_channels / 4),
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv2d(
                    input_channels,
                    int(output_channels / 4),
                    groups=groups
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )


        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 × 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, num_classes=100, groups=3):
        super().__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

        self.deepwise2 = DeepwiseAuxiliaryClassifier(channel=240, downsample=2)
        self.deepwise3 = DeepwiseAuxiliaryClassifier(channel=480, downsample=1)
        self.deepwise4 = DeepwiseAuxiliaryClassifier(channel=960, downsample=0)

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
        x = self.conv1(x)
        x = self.stage2(x)
        feature_list.append(x)
        x = self.stage3(x)
        feature_list.append(x)
        x = self.stage4(x)
        feature_list.append(x)
        x1 = self.deepwise2(feature_list[-3])
        x2 = self.deepwise3(feature_list[-2])
        x3 = self.deepwise4(feature_list[-1])
        x = self.deepwise4.fc(x3)
        feature = [x3, x2, x1]
        for index in range(len(feature)):
            feature[index] = F.normalize(feature[index], dim=1)


        if self.training:
            return x, feature
        else:
            return x

            #return [self.deepwise4(feature_list[-1]), self.deepwise3(feature_list[-2]),
             #       self.deepwise2(feature_list[-3])]


    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):
        """make shufflenet stage 

        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution 
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels, 
                    output_channels, 
                    stride=stride, 
                    stage=stage, 
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)

def shufflenet():
    return ShuffleNet([4, 8, 4])
                              



