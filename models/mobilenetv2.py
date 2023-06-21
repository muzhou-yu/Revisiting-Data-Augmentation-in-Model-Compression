import torch
import torch.nn as nn
import torch.nn.functional as F

ratio = 1.00
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

    def __init__(self, num_classes=100):
        super(AuxiliaryClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(320*ratio), int(1280*ratio), 1),
            nn.BatchNorm2d(int(1280*ratio)),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(int(1280*ratio), num_classes, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        #x = self.conv2(x)
        #x = x.view(x.size(0), -1)
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
        x = self.fc(x)
        return x


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


def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, int(24*ratio), 2, 6)
        self.stage3 = self._make_stage(3, int(24*ratio), int(32*ratio), 2, 6)
        self.stage4 = self._make_stage(4, int(32*ratio), int(64*ratio), 2, 6)
        self.stage5 = self._make_stage(3, int(64*ratio), int(96*ratio), 1, 6)
        self.stage6 = self._make_stage(3, int(96*ratio), int(160*ratio), 1, 6)
        self.stage7 = LinearBottleNeck(int(160*ratio), int(320*ratio), 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(int(320*ratio), int(1280*ratio), 1),
            nn.BatchNorm2d(int(1280*ratio)),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(int(1280*ratio), class_num, 1)

        self.fc1 = nn.Conv2d(int(1280 * ratio), class_num, 1)
        self.fc2 = nn.Conv2d(int(1280 * ratio), class_num, 1)
        self.fc3 = nn.Conv2d(int(1280 * ratio), class_num, 1)

        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=int(24*ratio),
                channel_out=int(32*ratio)
            ),
            SepConv(
                channel_in=int(32*ratio),
                channel_out=int(64*ratio)
            ),
            SepConv(
                channel_in=int(64*ratio),
                channel_out=int(320*ratio)
            ),
            nn.Sequential(
                nn.Conv2d(int(320*ratio), int(1280*ratio), 1),
                nn.BatchNorm2d(int(1280*ratio)),
                nn.ReLU6(inplace=True)
            ),
            nn.AdaptiveAvgPool2d(1),
        )


        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=int(32 * ratio),
                channel_out=int(64 * ratio)
            ),
            SepConv(
                channel_in=int(64 * ratio),
                channel_out=int(320 * ratio)
            ),
            nn.Sequential(
                nn.Conv2d(int(320 * ratio), int(1280 * ratio), 1),
                nn.BatchNorm2d(int(1280 * ratio)),
                nn.ReLU6(inplace=True)
            ),
            nn.AdaptiveAvgPool2d(1),
            #nn.Conv2d(int(1280 * ratio), class_num, 1)
        )

        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=int(64 * ratio),
                channel_out=int(320 * ratio)
            ),
            nn.Sequential(
                nn.Conv2d(int(320 * ratio), int(1280 * ratio), 1),
                nn.BatchNorm2d(int(1280 * ratio)),
                nn.ReLU6(inplace=True)
            ),
            nn.AdaptiveAvgPool2d(1),
            #nn.Conv2d(int(1280 * ratio), class_num, 1)
        )

        self.bn_means, self.bn_vars = [], []
        self.primary_classifier = AuxiliaryClassifier()

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
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        out1 = self.scala1(x)
        x = self.stage3(x)
        out2 = self.scala2(x)
        x = self.stage4(x)
        out3 = self.scala3(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.primary_classifier(x)
        batchsize = x.size(0)
        feature = [x.view(batchsize, -1), out3.view(batchsize, -1), out2.view(batchsize, -1), out1.view(batchsize, -1)]
        for index in range(len(feature)):
            feature[index] = F.normalize(feature[index], dim=1)

        x = self.primary_classifier.conv2(x)
        x = x.view(batchsize, -1)

        if self.training:
            ### return x, feature
            return x 
        else:
            return x


    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


def mobilenetv2():
    return MobileNetV2()
