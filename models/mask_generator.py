import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n=4, width=[64, 128, 256, 512]):
        super(Generator, self).__init__()
        self.layers = []
        for i in range(n):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(width[i], width[i], kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(width[i], 1, kernel_size=1, stride=1, padding=0),
                )
            )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, feat):
        mask_list = []
        for i, layer in enumerate(self.layers):
            batchsize, channel_num, width, height = feat[i].size(0), feat[i].size(1), feat[i].size(2), feat[i].size(3)
            mask = layer(feat[i])
            mask = mask.view(batchsize, -1)
            mask = F.softmax(mask, dim=1) * width * height
            mask = mask.view(batchsize, 1, width, height)
            mask_list.append(mask)
        return mask_list
