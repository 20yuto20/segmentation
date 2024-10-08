import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
from numpy import ndarray
from typing import NamedTuple


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_rate=0.3, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.in_is_out = in_ch == out_ch and stride == 1
        self.drop_rate = drop_rate
        self.shortcut = (
            nn.Sequential() if self.in_is_out else nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=stride, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=1, bias=False)

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=True)
        h = self.c1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = F.dropout(h, p=self.drop_rate, training=self.training)
        h = self.c2(h)
        return h + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, cfg, n_blocks, width_coef):
        super(WideResNet, self).__init__()

        self.num_of_class = cfg.dataset.n_class
        self.n_blocks = n_blocks
        self.n_chs = [16, 16 * width_coef[0], 32 * width_coef[1], 64 * width_coef[2]]

        self.conv1 = nn.Conv2d(3, self.n_chs[0], 3, padding=1, bias=False)
        self.conv2 = self._add_groups(self.n_blocks[0], self.n_chs[0], self.n_chs[1], cfg.network.wide_resnet.dropout_rate1)
        self.conv3 = self._add_groups(
            self.n_blocks[1],
            self.n_chs[1],
            self.n_chs[2],
            cfg.network.wide_resnet.dropout_rate2,
            stride=2,
        )
        self.conv4 = self._add_groups(
            self.n_blocks[2],
            self.n_chs[2],
            self.n_chs[3],
            cfg.network.wide_resnet.dropout_rate3,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(self.n_chs[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.full_conn = nn.Linear(self.n_chs[3], self.num_of_class)

        for m in self.modules():  # initialize weights?
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, h):
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = F.relu(self.bn(h), inplace=True)
        # h = F.avg_pool2d(h, 8)
        h = self.avgpool(h)
        h = h.view(-1, self.n_chs[3])
        h = self.full_conn(h)

        return F.log_softmax(h, dim=1)

    def _add_groups(self, n_blocks, in_ch, out_ch, drop_rate, stride=1):
        blocks = []

        for _ in range(int(n_blocks)):
            blocks.append(BasicBlock(in_ch, out_ch, stride=stride, drop_rate=drop_rate))

            in_ch, stride = out_ch, 1

        return nn.Sequential(*blocks)

