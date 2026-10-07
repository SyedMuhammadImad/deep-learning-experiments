"""Retained plain and residual convolution blocks; datasets and weights remain local."""
import torch
from torch import nn

class PlainBlock(nn.Module):
    """Two conv layers, NO skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class PlainNet(nn.Module):

    def __init__(self, num_blocks=3, channels=32, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.body = nn.Sequential(*[PlainBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, num_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))

class ResBlock(nn.Module):
    """Two conv layers WITH a residual (skip) connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)

class MiniResNet(nn.Module):

    def __init__(self, num_blocks=3, channels=32, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, num_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))
