import torch
import torch.nn as nn

from models import utils
from timm.models.layers import DropPath


class SimpleLandmarker(nn.Module):
    def __init__(self, channel_in, num_points=106, bottleneck_size=[2, 2]):
        super().__init__()
        self.reg_layer = nn.Sequential(*[
            nn.Conv2d(channel_in, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(128, 128, kernel_size=7, stride=3, padding=3),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * bottleneck_size[0] * bottleneck_size[1], num_points),
        ])

        self.apply(utils.init_weights)

    def forward(self, feat):
        x_reg = self.reg_layer(feat)

        return x_reg


class SimpleRegressor(nn.Module):
    def __init__(self, channel_in, num_class=1):
        super().__init__()
        self.reg_layer = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(channel_in, 256),
            nn.Linear(256, num_class),
        ])

        self.apply(utils.init_weights)

    def forward(self, feat):
        x_reg = self.reg_layer(feat)

        return x_reg


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, num_class, normalization='BatchNorm1d', activation='ReLU', dropblock=True):
        super().__init__()

        normalization = getattr(nn, normalization)
        activation = getattr(nn, activation)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            activation(),
            DropPath(0.2) if dropblock else nn.Identity(),
            normalization(in_features),
            nn.Linear(in_features, num_class),
        )

        self.apply(utils.init_weights)

    def forward(self, feat):

        return self.classifier(feat)


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10098610
# Modified GDC
class SimpleRecognizer(nn.Module):
    def __init__(self, channel_in, num_class):
        super().__init__()

        self.recognizer = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0, groups=channel_in),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(channel_in),

            nn.Linear(channel_in, num_class),
        )

        self.apply(utils.init_weights)

    def forward(self, feat):
        return self.recognizer(feat)

    def forward_embedding(self, feat):
        return self.recognizer[:-1](feat)
