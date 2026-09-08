import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1D time-series data.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, channels, timesteps]

        batch_size, channels, _ = x.size()

        scale = self.pool(x).view(batch_size, channels)

        scale = self.fc(scale)

        scale = scale.view(batch_size, channels, 1)

        return x * scale


class ResidualSEBlock(nn.Module):
    """
    Residual 1D convolutional block with Squeeze-and-Excitation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += identity

        out = self.relu(out)

        return out


class ResNetSE5(nn.Module):
    """
    ResNet-SE-5 for smartphone Human Activity Recognition.

    Input:
        [batch, 6, 150]

    Output:
        [batch, num_classes]
    """

    def __init__(
        self,
        in_channels=6,
        num_classes=5
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = ResidualSEBlock(
            64,
            64
        )

        self.block2 = ResidualSEBlock(
            64,
            128,
            stride=2
        )

        self.block3 = ResidualSEBlock(
            128,
            128
        )

        self.block4 = ResidualSEBlock(
            128,
            256,
            stride=2
        )

        self.block5 = ResidualSEBlock(
            256,
            256
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x