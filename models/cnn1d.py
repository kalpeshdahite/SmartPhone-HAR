import torch.nn as nn


class CNN1D(nn.Module):
    """Small 1D CNN for [batch, 6, 150] windows. encoder -> feat_dim vector -> classifier."""

    def __init__(self, in_channels=6, num_classes=5, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))