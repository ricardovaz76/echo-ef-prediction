import torch.nn as nn


# MLP Ejection Fraction Regression Head
class RegressionHead(nn.Module):
    # in_channels: a4c + psax temporal features (2 * 256) + 2 EF proxy signals
    def __init__(self, in_channels=514):
        super().__init__()

        # Simple MLP block
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)
