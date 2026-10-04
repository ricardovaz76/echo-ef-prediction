import torch.nn as nn


# Captures short-range motion (is the ventricle contracting or relaxing right now)
# that a single frame can't show. Two kernel-3 convs see ~2 frames on each side.
#
# Subclasses nn.Sequential so the conv weights keep the same names in checkpoints
class TemporalConv(nn.Sequential):
    def __init__(self, feature_dim=256, kernel_size=3):
        # No ReLU after the last conv: its output is added back as a residual, so it
        # needs to be able to lower features as well as raise them
        super().__init__(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    # x: (B, T, feature_dim)
    def forward(self, x):
        # Conv1d slides over the last dim, hence the permute to (B, feature_dim, T) and back.
        # Residual: the conv output is added on top of the original per-frame features.
        x = x.permute(0, 2, 1)
        x = x + super().forward(x)
        return x.permute(0, 2, 1)
