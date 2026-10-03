import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # Initailize transformer with 8 heads and 2 layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # temporal pooling after attention
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, 256)
        out = self.transformer(x)          # (B, T, 256)
        out = out.permute(0, 2, 1)         # (B, 256, T) for pooling
        out = self.pool(out).squeeze(-1)   # (B, 256)
        return out
