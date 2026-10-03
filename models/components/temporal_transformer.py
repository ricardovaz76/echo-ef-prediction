import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    # max_len: longest sequence the positional embedding covers, matches CLIP_LENGTH in data/clips.py
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.1, max_len=32):
        super().__init__()

        # Attention alone ignores frame order, so a learned vector per clip position is
        # added to each frame. This lets attention use when a frame happens, e.g. whether
        # the ventricle is shrinking toward ES or filling toward ED.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

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
        T = x.shape[1]
        if T > self.pos_embedding.shape[1]:
            raise ValueError(f"Sequence of {T} frames is longer than the positional embedding ({self.pos_embedding.shape[1]})")

        x = x + self.pos_embedding[:, :T]  # (B, T, 256)
        out = self.transformer(x)          # (B, T, 256)
        out = out.permute(0, 2, 1)         # (B, 256, T) for pooling
        out = self.pool(out).squeeze(-1)   # (B, 256)
        return out
