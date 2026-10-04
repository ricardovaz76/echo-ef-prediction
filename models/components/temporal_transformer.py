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

    # x: (B, T, 256)
    # padding_mask: (B, T) bool, True for padded frames, or None if every frame is real
    def forward(self, x, padding_mask=None):
        T = x.shape[1]
        if T > self.pos_embedding.shape[1]:
            raise ValueError(f"Sequence of {T} frames is longer than the positional embedding ({self.pos_embedding.shape[1]})")

        x = x + self.pos_embedding[:, :T]  # (B, T, 256)

        # Padded frames are repeats of the last real frame, so they are hidden from
        # attention and left out of the average to keep short videos from over-weighting it
        out = self.transformer(x, src_key_padding_mask=padding_mask)   # (B, T, 256)

        # temporal pooling: average over the real frames
        if padding_mask is None:
            return out.mean(dim=1)                                      # (B, 256)

        keep = (~padding_mask).unsqueeze(-1).to(out.dtype)              # (B, T, 1)
        return (out * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)   # (B, 256)
