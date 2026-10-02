import torch
import torch.nn as nn

from .components import TemporalTransformer


# Multi-Task Model Setup
class EchoNetModel(nn.Module):
    def __init__(self, seg_model, ef_head, feature_dim):
        super().__init__()

        # -------------------------
        # components
        # -------------------------
        self.ef_head = ef_head
        self.encoder = seg_model.encoder
        self.seg_model = seg_model

        # -------------------------
        # temporal transformer
        # -------------------------
        self.temporal_transformer = TemporalTransformer(
            d_model=feature_dim,
            nhead=8,
            num_layers=2
        )
        # -------------------------
        # temporal conv
        # -------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    # =========================================================
    # FORWARD
    # =========================================================
    def forward(self, a4c, psax):

        B, T_a, C, H, W = a4c.shape
        _, T_p, _, _, _ = psax.shape

        a4c_flat  = a4c.view(B * T_a, C, H, W)
        psax_flat = psax.view(B * T_p, C, H, W)

        # -------------------------
        # segmentation
        # -------------------------

        # encode once, reuse for both seg and regression
        f1_a4c, f2_a4c, f3_a4c   = self.encoder(a4c_flat)
        f1_psax, f2_psax, f3_psax = self.encoder(psax_flat)

        # segmentation: decoder uses the shared encoder features
        seg_a4c  = self.seg_model.decoder(f3_a4c,  f2_a4c,  f1_a4c).view(B, T_a, 1, H, W)
        seg_psax = self.seg_model.decoder(f3_psax, f2_psax, f1_psax).view(B, T_p, 1, H, W)

        # regression features: f3 is the deepest feature (7x7)
        feat_a4c  = f3_a4c.mean(dim=[-1, -2]).view(B, T_a, -1)   # (B, T, 256)
        feat_psax = f3_psax.mean(dim=[-1, -2]).view(B, T_p, -1)

        # -----------------------------
        # ED and ES feature extraction
        # -----------------------------

        # temporal conv for both the a4c and psax view
        feat_a4c  = feat_a4c.permute(0, 2, 1)
        feat_a4c  = feat_a4c + self.temporal_conv(feat_a4c)
        feat_a4c  = feat_a4c.permute(0, 2, 1)

        feat_psax = feat_psax.permute(0, 2, 1)
        feat_psax = feat_psax + self.temporal_conv(feat_psax)
        feat_psax = feat_psax.permute(0, 2, 1)

        # EF Proxy Signals
        a4c_areas  = seg_a4c.detach().squeeze(2).sum(dim=[-1, -2])
        psax_areas = seg_psax.detach().squeeze(2).sum(dim=[-1, -2])

        # ED frame: largest mask area
        a4c_ed_area  = a4c_areas.max(dim=1).values
        a4c_es_area  = a4c_areas.masked_fill(a4c_areas == 0, float('inf')).min(dim=1).values

        # ES frame: smallest non-zero mask area
        psax_ed_area = psax_areas.max(dim=1).values
        psax_es_area = psax_areas.masked_fill(psax_areas == 0, float('inf')).min(dim=1).values

        # Geometry-Based Ejection Fraction for a4c and psax masks
        # This formula is meant to mimic the Ejection Fraction Formula
        # But since we are working with 2D frames and areas instead of volumes
        # The formula has been made for 2D frames
        ef_proxy_a4c  = (1 - (a4c_es_area  / (a4c_ed_area  + 1e-6))).clamp(0, 1)
        ef_proxy_psax = (1 - (psax_es_area / (psax_ed_area + 1e-6))).clamp(0, 1)

        # temporal transformation
        a4c_feat  = self.temporal_transformer(feat_a4c)
        psax_feat = self.temporal_transformer(feat_psax)

        # -------------------------
        # regression
        # -------------------------

        # fusion
        fused = torch.cat([
            a4c_feat,
            psax_feat,
            ef_proxy_a4c.unsqueeze(1),
            ef_proxy_psax.unsqueeze(1),
        ], dim=1)

        # predict EF

        ef = self.ef_head(fused)

        return ef, (seg_a4c, seg_psax)
