import torch
import torch.nn as nn

from .components import TemporalTransformer


# (B, T) bool mask that is True for padded frames, the positions at or past each clip's
# number of real frames. None when no frame counts are given (every frame is real).
def padding_mask(valid, T):
    if valid is None:
        return None
    positions = torch.arange(T, device=valid.device)
    return positions.unsqueeze(0) >= valid.unsqueeze(1)


# Dual-view multi-task model
#
# Predicts ejection fraction from an A4C and a PSAX video of the same visit.
# Segmentation is trained alongside EF so the shared encoder learns where the
# left ventricle is, which gives the EF prediction more meaningful features
# than learning from EF labels alone.
class EchoNetModel(nn.Module):
    def __init__(self, seg_model, ef_head, feature_dim):
        super().__init__()

        # -------------------------
        # components
        # -------------------------
        self.ef_head = ef_head

        # Shared with seg_model (same module, not a copy), so what the encoder learns
        # from segmentation directly benefits EF regression
        self.encoder = seg_model.encoder
        self.seg_model = seg_model

        # -------------------------
        # temporal transformer
        # -------------------------
        # EF depends on how the heart changes over the whole cardiac cycle, so this
        # lets every frame attend to every other frame before summarizing the video
        self.temporal_transformer = TemporalTransformer(
            d_model=feature_dim,
            nhead=8,
            num_layers=2
        )
        # -------------------------
        # temporal conv
        # -------------------------
        # Captures short-range motion (is the ventricle contracting or relaxing right now)
        # that a single frame can't show. Two kernel-3 convs see ~2 frames on each side.
        # No ReLU after the last conv: its output is added back as a residual, so it
        # needs to be able to lower features as well as raise them
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        )

    # =========================================================
    # FORWARD
    # =========================================================
    # a4c: (B, T_a, 1, H, W), psax: (B, T_p, 1, H, W)
    # The views can have different frame counts since they are separate recordings
    # valid_a4c, valid_psax: (B,) number of real frames in each clip, the rest is
    # padding; None if every frame is real
    def forward(self, a4c, psax, valid_a4c=None, valid_psax=None):

        B, T_a, C, H, W = a4c.shape
        _, T_p, _, _, _ = psax.shape

        # The encoder is a 2D CNN, so frames are folded into the batch dimension
        # to run every frame through it in one pass
        a4c_flat  = a4c.view(B * T_a, C, H, W)
        psax_flat = psax.view(B * T_p, C, H, W)

        # -------------------------
        # encoding + segmentation
        # -------------------------

        # Encode once and reuse the features for both tasks to avoid a second, costly encoder pass
        f1_a4c, f2_a4c, f3_a4c   = self.encoder(a4c_flat)
        f1_psax, f2_psax, f3_psax = self.encoder(psax_flat)

        # Left ventricle masks for every frame, returned as raw logits for the segmentation loss
        seg_a4c  = self.seg_model.decoder(f3_a4c,  f2_a4c,  f1_a4c).view(B, T_a, 1, H, W)
        seg_psax = self.seg_model.decoder(f3_psax, f2_psax, f1_psax).view(B, T_p, 1, H, W)

        # Regression only needs *what* is in each frame, not *where*, so the deepest
        # feature map (f3, 7x7) is averaged over space into one 256-d vector per frame
        feat_a4c  = f3_a4c.mean(dim=[-1, -2]).view(B, T_a, -1)   # (B, T, 256)
        feat_psax = f3_psax.mean(dim=[-1, -2]).view(B, T_p, -1)

        # -----------------------------
        # local temporal context
        # -----------------------------

        # Adds short-range motion to each frame's features. The residual keeps the
        # original frame appearance intact so the conv only has to learn the motion on top.
        # Conv1d slides over the last dim, hence the permute to (B, 256, T) and back.
        feat_a4c  = feat_a4c.permute(0, 2, 1)
        feat_a4c  = feat_a4c + self.temporal_conv(feat_a4c)
        feat_a4c  = feat_a4c.permute(0, 2, 1)

        feat_psax = feat_psax.permute(0, 2, 1)
        feat_psax = feat_psax + self.temporal_conv(feat_psax)
        feat_psax = feat_psax.permute(0, 2, 1)

        # -----------------------------
        # EF proxy signals
        # -----------------------------
        # Gives the regression head an explicit, geometry-based hint: EF is how much the
        # ventricle shrinks from its largest (ED) to its smallest (ES) size, which can be
        # approximated from the predicted mask areas. Detached so the proxy can't pull
        # the segmentation toward whatever makes EF easier instead of accurate masks.
        #
        # The decoder outputs logits, so sigmoid turns them into per-pixel probabilities
        # whose sum is the soft mask area
        a4c_areas  = torch.sigmoid(seg_a4c.detach()).squeeze(2).sum(dim=[-1, -2])     # (B, T_a)
        psax_areas = torch.sigmoid(seg_psax.detach()).squeeze(2).sum(dim=[-1, -2])    # (B, T_p)

        # ED (end-diastole, ventricle fully filled) = largest area
        # ES (end-systole, ventricle fully contracted) = smallest non-zero area
        a4c_ed_area  = a4c_areas.max(dim=1).values
        a4c_es_area  = a4c_areas.masked_fill(a4c_areas == 0, float('inf')).min(dim=1).values

        psax_ed_area = psax_areas.max(dim=1).values
        psax_es_area = psax_areas.masked_fill(psax_areas == 0, float('inf')).min(dim=1).values

        # 2D version of EF = 1 - ESV / EDV, using areas as a stand-in for volumes
        ef_proxy_a4c  = (1 - (a4c_es_area  / (a4c_ed_area  + 1e-6))).clamp(0, 1)
        ef_proxy_psax = (1 - (psax_es_area / (psax_ed_area + 1e-6))).clamp(0, 1)

        # -----------------------------
        # global temporal context
        # -----------------------------
        # Summarizes each video into a single vector that reflects the full cardiac cycle
        a4c_feat  = self.temporal_transformer(feat_a4c,  padding_mask(valid_a4c,  T_a))   # (B, 256)
        psax_feat = self.temporal_transformer(feat_psax, padding_mask(valid_psax, T_p))   # (B, 256)

        # -------------------------
        # regression
        # -------------------------

        # Combine both views since each sees the ventricle from a different angle
        # 256 (A4C) + 256 (PSAX) + 1 + 1 (EF proxies) = 514
        fused = torch.cat([
            a4c_feat,
            psax_feat,
            ef_proxy_a4c.unsqueeze(1),
            ef_proxy_psax.unsqueeze(1),
        ], dim=1)

        # Predicts normalized EF (trained on z-scored targets for stability),
        # convert back with ef * EF_STD + EF_MEAN
        ef = self.ef_head(fused)   # (B, 1)

        return ef, (seg_a4c, seg_psax)
