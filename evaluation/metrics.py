import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#========================
# REGRESSION METRICS
#========================
def regression_metrics(pred, target):

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    mae = mean_absolute_error(target, pred)
    mse = mean_squared_error(target, pred)
    rmse = np.sqrt(mse)

    within_5 = np.mean(np.abs(pred - target) <= 5)

    return mae, mse, rmse, within_5


def r2_metric(preds, targets):
    return r2_score(targets, preds)


#========================
# SEGMENTATION METRICS
#========================
def dice_score(pred, target, eps=1e-6):

    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (2 * intersection + eps) / (
        pred.sum(dim=1) + target.sum(dim=1) + eps
    )

    return dice.mean()


# Computes mean Dice score over annotated frames only (frames with a ground truth mask).
# Used during validation to monitor segmentation quality across the full video.
def dice_over_valid_frames(pred_seg, gt_seg, eps=1e-6):
    B, T = pred_seg.shape[:2]
    pred = torch.sigmoid(pred_seg).view(B * T, -1)
    gt   = gt_seg.view(B * T, -1)

    valid = gt.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred  = pred[valid]
    gt    = gt[valid]

    intersection = (pred * gt).sum(dim=1)
    dice = (2 * intersection + eps) / (pred.sum(dim=1) + gt.sum(dim=1) + eps)
    return dice.mean()


#================================================================
# Extract the ED and ES frame from the segmentation Predictions
#================================================================
def get_frame(seg, idx):
    B = seg.shape[0]
    idx = idx.long()

    idx = idx.clamp(0, seg.shape[1] - 1)
    batch_idx = torch.arange(B, device=seg.device)
    return seg[batch_idx, idx]  # (B, 1, H, W)
