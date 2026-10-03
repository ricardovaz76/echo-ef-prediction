import torch
import torch.nn.functional as F


# Weighted Smooth L1 loss is used for regression to put more emphasis on the severe
# and borderline cases which are more clinically important to predict accurately
#
# ef_mean and ef_std are the train set EF stats, used to move the 35/50 EF cutoffs
# into the same normalized space as the targets
def weighted_smooth_l1(pred, target, ef_mean, ef_std):
    base_loss = F.smooth_l1_loss(pred, target, reduction="none")

    severe     = (35 - ef_mean) / ef_std
    borderline = (50 - ef_mean) / ef_std

    weight = torch.ones_like(target)
    weight = torch.where(target < severe,
                         torch.full_like(target, 2.0), weight)
    weight = torch.where((target >= severe) & (target < borderline),
                         torch.full_like(target, 1.5), weight)

    return (weight * base_loss).mean()


# Dice loss is used for segmentation tasks to directly optimize the overlap
# between the predicted segmentation mask and the ground truth mask. It is particularly
# effective for this case because the heart region is often small compared to the overall image.
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (2 * intersection + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )

    return 1 - dice.mean()


# Focal loss is used to handle class imbalance in segmentation masks
# The loss focuses more on the hard examples which is the heart region in this case
# while downweighting the easy examples which is the background
# This helps ensure the heart is correctly targeted during training despite being a small portion
# of the image.
def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(5.0, device=logits.device)
    )

    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma

    # alpha weights foreground vs background separately
    alpha_weight = targets * alpha + (1 - targets) * (1 - alpha)

    loss = alpha_weight * focal_weight * bce
    return loss.mean()


# Combines the focal loss and dice loss for segmentation to leverage their complementary strengths
def compute_seg_loss(pred, target):
    B, T = pred.shape[:2]

    pred = pred.view(B * T, 1, 112, 112)
    target = target.view(B * T, 1, 112, 112)

    valid = target.view(B * T, -1).sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred = pred[valid]
    target = target[valid]

    focal = focal_loss_with_logits(pred, target)
    dice = dice_loss(pred, target)

    return focal + dice


# Combines the regression loss and segmentation loss into a single loss function for training
def compute_loss(pred_reg, pred_seg_pair, y_reg, y_seg_pair, pred_ed_es=None, gt_ed_es=None, seg_enable=True, *, ef_mean, ef_std):

    pred_seg_a4c, pred_seg_psax = pred_seg_pair
    y_seg_a4c, y_seg_psax = y_seg_pair

    # -------------------------
    # REGRESSION
    # -------------------------
    loss_reg = weighted_smooth_l1(
        pred_reg.view(-1),
        y_reg.view(-1),
        ef_mean,
        ef_std
    )

    # -------------------------
    # FULL VIDEO SEGMENTATION LOSS
    # -------------------------
    # This segmentation loss is computed across the entire video but since there are only 2 annotated frames,
    # the loss is really computed on those 2 frames but the gradients can flow through the entire video which allows
    # the model to learn from the unannotated frames as well.

    loss_seg = (
        compute_seg_loss(pred_seg_a4c, y_seg_a4c) +
        compute_seg_loss(pred_seg_psax, y_seg_psax)
    ) * 0.5

    # -------------------------
    # ED/ES AUXILIARY LOSS
    # -------------------------
    # This auxiliary segmentation loss is computed only on the ED and ES frames from both the predicted segmentation masks and the
    # ground truth segmentation masks, this loss is used to provide a stronger and more direct supervision on the signal from the ED
    # and ES frames which are clinicly the most important frames for the EF regression tasks. Without it, the regression task might have to
    # rely on the weak supervision from the full video.

    loss_ed_es = torch.tensor(0.0, device=pred_reg.device)

    if pred_ed_es is not None and gt_ed_es is not None:

        pred_ed_a4c, pred_es_a4c, pred_ed_psax, pred_es_psax = pred_ed_es
        gt_ed_a4c, gt_es_a4c, gt_ed_psax, gt_es_psax = gt_ed_es

        loss_ed_es = (
            compute_seg_loss(pred_ed_a4c.unsqueeze(1), gt_ed_a4c.unsqueeze(1)) +
            compute_seg_loss(pred_es_a4c.unsqueeze(1), gt_es_a4c.unsqueeze(1)) +
            compute_seg_loss(pred_ed_psax.unsqueeze(1), gt_ed_psax.unsqueeze(1)) +
            compute_seg_loss(pred_es_psax.unsqueeze(1), gt_es_psax.unsqueeze(1))
        ) / 4

        loss_seg = loss_seg + 0.5 * loss_ed_es

    # -------------------------
    # FINAL COMBINATION
    # -------------------------
    # Combines the regression loss and the segmentation loss. The segmentation loss is removed from the total loss once fine tuning starts
    # to allow the model to focus solely on optimizing the regression performance while the segmentation is frozen.

    if not seg_enable:
        loss_seg = torch.tensor(0.0, device=pred_reg.device)

    total = loss_reg + 1.0 * loss_seg

    return total, loss_reg, loss_seg
