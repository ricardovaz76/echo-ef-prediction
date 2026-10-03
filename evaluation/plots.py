import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve


#========================
# SEGMENTATION OVERLAY
#========================
def show_overlay(ax, frame, pred_mask, gt_mask=None, title=""):
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
    ax.imshow(pred_mask, alpha=0.3, cmap="Reds")

    if gt_mask is not None:
        ax.contour(gt_mask, colors="green", linewidths=1)

    ax.set_title(title)
    ax.axis("off")


# Saves one figure per sample with the A4C and PSAX ED/ES predictions
#
# The ground truth mask is used as the background instead of the echo frame
# so the comparison focuses on predicted mask (red) vs ground truth (green outline)
def plot_segmentation_overlays(all_samples, out_dir, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)

    for idx, sample in enumerate(all_samples[:num_samples]):
        (_, _, _, _,
         pred_ed_a4c_mask,  gt_ed_a4c_mask,
         pred_es_a4c_mask,  gt_es_a4c_mask,
         pred_ed_psax_mask, gt_ed_psax_mask,
         pred_es_psax_mask, gt_es_psax_mask) = sample

        # (pred logits, ground truth, title) for each panel
        panels = [
            (pred_ed_a4c_mask,  gt_ed_a4c_mask,  "A4C - ED"),
            (pred_es_a4c_mask,  gt_es_a4c_mask,  "A4C - ES"),
            (pred_ed_psax_mask, gt_ed_psax_mask, "PSAX - ED"),
            (pred_es_psax_mask, gt_es_psax_mask, "PSAX - ES"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (pred_logits, gt, title) in zip(axes, panels):
            # The first dimension [0] removes the channel dimension of the (1, H, W) masks
            gt_np = gt[0].numpy()

            # sigmoid + threshold masks
            pred_np = (torch.sigmoid(pred_logits[0]).numpy() > 0.5).astype(float)

            show_overlay(ax, gt_np, pred_np, gt_np, title=title)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlay_{idx:02d}.png"), bbox_inches="tight")
        plt.close(fig)


#========================
# REGRESSION PLOTS
#========================
def plot_regression(all_preds, all_targets, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # convert tensors → numpy
    y_pred = all_preds.numpy().flatten()
    y_true = all_targets.numpy().flatten()

    # Predicted vs True
    fig = plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, alpha=0.5)

    # perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("True EF")
    plt.ylabel("Predicted EF")
    plt.title("EF Regression: Predicted vs True")

    plt.axis("equal")
    fig.savefig(os.path.join(out_dir, "scatter.png"))
    plt.close(fig)

    # Error vs True EF
    fig = plt.figure()
    plt.plot(y_true, y_true - y_pred, '.')
    plt.axhline(0)
    plt.title("Error vs True EF")
    fig.savefig(os.path.join(out_dir, "error_vs_true.png"))
    plt.close(fig)

    # Error distribution
    fig = plt.figure()
    plt.hist(y_pred - y_true, bins=30)
    plt.title("EF Error Distribution")
    fig.savefig(os.path.join(out_dir, "error_distribution.png"))
    plt.close(fig)


#========================
# ROC CURVE
#========================
def plot_roc(all_preds, all_targets, path, threshold=50):
    # binarize — EF < 50 = reduced (1), >= 50 = normal (0)
    y_true  = (all_targets.numpy() < threshold).astype(int)
    y_score = 1 - (all_preds.numpy() / 100)   # higher score = more likely reduced

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — Reduced EF Detection (EF < {threshold}%)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return roc_auc


#========================
# BLAND-ALTMAN PLOT
#========================
def plot_bland_altman(all_preds, all_targets, path):
    pred_efs = all_preds.numpy()
    gt_efs   = all_targets.numpy()

    mean_ef = (pred_efs + gt_efs) / 2
    diff_ef = pred_efs - gt_efs

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(mean_ef, diff_ef, alpha=0.5)
    plt.axhline(y=diff_ef.mean(), color='red', linestyle='--', label=f"Mean bias: {diff_ef.mean():.2f}")
    plt.axhline(y=diff_ef.mean() + 1.96*diff_ef.std(), color='gray', linestyle='--', label="±1.96 SD")
    plt.axhline(y=diff_ef.mean() - 1.96*diff_ef.std(), color='gray', linestyle='--')
    plt.xlabel("Mean EF")
    plt.ylabel("Predicted - True EF")
    plt.title("Bland-Altman Plot")
    plt.legend()
    fig.savefig(path)
    plt.close(fig)
