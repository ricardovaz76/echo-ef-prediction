import torch

from .metrics import dice_score, get_frame, r2_metric


# Runs the model over the test set and computes the final metrics
#
# ef_mean and ef_std are the train set EF stats used to convert predictions back to EF %
# num_visual_samples is how many samples per batch to keep for the segmentation overlays
#
# Returns:
#   metrics:     dict of mae, rmse, r2, within_5 and dice
#   all_preds:   predicted EF for every sample (EF %)
#   all_targets: ground truth EF for every sample (EF %)
#   all_samples: ED/ES frames with their predicted and ground truth masks for visualization
@torch.no_grad()
def evaluate(model, loader, device, ef_mean, ef_std, num_visual_samples=2):

    model.eval()

    total_mae      = 0.0
    total_mse      = 0.0
    total_within_5 = 0.0
    total_dice     = 0.0
    total_samples  = 0

    all_preds   = []
    all_targets = []
    all_samples = []

    for batch in loader:

        # -------------------------
        # DATA
        # -------------------------
        a4c  = batch["a4c"].to(device)
        psax = batch["psax"].to(device)

        y_reg      = batch["ef"].to(device)
        y_seg_a4c  = batch["seg_a4c"].to(device)
        y_seg_psax = batch["seg_psax"].to(device)

        ed_a4c_idx  = batch["ed_a4c"].to(device)
        es_a4c_idx  = batch["es_a4c"].to(device)
        ed_psax_idx = batch["ed_psax"].to(device)
        es_psax_idx = batch["es_psax"].to(device)

        # extract GT ED/ES frames from full seg video
        y_ed_a4c  = get_frame(y_seg_a4c,  ed_a4c_idx)
        y_es_a4c  = get_frame(y_seg_a4c,  es_a4c_idx)
        y_ed_psax = get_frame(y_seg_psax, ed_psax_idx)
        y_es_psax = get_frame(y_seg_psax, es_psax_idx)

        # -------------------------
        # FORWARD
        # -------------------------
        pred_reg, (seg_a4c_frames, seg_psax_frames) = model(a4c, psax)

        # -------------------------
        # REGRESSION METRICS
        # -------------------------
        pred_reg_raw = pred_reg.reshape(-1) * ef_std + ef_mean
        y_reg_raw    = y_reg.reshape(-1)

        batch_size   = a4c.size(0)
        diff = pred_reg_raw - y_reg_raw

        total_mae      += torch.abs(diff).mean().item()
        total_mse      += (diff ** 2).mean().item()
        total_within_5 += (torch.abs(diff) <= 5).float().mean().item()
        total_samples  += batch_size

        all_preds.append(pred_reg_raw.cpu())
        all_targets.append(y_reg_raw.cpu())

        # -------------------------
        # SEGMENTATION METRICS
        # -------------------------
        pred_ed_a4c  = get_frame(seg_a4c_frames,  ed_a4c_idx)
        pred_es_a4c  = get_frame(seg_a4c_frames,  es_a4c_idx)
        pred_ed_psax = get_frame(seg_psax_frames, ed_psax_idx)
        pred_es_psax = get_frame(seg_psax_frames, es_psax_idx)

        dice_a4c  = (dice_score(pred_ed_a4c,  y_ed_a4c)  +
                     dice_score(pred_es_a4c,  y_es_a4c))  / 2
        dice_psax = (dice_score(pred_ed_psax, y_ed_psax) +
                     dice_score(pred_es_psax, y_es_psax)) / 2

        total_dice += ((dice_a4c + dice_psax) / 2).item()

        # -------------------------
        # VISUAL SAMPLES
        # -------------------------
        for i in range(min(num_visual_samples, batch_size)):
            ed_a = ed_a4c_idx[i].item()
            es_a = es_a4c_idx[i].item()
            ed_p = ed_psax_idx[i].item()
            es_p = es_psax_idx[i].item()

            all_samples.append((
                a4c[i,  ed_a, 0].cpu(),
                a4c[i,  es_a, 0].cpu(),
                psax[i, ed_p, 0].cpu(),
                psax[i, es_p, 0].cpu(),
                pred_ed_a4c[i].cpu(),  y_ed_a4c[i].cpu(),
                pred_es_a4c[i].cpu(),  y_es_a4c[i].cpu(),
                pred_ed_psax[i].cpu(), y_ed_psax[i].cpu(),
                pred_es_psax[i].cpu(), y_es_psax[i].cpu(),
            ))

    # -------------------------
    # FINAL METRICS
    # -------------------------
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {
        "mae":      total_mae      / total_samples,
        "rmse":     (total_mse     / total_samples) ** 0.5,
        "r2":       r2_metric(all_preds, all_targets),
        "within_5": total_within_5 / total_samples,
        "dice":     total_dice     / total_samples,
    }

    return metrics, all_preds, all_targets, all_samples


def print_metrics(metrics):
    print(f"Test MAE:            {metrics['mae']:.4f}")
    print(f"Test RMSE:           {metrics['rmse']:.4f}")
    print(f"R²:                  {metrics['r2']:.4f}")
    print(f"Within ±5 EF:        {metrics['within_5']:.4f}")
    print(f"Dice (A4C + PSAX):   {metrics['dice']:.4f}")
