import torch

from evaluation.metrics import dice_over_valid_frames, get_frame, r2_metric, regression_metrics
from .losses import compute_loss


# ef_mean and ef_std are the train set EF stats used to normalize the EF targets
def train_one_epoch(model, loader, optimizer, device, ef_mean, ef_std, seg_enable=True):

    model.train()

    total_loss     = 0.0
    total_reg_loss = 0.0
    total_seg_loss = 0.0

    for batch_idx, batch in enumerate(loader):

        # -------------------
        # inputs
        # -------------------
        a4c  = batch["a4c"].to(device)
        psax = batch["psax"].to(device)

        # number of real frames per clip, the rest is padding
        valid_a4c  = batch["valid_a4c"].to(device)
        valid_psax = batch["valid_psax"].to(device)

        # -------------------
        # Normalize EF
        # -------------------
        y_reg = batch["ef"].to(device)
        y_reg = (y_reg - ef_mean) / (ef_std + 1e-8)

        y_seg_a4c  = batch["seg_a4c"].to(device)   # (B, T, 1, H, W)
        y_seg_psax = batch["seg_psax"].to(device)

        ed_a4c  = batch["ed_a4c"].to(device)
        es_a4c  = batch["es_a4c"].to(device)
        ed_psax = batch["ed_psax"].to(device)
        es_psax = batch["es_psax"].to(device)

        # -------------------
        # Forward Pass
        # -------------------
        pred_reg, (seg_a4c_frames, seg_psax_frames) = model(a4c, psax, valid_a4c, valid_psax)

        # -------------------
        # sample ED/ES frames
        # -------------------
        pred_ed_a4c  = get_frame(seg_a4c_frames,  ed_a4c)
        pred_es_a4c  = get_frame(seg_a4c_frames,  es_a4c)
        pred_ed_psax = get_frame(seg_psax_frames, ed_psax)
        pred_es_psax = get_frame(seg_psax_frames, es_psax)

        gt_ed_a4c  = get_frame(y_seg_a4c,  ed_a4c)
        gt_es_a4c  = get_frame(y_seg_a4c,  es_a4c)
        gt_ed_psax = get_frame(y_seg_psax, ed_psax)
        gt_es_psax = get_frame(y_seg_psax, es_psax)

        # -------------------
        # loss
        # -------------------
        loss, reg_loss, seg_loss = compute_loss(
            pred_reg,
            (seg_a4c_frames, seg_psax_frames),
            y_reg,
            (y_seg_a4c, y_seg_psax),
            pred_ed_es=(pred_ed_a4c, pred_es_a4c, pred_ed_psax, pred_es_psax),
            gt_ed_es=(gt_ed_a4c, gt_es_a4c, gt_ed_psax, gt_es_psax),
            seg_enable=seg_enable,
            ef_mean=ef_mean,
            ef_std=ef_std
        )

        # -------------------
        # Backward Pass
        # -------------------
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss     += loss.item()
        total_reg_loss += reg_loss.item()
        total_seg_loss += seg_loss.item()

    n = len(loader)
    return (
        total_loss     / n,
        total_reg_loss / n,
        total_seg_loss / n,
    )


@torch.no_grad()
def validate(model, loader, device, ef_mean, ef_std):

    model.eval()

    total_mae     = 0.0
    total_dice    = 0.0
    total_samples = 0

    all_preds   = []
    all_targets = []

    for batch in loader:

        # -------------------
        # Grab Data
        # -------------------
        a4c  = batch["a4c"].to(device)
        psax = batch["psax"].to(device)

        # number of real frames per clip, the rest is padding
        valid_a4c  = batch["valid_a4c"].to(device)
        valid_psax = batch["valid_psax"].to(device)

        y_reg      = batch["ef"].to(device)
        y_seg_a4c  = batch["seg_a4c"].to(device)
        y_seg_psax = batch["seg_psax"].to(device)

        # -------------------
        # Forward Pass
        # -------------------
        pred_reg, (seg_a4c_frames, seg_psax_frames) = model(a4c, psax, valid_a4c, valid_psax)

        # -------------------
        # MAE: denormalize pred only
        # -------------------
        pred_reg_raw = pred_reg.reshape(-1) * ef_std + ef_mean   # (B,) always
        y_reg_raw    = y_reg.reshape(-1)                         # (B,) always

        # MAE score
        mae = torch.abs(pred_reg_raw - y_reg_raw).mean()

        # Save MAE and history
        batch_size = a4c.size(0)
        total_mae += mae.item()
        all_preds.append(pred_reg_raw.cpu())
        all_targets.append(y_reg_raw.cpu())

        # -------------------
        # Dice: valid frames only
        # -------------------
        dice_a4c  = dice_over_valid_frames(seg_a4c_frames,  y_seg_a4c)
        dice_psax = dice_over_valid_frames(seg_psax_frames, y_seg_psax)
        dice      = (dice_a4c + dice_psax) / 2

        total_dice    += dice.item()
        total_samples += batch_size

    # -------------------
    # epoch-level metrics
    # -------------------
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mae, mse, rmse, within_5 = regression_metrics(all_preds, all_targets)
    r2 = r2_metric(
        all_preds.numpy(),
        all_targets.numpy()
    )

    return {
        "mae":      mae,
        "rmse":     rmse,
        "r2":       r2,
        "within_5": within_5,
        "dice":     total_dice / total_samples,
    }
