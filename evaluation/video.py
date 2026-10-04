import os
import random

import cv2
import numpy as np
import torch


def overlay_mask(frame, mask, ef_text=None, alpha=0.4):
    frame = frame.copy()

    if len(frame.shape) == 2:
        frame = np.stack([frame]*3, axis=-1)

    # normalize mask for visibility
    mask = mask.astype(np.float32)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    color = np.zeros_like(frame)
    color[..., 0] = (mask * 255).astype(np.uint8)

    out = (frame * (1 - alpha) + color * alpha).astype(np.uint8)

    # =========================
    # ADD EF TEXT (BOTTOM LEFT)
    # =========================
    if ef_text is not None:
        font_scale = 0.4
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(
            ef_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        x = 2
        y = out.shape[0] - 5 - baseline

        cv2.rectangle(
            out,
            (x - 3, y - text_h - 3),
            (x + text_w + 3, y + baseline + 3),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            out,
            ef_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return out


def save_segmentation_video(frames, masks, pred_ef=None, path="segmentation.mp4", fps=10):

    H, W = frames.shape[1], frames.shape[2]

    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )
    ef_text = None
    if pred_ef is not None:
        ef_text = f"EF:{pred_ef:.1f}"

    for t in range(frames.shape[0]):
        frame = frames[t]
        mask = masks[t]

        frame = frame.astype(np.float32)

        if frame.max() <= 1.0:
            frame = frame * 255

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        overlay = overlay_mask(frame, mask, ef_text=ef_text)

        out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    out.release()


# Picks a test sample and saves its A4C and PSAX segmentation videos with the predicted EF
#
# idx:  which sample to render, a random one if None
# seed: set to reproduce the same random sample
#
# Returns the predicted and ground truth EF
@torch.no_grad()
def render_sample_videos(model, dataset, device, ef_mean, ef_std, out_dir, idx=None, seed=None):
    os.makedirs(out_dir, exist_ok=True)

    if idx is None:
        idx = random.Random(seed).randint(0, len(dataset) - 1)

    model.eval()

    sample = dataset[idx]

    # add the batch dimension the model expects
    a4c = sample["a4c"].unsqueeze(0).to(device)
    psax = sample["psax"].unsqueeze(0).to(device)

    # number of real frames per clip, the rest is padding
    valid_a4c = torch.tensor([sample["valid_a4c"]], device=device)
    valid_psax = torch.tensor([sample["valid_psax"]], device=device)

    pred_reg, (seg_a4c, seg_psax) = model(a4c, psax, valid_a4c, valid_psax)

    i = 0

    # =========================
    # EF (DENORMALIZE)
    # =========================
    pred_ef = pred_reg[i].item() * ef_std + ef_mean
    gt_ef = sample["ef"].item()

    print("Sample index:", idx)
    print("Predicted EF:", pred_ef)
    print("Ground Truth EF:", gt_ef)

    # =========================
    # A4C VIDEO
    # =========================
    a4c_frames = a4c[i].cpu().numpy()[:, 0]
    a4c_masks = torch.sigmoid(seg_a4c[i]).detach().cpu().numpy()[:, 0]

    save_segmentation_video(
        a4c_frames,
        a4c_masks,
        pred_ef=pred_ef,
        path=os.path.join(out_dir, "a4c_model_video.mp4")
    )

    # =========================
    # PSAX VIDEO
    # =========================
    psax_frames = psax[i].cpu().numpy()[:, 0]
    psax_masks = torch.sigmoid(seg_psax[i]).detach().cpu().numpy()[:, 0]

    save_segmentation_video(
        psax_frames,
        psax_masks,
        pred_ef=pred_ef,
        path=os.path.join(out_dir, "psax_model_video.mp4")
    )

    return pred_ef, gt_ef
