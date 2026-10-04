import numpy as np
import torch


# Runs only the segmentation part of the model (encoder + decoder) on every frame
#
# This is the first stage of inference: without tracings, ED and ES are found from
# how the predicted ventricle area changes over the whole video.
#
# frames: resampled uint8 frames (T, H, W)
# Returns per-frame mask probabilities (T, H, W) as float32 on the CPU
@torch.no_grad()
def segment_frames(model, frames, device, batch_size=64):
    model.eval()

    probs = []
    for start in range(0, len(frames), batch_size):
        # Same normalization as EchoDataset: uint8 -> [0, 1], with a channel dim
        x = torch.from_numpy(frames[start:start + batch_size]).float().div(255.0).unsqueeze(1).to(device)

        logits = model.seg_model(x)                      # (b, 1, H, W)
        probs.append(torch.sigmoid(logits)[:, 0].cpu())  # (b, H, W)

    if not probs:
        return np.zeros((0,) + frames.shape[1:], dtype=np.float32)

    return torch.cat(probs).numpy()


# Soft ventricle area per frame: the sum of mask probabilities, computed the same
# way as the EF proxy inside the model
def area_curve(probs):
    return probs.sum(axis=(1, 2))


# Time in seconds of each resampled frame in the original video, so results can be
# lined up with the original video while it plays, whatever its frame rate
def frame_times(src_idx, fps):
    if fps <= 0:
        return np.asarray(src_idx, dtype=np.float64)
    return np.asarray(src_idx, dtype=np.float64) / fps
