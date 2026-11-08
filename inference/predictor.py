import numpy as np
import torch
from evaluation import resolve_clip_config
from models import build_model
from data import load_video_frames, resample_frames, clip_start, extract_clip
from .beats import find_ed_es
from .video import encode_overlay_video


class EchoPredictor:
  def __init__(self, checkpoint_path, device=None):
    self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

    ef_mean, ef_std = checkpoint.get("ef_mean"), checkpoint.get("ef_std")
    if ef_mean is None or ef_std is None:
      raise ValueError(
        "This checkpoint does not store ef_mean/ef_std."
      )

    self.ef_mean = ef_mean
    self.ef_std = ef_std

    config = resolve_clip_config(checkpoint)
    self.config = config

    model = build_model(self.device, clip_length=config["clip_length"])
    model.load_state_dict(checkpoint["model_state_dict"])

    self.model = model.eval()

  def load_view(self, video_path):
    frames, fps = load_video_frames(video_path)
    if len(frames) == 0:
      raise ValueError(
        f"No frames were extracted from {video_path}"
      )

    if fps <= 0:
      raise ValueError(
        f"No fps was detected during frame extraction for {video_path}"
      )

    rs_frames, _ = resample_frames(frames, fps, self.config["target_fps"])

    return rs_frames

  def convert_to_model_input(self, frames):
    video = (torch.from_numpy(frames).to(self.device).float()/255.0).unsqueeze(1).unsqueeze(0)
    return video

  # Returns the soft LV area per frame (T,) and the masks thresholded at 0.5 (T, H, W).
  # Masks go to the CPU chunk by chunk so a long video can't fill the GPU's memory
  def segment_frames(self, frames, batch_size=64):
    seg_model = self.model.seg_model
    seg_input = self.convert_to_model_input(frames) # (B, T, C, H, W)
    seg_input = seg_input.squeeze(0) # -> (T, C, H, W) expected input for segmentations

    areas, masks = [], []
    with torch.inference_mode():
      for i in range(0, len(frames), batch_size):
        probs = torch.sigmoid(seg_model(seg_input[i:i+batch_size])).squeeze(1) # (N, H, W)
        areas.append(probs.sum(dim=[-1, -2]))
        masks.append((probs > 0.5).cpu())

    return torch.cat(areas, dim=0).cpu().numpy(), torch.cat(masks, dim=0).numpy()

  # Turns one video into the model's clip, placed around the predicted ED -> ES span like
  # training clips. Returns a dict of the clip (1, clip_length, 1, 112, 112), n_valid (1,)
  # (real frames, so the transformer ignores padding), and the whole video's frames and masks
  def prepare_view(self, video_path):
    clip_length = self.config["clip_length"]

    frames = self.load_view(video_path)
    areas, masks = self.segment_frames(frames)
    ed, es = find_ed_es(areas, clip_length)

    # extract_clip also tracks each frame's source index, which the model never
    # reads, so positions in the resampled video are enough here
    start = clip_start(ed, es, len(frames), clip_length)
    clip, _, n_valid = extract_clip(frames, np.arange(len(frames)), start, clip_length)

    clip = self.convert_to_model_input(clip)
    n_valid = torch.tensor([n_valid], device=self.device)

    return {"clip": clip, "n_valid": n_valid, "frames": frames, "masks": masks}

  def _predict_ef(self, a4c, psax):
    with torch.inference_mode():
      ef, _ = self.model(a4c["clip"], psax["clip"], a4c["n_valid"], psax["n_valid"])   # (1, 1)

    # The model was trained on z-scored EF, so undo that with the train set stats
    return ef.item() * self.ef_std + self.ef_mean

  # Predicts EF (%) from an A4C and a PSAX video of the same visit
  def predict(self, a4c_path, psax_path):
    return self._predict_ef(self.prepare_view(a4c_path), self.prepare_view(psax_path))

  # Same as predict, plus each whole video with its LV masks overlaid, as H.264 MP4 bytes
  def predict_with_videos(self, a4c_path, psax_path):
    a4c = self.prepare_view(a4c_path)
    psax = self.prepare_view(psax_path)
    fps = self.config["target_fps"]

    return {
      "ef": self._predict_ef(a4c, psax),
      "videos": {
        "a4c": encode_overlay_video(a4c["frames"], a4c["masks"], fps),
        "psax": encode_overlay_video(psax["frames"], psax["masks"], fps),
      },
    }

  # The first pass on a GPU is slow (~1.5 s) while CUDA loads its kernels, so a
  # server runs one blank clip through the model at startup instead of making
  # the first request wait
  def warm_up(self):
    clip_length = self.config["clip_length"]
    blank = torch.zeros(1, clip_length, 1, 112, 112, device=self.device)
    n_valid = torch.tensor([clip_length], device=self.device)

    with torch.inference_mode():
      self.model(blank, blank, n_valid, n_valid)

