import os

import cv2
import numpy as np

from .clips import CLIP_LENGTH, clip_start, extract_clip, fits_in_clip, nearest_position
from .masks import build_frame_dict, get_ed_es_by_area


# Videos are recorded anywhere from 25 to 135 FPS, so they are resampled to one
# frame rate so a fixed number of frames always covers the same amount of time
TARGET_FPS = 32


# Returns the frames (T, 112, 112) and the video's frame rate
def load_video_frames(video_path):
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.resize(frame, (112, 112))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames.append(frame)

  cap.release()

  # Return an empty placeholder if no frames were grabbed
  if len(frames) == 0:
    return np.zeros((0, 112, 112), dtype=np.uint8), fps

  # Kept as raw uint8 pixels (0-255) to save 4x disk space over float32,
  # normalized to [0.0, 1.0] when loaded in EchoDataset
  return np.array(frames, dtype=np.uint8), fps    # (T, 112, 112)


# Resamples frames from fps to target_fps by picking the nearest original frame for
# each output time step (drops frames for faster videos, repeats them for slower ones)
#
# Returns the resampled frames and src_idx, the original frame index of each output
# frame, which is used to map annotated frames onto the resampled video
def resample_frames(frames, fps, target_fps=TARGET_FPS):
  T = len(frames)

  # Nothing to resample, or the video doesn't report a usable frame rate
  if T == 0 or fps <= 0:
    return frames, np.arange(T)

  n_out = max(1, int(round(T * target_fps / fps)))
  src_idx = np.round(np.arange(n_out) * fps / target_fps).astype(int)
  src_idx = np.clip(src_idx, 0, T - 1)

  return frames[src_idx], src_idx


# Builds the training clip for one video from its annotated ED and ES frames
#
# Resampling can drop the exact annotated frames, which would leave their masks on
# a neighbouring frame that doesn't match the image. So the closest resampled frame
# is swapped for the exact annotated frame (shifting its timing by at most half a
# resampled step) to keep every mask aligned with the image it was traced on.
#
# frames: all original frames (T, H, W), ed/es: original 0-based frame indices
#
# Returns the clip (clip_length, H, W), clip_src, the original frame index of each
# clip frame, and n_valid, the number of real (unpadded) frames, or None if ED and
# ES are too far apart to fit in one clip
def build_training_clip(frames, fps, ed, es, clip_length=CLIP_LENGTH, target_fps=TARGET_FPS):
  rs_frames, src_idx = resample_frames(frames, fps, target_fps)
  rs_frames, src_idx = rs_frames.copy(), src_idx.copy()

  # Swap in the exact annotated frames, earliest first so they stay in time order
  first, second = sorted((ed, es))
  pos_first = nearest_position(src_idx, first)
  pos_second = nearest_position(src_idx, second)

  # At high frame rates both can map to the same resampled frame, so the later
  # one takes the next slot (or the earlier one the previous slot at the end)
  if pos_second == pos_first:
    if pos_first + 1 < len(src_idx):
      pos_second = pos_first + 1
    elif pos_first > 0:
      pos_first = pos_first - 1
    else:
      return None

  for pos, frame in ((pos_first, first), (pos_second, second)):
    src_idx[pos] = frame
    rs_frames[pos] = frames[frame]

  if ed == first:
    ed_pos = pos_first
    es_pos = pos_second
  else:
    ed_pos = pos_second
    es_pos = pos_first

  # Skip video if ed and es frames don't fit within the clip length constraints
  if not fits_in_clip(ed_pos, es_pos, clip_length):
    return None

  start = clip_start(ed_pos, es_pos, len(src_idx), clip_length)
  return extract_clip(rs_frames, src_idx, start, clip_length)


# Where a video's saved clip lives, shared by extraction and EchoDataset
def clip_path(save_dir, fname):
  return os.path.join(save_dir, fname.replace(".avi", ".npz"))


# Saves the training clip of every video in a view so training can load clips
# directly instead of decoding videos every epoch
#
# Each .npz holds frames (clip_length, 112, 112) uint8, clip_src, the original
# frame index of each clip frame, which EchoDataset uses to place the masks, and
# n_valid, the number of real frames before any padding.
# Videos without both an ED and ES tracing, or with ED and ES too far apart to
# fit in one clip, are not saved.
def preprocess_view(df, video_dir, save_dir, patient_ids, mask_dict, clip_length=CLIP_LENGTH, target_fps=TARGET_FPS):
    os.makedirs(save_dir, exist_ok=True)

    # Clear clips from a previous extraction, so a video skipped this time can't
    # leave behind a stale clip made with different settings
    for old in os.listdir(save_dir):
        if old.endswith(".npz"):
            os.remove(os.path.join(save_dir, old))

    frame_dict = build_frame_dict(mask_dict)
    fname_to_pid = df.drop_duplicates(subset="FileName").set_index("FileName")["patient_id"].to_dict()

    saved = skipped = 0
    for filename in df["FileName"].unique():

        if fname_to_pid[filename] not in patient_ids:
            continue

        frames, fps = load_video_frames(os.path.join(video_dir, filename))

        ed, es = get_ed_es_by_area(mask_dict, filename, frame_dict[filename], max_frame=len(frames))
        if ed is None or ed == es:
            skipped += 1
            continue

        clip = build_training_clip(frames, fps, ed, es, clip_length, target_fps)
        if clip is None:
            skipped += 1
            continue

        clip_frames, clip_src, n_valid = clip
        np.savez(clip_path(save_dir, filename), frames=clip_frames, clip_src=clip_src, n_valid=n_valid)
        saved += 1

    print(f"{save_dir}: saved {saved} clips, skipped {skipped} videos")
