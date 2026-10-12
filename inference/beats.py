import numpy as np


def find_ed_es(areas, clip_length, smooth_window=3):
  if smooth_window % 2 == 0:
    raise ValueError(f"smooth_window must be odd, got {smooth_window}")
  
  kernel = np.ones(smooth_window)/smooth_window
  pad = smooth_window//2
  padded = np.pad(areas, pad, mode="edge")
  smoothed = np.convolve(padded, kernel, mode="valid")
  es = np.argmin(smoothed)
  lo = max(es - (clip_length - 1), 0)
  hi = min(es + (clip_length - 1) + 1, len(smoothed))
  ed = lo + np.argmax(smoothed[lo:hi])

  if ed == es:
    raise ValueError("LV area does not change across the video, no cardiac motion found.")

  return int(ed), int(es)
