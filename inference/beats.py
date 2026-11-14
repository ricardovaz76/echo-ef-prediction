import numpy as np
from scipy.signal import find_peaks


# Peaks closer than this (seconds) can't be separate beats: children's heart rates reach ~200 bpm
MIN_BEAT_INTERVAL = 0.25

# Tuned on the validation split against the annotated ED/ES frames. Prominence is how far
# a peak must stand out, as a fraction of the curve's range, so noise isn't taken for a beat
SMOOTH_WINDOW = 5
MIN_PROMINENCE = 0.1


def smooth(areas, window):
  if window % 2 == 0:
    raise ValueError(f"smooth_window must be odd, got {window}")

  kernel = np.ones(window)/window
  padded = np.pad(areas, window//2, mode="edge")
  return np.convolve(padded, kernel, mode="valid")


def find_ed_es(areas, clip_length, smooth_window=3):
  smoothed = smooth(areas, smooth_window)
  es = np.argmin(smoothed)
  lo = max(es - (clip_length - 1), 0)
  hi = min(es + (clip_length - 1) + 1, len(smoothed))
  ed = lo + np.argmax(smoothed[lo:hi])

  if ed == es:
    raise ValueError("LV area does not change across the video, no cardiac motion found.")

  return int(ed), int(es)


# Returns every heartbeat as an (ed, es) pair: each area peak with the first trough before the
# next peak. Videos with no complete beat fall back to find_ed_es, which raises if there's no motion
def find_beats(areas, clip_length, fps, smooth_window=SMOOTH_WINDOW, min_prominence=MIN_PROMINENCE):
  curve = smooth(areas, smooth_window)
  distance = max(1, int(round(MIN_BEAT_INTERVAL * fps)))
  prominence = min_prominence * (curve.max() - curve.min())

  peaks, _ = find_peaks(curve, distance=distance, prominence=prominence)
  troughs, _ = find_peaks(-curve, distance=distance, prominence=prominence)

  beats = []
  for i, ed in enumerate(peaks):
    next_ed = peaks[i + 1] if i + 1 < len(peaks) else len(curve)
    es_after = troughs[(troughs > ed) & (troughs < next_ed)]

    # A pair too far apart for one clip isn't one beat, e.g. a dip from the probe moving
    if len(es_after) > 0 and es_after[0] - ed < clip_length:
      beats.append((int(ed), int(es_after[0])))

  return beats or [find_ed_es(areas, clip_length)]
