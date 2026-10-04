import numpy as np
from scipy.signal import find_peaks


# Shortest time between two beats. Children's heart rates reach ~200 bpm (a 0.3 s
# cycle), so peaks closer than this can't be separate heartbeats.
MIN_BEAT_INTERVAL = 0.25

# Light smoothing of the area curve, in frames, to remove frame-to-frame jitter
#
# This and MIN_PROMINENCE were tuned on the validation split against the annotated
# ED/ES frames: on the test split, ED and ES each land within 2 frames (~60 ms at
# 32 FPS) of the annotation ~70% of the time, with a median error of 1 frame
SMOOTH_WINDOW = 5

# How far a peak or trough has to stand out from its surroundings, as a fraction of
# the curve's overall range, so noise isn't mistaken for a heartbeat
MIN_PROMINENCE = 0.1


def smooth(curve, window=SMOOTH_WINDOW):
    if window <= 1 or len(curve) < window:
        return np.asarray(curve, dtype=np.float64)

    # Moving average, with the edges padded by their own values so the ends aren't pulled down
    pad = window // 2
    padded = np.pad(np.asarray(curve, dtype=np.float64), pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


# Finds the heartbeats in a ventricle area curve sampled at fps frames per second
#
# ED (end-diastole) is where the ventricle is largest, a peak in the curve, and
# ES (end-systole) where it's smallest, a trough. Each beat pairs an ED with the
# first ES after it, before the next ED.
#
# Returns a list of beats as dicts with the ED and ES positions in the curve, and
# whether the beat came from the fallback (largest and smallest point of the whole
# curve) because no complete beat was found
def find_beats(area, fps, min_interval=MIN_BEAT_INTERVAL, smooth_window=SMOOTH_WINDOW, min_prominence=MIN_PROMINENCE):
    area = np.asarray(area, dtype=np.float64)
    if len(area) < 2:
        return []

    curve = smooth(area, smooth_window)
    spread = curve.max() - curve.min()
    if spread <= 0:
        return []

    distance = max(1, int(round(min_interval * fps)))
    prominence = min_prominence * spread

    peaks, _ = find_peaks(curve, distance=distance, prominence=prominence)        # ED candidates
    troughs, _ = find_peaks(-curve, distance=distance, prominence=prominence)     # ES candidates

    beats = []
    for i, ed in enumerate(peaks):
        next_ed = peaks[i + 1] if i + 1 < len(peaks) else len(curve)
        es_after = troughs[(troughs > ed) & (troughs < next_ed)]
        if len(es_after):
            beats.append({"ed": int(ed), "es": int(es_after[0]), "fallback": False})

    # No complete beat: fall back to the largest and smallest point, the same rule
    # training uses to pick ED and ES from the annotated frames
    if not beats:
        ed, es = int(np.argmax(curve)), int(np.argmin(curve))
        if ed != es:
            beats.append({"ed": ed, "es": es, "fallback": True})

    return beats
