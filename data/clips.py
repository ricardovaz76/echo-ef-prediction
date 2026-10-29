import numpy as np


# Number of frames per clip. At TARGET_FPS (32) this is 1 second, enough to hold
# the ED -> ES span of ~98.5% of videos
CLIP_LENGTH = 32


# Position in a resampled video whose source frame is closest to an original frame index
# Used to map annotated (training) frame numbers onto the resampled video
def nearest_position(src_idx, frame):
    return int(np.argmin(np.abs(src_idx - frame)))


# Whether ED and ES are close enough together to fit in one clip
# Pairs further apart are most likely from different heartbeats
def fits_in_clip(ed_pos, es_pos, clip_length=CLIP_LENGTH):
    return abs(ed_pos - es_pos) < clip_length


# Start position that centers the ED -> ES span in the clip, kept inside the video.
# Shared by training and inference so the model always sees ED and ES placed the same way
def clip_start(ed_pos, es_pos, n_frames, clip_length=CLIP_LENGTH):
    lo, hi = min(ed_pos, es_pos), max(ed_pos, es_pos)

    start = (lo + hi + 1) // 2 - clip_length // 2
    start = min(start, n_frames - clip_length)

    return max(start, 0)


# Cuts clip_length frames from start, padding short videos by repeating the last frame.
# Returns the clip (clip_length, H, W), clip_src (original frame index of each clip
# frame) and n_valid (number of real frames before the padding)
def extract_clip(frames, src_idx, start, clip_length=CLIP_LENGTH):
    clip = frames[start:start + clip_length]
    clip_src = src_idx[start:start + clip_length]
    n_valid = len(clip)

    n_pad = clip_length - n_valid
    if n_pad > 0:
        clip = np.concatenate([clip, np.repeat(clip[-1:], n_pad, axis=0)])
        clip_src = np.concatenate([clip_src, np.repeat(clip_src[-1:], n_pad)])

    return clip, clip_src, n_valid
