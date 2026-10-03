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


# Start position of the clip so the ED -> ES span sits in the middle of it,
# shifted back inside the video when the span is near the start or end
#
# Shared by training (annotated ED/ES) and inference (predicted ED/ES) so the
# model always sees ED and ES placed the same way
def clip_start(ed_pos, es_pos, n_frames, clip_length=CLIP_LENGTH):
    lo, hi = min(ed_pos, es_pos), max(ed_pos, es_pos)

    start = (lo + hi + 1) // 2 - clip_length // 2
    start = min(start, n_frames - clip_length)

    return max(start, 0)


# Cuts clip_length frames out of the resampled video starting at start
#
# Videos shorter than clip_length are padded by repeating their last frame, so
# every clip has the same length
#
# Returns the clip (clip_length, H, W) and clip_src, the original frame index of
# each clip frame
def extract_clip(frames, src_idx, start, clip_length=CLIP_LENGTH):
    clip = frames[start:start + clip_length]
    clip_src = src_idx[start:start + clip_length]

    n_pad = clip_length - len(clip)
    if n_pad > 0:
        clip = np.concatenate([clip, np.repeat(clip[-1:], n_pad, axis=0)])
        clip_src = np.concatenate([clip_src, np.repeat(clip_src[-1:], n_pad)])

    return clip, clip_src
