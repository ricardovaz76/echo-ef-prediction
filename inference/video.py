import io

import av
import cv2
import numpy as np


OVERLAY_COLOR = np.array([255, 64, 64], dtype=np.float32)
OVERLAY_ALPHA = 0.4
UPSCALE = 4


# Draws the LV mask over one grayscale frame, upscaled evenly so the video keeps its proportions.
# Re-thresholding the upscaled mask smooths the blocky edges of the 112x112 prediction
def overlay_frame(frame, mask, scale=UPSCALE):
  size = (frame.shape[1] * scale, frame.shape[0] * scale)
  image = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

  mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5
  image[mask] = ((1 - OVERLAY_ALPHA) * image[mask] + OVERLAY_ALPHA * OVERLAY_COLOR).astype(np.uint8)

  return image


# Encodes frames (T, H, W) uint8 and masks (T, H, W) bool as H.264 MP4 bytes.
# OpenCV's pip builds have no H.264 encoder, and browsers can't play its mp4v output
def encode_overlay_video(frames, masks, fps):
  buffer = io.BytesIO()

  with av.open(buffer, mode="w", format="mp4") as container:
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.pix_fmt = "yuv420p"   # the only pixel format browsers reliably play
    stream.width = frames.shape[2] * UPSCALE
    stream.height = frames.shape[1] * UPSCALE

    for frame, mask in zip(frames, masks):
      packets = stream.encode(av.VideoFrame.from_ndarray(overlay_frame(frame, mask), format="rgb24"))
      container.mux(packets)

    container.mux(stream.encode())   # flushes the encoder's buffered frames

  return buffer.getvalue()
