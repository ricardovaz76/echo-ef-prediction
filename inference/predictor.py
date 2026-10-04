import numpy as np
import torch

from data import CLIP_LENGTH, TARGET_FPS, clip_start, extract_clip, load_video_frames, resample_frames
from models import build_model
from .beats import find_beats
from .segmentation import area_curve, frame_times, segment_frames


# Predicts ejection fraction from an A4C and a PSAX video
#
# The checkpoint is loaded once and reused for every prediction, so a service
# can create one predictor at startup and call predict() for each request.
#
# Inference has no tracings, so each video goes through two stages:
#   1. segment every frame and find the heartbeats (ED/ES) from the ventricle area curve
#   2. cut a clip around each beat with the same placement rule as training and run
#      the full model on every A4C beat x PSAX beat combination, averaging the EF
class EchoPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # The model predicts normalized EF, so the train set stats are needed to convert back to EF %
        if "ef_mean" not in checkpoint or "ef_std" not in checkpoint:
            raise ValueError(f"{checkpoint_path} doesn't store ef_mean/ef_std, retrain with training/train.py")
        self.ef_mean = float(checkpoint["ef_mean"])
        self.ef_std = float(checkpoint["ef_std"])

        # Clips must be built exactly like the ones the model was trained on
        self.config = checkpoint.get("config", {"clip_length": CLIP_LENGTH, "target_fps": TARGET_FPS})
        self.clip_length = self.config["clip_length"]
        self.target_fps = self.config["target_fps"]

        self.model = build_model(self.device, clip_length=self.clip_length)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    # Stage 1 for one video: resample, segment every frame, find the beats and cut
    # a clip around each one
    def _analyze_view(self, video_path, view_name):
        frames, fps = load_video_frames(video_path)
        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from the {view_name} video ({video_path})")

        rs_frames, src_idx = resample_frames(frames, fps, self.target_fps)
        probs = segment_frames(self.model, rs_frames, self.device)
        area = area_curve(probs)
        times = frame_times(src_idx, fps)

        # max_span keeps every beat short enough to fit in one clip
        beats = find_beats(area, self.target_fps, max_span=self.clip_length - 1)
        if not beats:
            raise ValueError(f"No heartbeat detected in the {view_name} video ({video_path})")

        clips = []
        for b in beats:
            start = clip_start(b["ed"], b["es"], len(rs_frames), self.clip_length)
            clip, _, n_valid = extract_clip(rs_frames, src_idx, start, self.clip_length)
            clips.append((clip, n_valid))

        return {
            "fps": float(fps),
            "duration": float(len(frames) / fps) if fps > 0 else None,
            "times": times,
            "area": area,
            "src_idx": src_idx,
            "beats": beats,
            "clips": clips,
            "masks": probs,
        }

    # Stage 2: run the full model on every A4C beat x PSAX beat combination at once
    @torch.no_grad()
    def _predict_beats(self, a4c, psax):
        pairs = [(i, j) for i in range(len(a4c["clips"])) for j in range(len(psax["clips"]))]

        def batch(view, idx):
            # Same normalization as EchoDataset: uint8 -> [0, 1], with a channel dim
            clips = np.stack([view["clips"][k][0] for k in idx])
            frames = torch.from_numpy(clips).float().div(255.0).unsqueeze(2).to(self.device)   # (B, T, 1, H, W)
            valid = torch.tensor([view["clips"][k][1] for k in idx], device=self.device)
            return frames, valid

        a4c_frames, a4c_valid = batch(a4c, [i for i, _ in pairs])
        psax_frames, psax_valid = batch(psax, [j for _, j in pairs])

        pred, _ = self.model(a4c_frames, psax_frames, a4c_valid, psax_valid)
        efs = (pred.reshape(-1) * self.ef_std + self.ef_mean).cpu().tolist()

        return [{"a4c_beat": i, "psax_beat": j, "ef": ef} for (i, j), ef in zip(pairs, efs)]

    # Describes one view for the result: its area curve over time and where each beat is
    def _view_result(self, view, return_masks):
        times, src_idx = view["times"], view["src_idx"]
        result = {
            "fps": view["fps"],
            "duration": view["duration"],
            "times": times.tolist(),
            "area": view["area"].tolist(),
            "beats": [
                {
                    "ed_time": float(times[b["ed"]]),
                    "es_time": float(times[b["es"]]),
                    "ed_frame": int(src_idx[b["ed"]]),
                    "es_frame": int(src_idx[b["es"]]),
                    "fallback": b["fallback"],
                }
                for b in view["beats"]
            ],
        }
        if return_masks:
            result["masks"] = view["masks"]   # (T, H, W) probabilities, one per entry in times
        return result

    # Predicts EF from an A4C and a PSAX video
    #
    # Returns a dict of plain Python types (ready to send as JSON):
    #   ef:        final EF %, the average over every beat combination
    #   beat_efs:  EF % for each A4C beat x PSAX beat combination
    #   a4c, psax: per view, the area curve with timestamps in seconds of the original
    #              video, and the ED/ES time and original frame of each beat
    #   config:    the clip settings used
    # With return_masks=True each view also includes its per-frame mask probabilities
    # as a numpy array (T, H, W), which is not JSON-ready
    def predict(self, a4c_path, psax_path, return_masks=False):
        a4c = self._analyze_view(a4c_path, "A4C")
        psax = self._analyze_view(psax_path, "PSAX")

        beat_efs = self._predict_beats(a4c, psax)

        return {
            "ef": float(np.mean([b["ef"] for b in beat_efs])),
            "beat_efs": beat_efs,
            "a4c": self._view_result(a4c, return_masks),
            "psax": self._view_result(psax, return_masks),
            "config": dict(self.config),
        }
