import torch
from evaluation import resolve_clip_config
from models import build_model


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