# Tests for the parts of the API that can break in production: startup, the
# upload contract, error status codes, upload limits, temp file cleanup and CORS

# Run from the repo root:
#   python3 -m pytest tests/ -v

import os
import tempfile
import pytest
import torch

os.environ.setdefault("ECHO_CHECKPOINT", "checkpoints/best_model_mae.pth")
os.environ.setdefault("ECHO_CORS_ORIGINS", "http://localhost:5173")

from fastapi.testclient import TestClient

import api.main
from api.main import app
from data import CLIP_LENGTH, TARGET_FPS
from models import build_model


# Bytes that no video decoder can read, for the error paths
FAKE_VIDEO = b"not a video"


# A checkpoint with untrained weights in the same format training saves, for
# machines without the real one (it's gitignored, so e.g. a fresh CI checkout)
def save_random_checkpoint(folder):
  config = {"clip_length": CLIP_LENGTH, "target_fps": TARGET_FPS}
  model = build_model(torch.device("cpu"), clip_length=config["clip_length"])

  path = folder / "random_weights.pth"
  torch.save({"model_state_dict": model.state_dict(), "ef_mean": 60.0, "ef_std": 10.0, "config": config}, path)
  return path


# One client per file, since the lifespan loads the model. Without the real checkpoint
# a random-weights one is used, so CI runs the tests instead of skipping them all
@pytest.fixture(scope="module")
def client(tmp_path_factory):
  with pytest.MonkeyPatch.context() as mp:
    if not os.path.exists(os.environ["ECHO_CHECKPOINT"]):
      mp.setenv("ECHO_CHECKPOINT", str(save_random_checkpoint(tmp_path_factory.mktemp("checkpoint"))))

    with TestClient(app) as c:
      yield c


def fake_upload(name):
  return (name, FAKE_VIDEO)


def test_health(client):
  response = client.get("/health")

  assert response.status_code == 200
  assert {"status", "device", "checkpoint"} <= response.json().keys()


def test_predict_missing_file(client):
  response = client.post("/predict", files={"a4c": fake_upload("a4c.avi")})

  assert response.status_code == 422


# An unreadable upload is the client's mistake, so it must be a 400 that names
# their file, not a 500 or a message exposing the server's temp path
def test_predict_unreadable_video(client):
  response = client.post("/predict", files={"a4c": fake_upload("my_scan.avi"), "psax": fake_upload("other.avi")})

  assert response.status_code == 400
  assert "my_scan.avi" in response.json()["detail"]
  assert tempfile.gettempdir() not in response.json()["detail"]


# The route reads MAX_UPLOAD_BYTES on every request, so shrinking it here makes
# the small fake file count as too large without uploading 50 MB
def test_predict_too_large(client, monkeypatch):
  monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 5)

  response = client.post("/predict", files={"a4c": fake_upload("a4c.avi"), "psax": fake_upload("psax.avi")})

  assert response.status_code == 413


# Uploads are written to temp folders, so a request that fails part way must still
# remove them, or a long-running server slowly fills its disk. Pointing tempfile at
# an empty folder makes any leftover easy to see
def test_failed_requests_leave_no_temp_files(client, monkeypatch, tmp_path):
  monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

  client.post("/predict", files={"a4c": fake_upload("a4c.avi"), "psax": fake_upload("psax.avi")})   # 400
  monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 5)
  client.post("/predict", files={"a4c": fake_upload("a4c.avi"), "psax": fake_upload("psax.avi")})   # 413

  assert list(tmp_path.iterdir()) == []


# Without the header the frontend's browser silently hides every response, and an
# origin that isn't listed must not get one
def test_cors_allows_only_listed_origins(client):
  allowed = client.get("/health", headers={"Origin": "http://localhost:5173"})
  other = client.get("/health", headers={"Origin": "http://evil.example"})

  assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
  assert "access-control-allow-origin" not in other.headers
