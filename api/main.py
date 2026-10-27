from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from inference import EchoPredictor
import os
import tempfile
from .helpers import save_upload, resolve_checkpoint, UploadTooLarge

MAX_UPLOAD_BYTES = int(os.environ.get("ECHO_MAX_UPLOAD_MB", "50")) * 1024 * 1024

@asynccontextmanager
async def lifespan(app):
  checkpoint = os.environ.get("ECHO_CHECKPOINT")
  if not checkpoint:
    raise RuntimeError("ECHO_CHECKPOINT is not set, e.g. start with --env-file .env.local")

  app.state.checkpoint = checkpoint
  app.state.predictor = EchoPredictor(resolve_checkpoint(checkpoint), device=os.environ.get("ECHO_DEVICE"))
  app.state.predictor.warm_up()
  yield

app = FastAPI(lifespan=lifespan)
origins = [o.strip() for o in os.environ.get("ECHO_CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_methods=["GET", "POST"],
  allow_headers=["*"]
)

@app.get("/health")
def health():
  predictor = app.state.predictor
  return {"status": "ok", "device": str(predictor.device), "checkpoint": app.state.checkpoint}

@app.post("/predict")
def predict(a4c: UploadFile = File(...), psax: UploadFile = File(...)):
  with tempfile.TemporaryDirectory() as tmp_dir:
    # Try block for file uploads
    try:
      a4c_path = save_upload(a4c, tmp_dir, "a4c", MAX_UPLOAD_BYTES)
      psax_path = save_upload(psax, tmp_dir, "psax", MAX_UPLOAD_BYTES)
    except UploadTooLarge as e:
      raise HTTPException(status_code=413, detail=str(e))

    # Try block for model predictions
    try:
      prediction = app.state.predictor.predict(a4c_path, psax_path)
    except ValueError as e:
      msg = str(e).replace(a4c_path, a4c.filename).replace(psax_path, psax.filename)
      raise HTTPException(status_code=400, detail=msg)
    
  return {"ef": round(prediction, 1)}