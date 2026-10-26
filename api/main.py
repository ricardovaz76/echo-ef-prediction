from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from inference import EchoPredictor
import os
import tempfile
import shutil

def save_upload(upload, folder, name):
  extension = os.path.splitext(upload.filename or "")[1]
  path = os.path.join(folder, name + extension)
  with open(path, "wb") as f:
    shutil.copyfileobj(upload.file, f)
  return path

@asynccontextmanager
async def lifespan(app):
  # runs once at startup, before the first request
  app.state.predictor = EchoPredictor("checkpoints/best_model_mae.pth")
  app.state.predictor.warm_up()
  yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
  predictor = app.state.predictor
  return {"status": "ok", "device": str(predictor.device)}

@app.post("/predict")
def predict(a4c: UploadFile = File(...), psax: UploadFile = File(...)):
  with tempfile.TemporaryDirectory() as tmp_dir:
    a4c_path = save_upload(a4c, tmp_dir, "a4c")
    psax_path = save_upload(psax, tmp_dir, "psax")
    try:
      prediction = app.state.predictor.predict(a4c_path, psax_path)
    except ValueError as e:
      msg = str(e).replace(a4c_path, a4c.filename).replace(psax_path, psax.filename)
      raise HTTPException(status_code=400, detail=msg)
  return {"ef": round(prediction, 1)}