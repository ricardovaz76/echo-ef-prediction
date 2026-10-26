from fastapi import FastAPI
from contextlib import asynccontextmanager
from inference import EchoPredictor

@asynccontextmanager
async def lifespane(app):
  # runs once at startup, before the first request
  app.state.predictor = EchoPredictor("checkpoints/best_model_mae.pth")
  app.state.predictor.warm_up()
  yield

app = FastAPI(lifespan=lifespane)

@app.get("/health")
def health():
  predictor = app.state.predictor
  return {"status": "ok", "device": str(predictor.device)}