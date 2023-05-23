from typing import Annotated

import onnxruntime as rt
import numpy as np

from fastapi import FastAPI, Query

app = FastAPI()

sess = rt.InferenceSession("rf_volume.onnx", providers=["CPUExecutionProvider"])


@app.get("/predict")
async def predict(
    vol_moving_avg: Annotated[int, Query(title="Volumed averaged over the last 30 days", ge=0)],
    adj_close_rolling_med: Annotated[int, Query(title="Median adjusted closing price over the last 30 days", ge=0)],
) -> Annotated[float, Query(title="Predicted volume for the next day", ge=0)]:
    return float(sess.run(None, {"X": np.array([[vol_moving_avg, adj_close_rolling_med]], dtype=np.float32)})[0][0][0])


@app.get("/healthcheck")
async def read_root():
    """Fly.io check health of node using this endpoint."""
    return {"status": "ok"}
