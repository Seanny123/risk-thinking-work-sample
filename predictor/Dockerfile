FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY predict.py ./
COPY rf_volume.onnx ./

RUN pip install --no-cache-dir onnxruntime fastapi uvicorn

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
