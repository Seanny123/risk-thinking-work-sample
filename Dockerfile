FROM python:3.10-slim-bullseye

RUN pip install --no-cache-dir pandas ibis-framework[duckdb] prefect-ray scikit-learn skl2onnx more-itertools ray

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY train/ ./train
COPY run_pipeline.sh ./

CMD ["bash", "run_pipeline.sh"]
