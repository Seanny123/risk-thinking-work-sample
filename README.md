# Risk Thinking Data Engineer Work Sample

Assumes the Kaggle dataset is downloaded and unpacked into a folder
named `data/` at the repository root. Including this step in the
pipeline was unfeasible given it would require:
- Sharing my Kaggle API key
- Hosting the files somewhere myself

The pipeline can be run inside Docker from the repository root using:
```bash
docker build -t local/volume-pipeline .
# note: --shm-size is to improve Ray performance
docker run -v $(pwd)/data:/app/data --shm-size=4.86GB --rm -it local/volume-pipeline
```

## Approach

- Used Prefect as the data pipeline DAG tool.
- Used Ray for parallelism, integrated with Prefect using the
  `prefect-ray` task.
- Used Ibis with the `duckdb` backend to extract the `vol_moving_avg`
  and `adj_close_rolling_med` features. Note that Ibis is backend
  agnostic, however only the `Spark`, `pandas` and `duckdb` backends
  currently support the `trailing_window` API used.
- Data is stored as a folder of `parquet` files for each stock/etf.
  It is assumed this is a sufficient "structured format" as requested
  in the Work Sample instructions. If a single dataset is needed for
  a future applications, for example querying a specific day across
  all stocks/etfs, it is trivial to amalgamate the data using
  `pyarrow.parquet.ParquetDataset`
- Basic integration tests are stored in `tests/`


## Notes on ML Training

In the instructions, the purpose of the model is not stated.

This works sample assumes the most practical use case given the
starter code is to predict the Volume of the next day, named
`next_day_volume`, given the `vol_moving_avg` and
`adj_close_rolling_med` features of the current day for a specific
stock/etf.

The previous pipeline steps generate a massive (2.8 x 10^7) amount of
data. My old laptop could not train a model to leverage that quantity
of data, with the exception of poorly performing linear models.
Instead, a manageable subset of the data (1 x 10^4) is randomly
sampled for training.

The performance of the `RandomForestModel` is terrible.

There are many options for improving this model:
- Given access to better hardware, such as a CPU cluster or a GPU, a
  model which scales better to a large amount of data could be used.
  For example, `xgboost` support both CPU cluster (via Spark or Ray)
  or GPU acceleration.
- Instead of randomly sampling data points to train the model, a
  representative sub-sample could be found.
- Improve the input features by using features lags, which
  would let me use models better-suited for time-series forecasting,
  but would be incompatible with the API for the Model Serving step.
- Use k-fold cross-validation to better estimate model
  performance.
- Fine-tune the models further using hyper-parameter tuning,
  but given how poorly it is performing, this feels futile.

Given this work sample is for a data engineering position, and not a
data science role, these options were not investigated.

## Notes on Model Serving

The model is served using ONNX on Fly.io using FastAPI. The
Dockerfile and code is stored in `predictor/`.

Given the terrible performance of the ML model, it felt futile to
optimize for serving performance and to profile the current serving
configuration, since a better performing model would likely be a
completely different size and inference performance.

Given a better performing model, the FastAPI endpoint could be
[optimized using Ray
Serve](https://www.anyscale.com/blog/ray-serve-fastapi-the-best-of-both-worlds),
which provides several features, such as:

- Micro-batching to handle large amounts of requests
- Hot-loading models on the back-end for better computational
  resource usage
- Auto-scaling resource usage according to demand/usage with
  Kubernetes using Kuberay

## References

See `references.md` for a collection of documentation consulted and
conversations with Bing Chat.

## Pipeline run logs

Output from a pipeline run can be found in `pipeline_logs.txt` in the
root of this repository.
