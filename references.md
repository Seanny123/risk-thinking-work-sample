# References

As requested by the Work Sample instructions, here are all the
resources consulted to complete this task.

## Official Docs

- [Prefect](https://docs.prefect.io/)
- [Ibis](https://ibis-project.org/docs/)
- [DuckDB](https://duckdb.org/docs/)
- [PyArrow](https://arrow.apache.org/docs/)
- [sklearn-onnx](http://onnx.ai/sklearn-onnx/)
- [FastAPI](https://fastapi.tiangolo.com)

## Tutorials and Blog Posts

- [FastAPI on
  Fly.io](https://ahmadrosid.com/blog/deploy-fastapi-flyio)
    - [Necessity of PYTHONUNBUFFERED in
      Dockerfile](https://stackoverflow.com/q/59812009/1079075)
- [FastAPI + Ray
  Serve](https://www.anyscale.com/blog/ray-serve-fastapi-the-best-of-both-worlds)
- [Evaluating time series
  models](https://stats.stackexchange.com/a/414664/29949), which led
  to "Forecast Evaluation for Data Scientists: Common Pitfalls and
  Best Practices" by Hewamalage et al, by going through the papers
  that have cited "Evaluating time series forecasting models: an
  empirical study on performance estimation methods" by Cerqueira et
  al.
- Reminding myself of [the difference between MAE and
  MSE](https://stats.stackexchange.com/q/48267/29949).
- Trying to [find a regression/forecasting model that can train on
  billions of data
  points](https://vaex.io/blog/ml-impossible-train-a-1-billion-sample-model-in-20-minutes-with-vaex-and-scikit-learn-on-your)
  while only using a CPU. Of course its a linear model, which I
  didn't up using, since it performed worse than the
  RandomForestRegressor.
- [Make scikit-learn multi-process](https://machinelearningmastery.com/multi-core-machine-learning-in-python/).

## Miscellaneous debugging

- [Deploying with Fly.io while only using the free
  tier](https://community.fly.io/t/no-machines-in-group-app-launching-one-new-machine-error-error-creating-a-new-machine-failed-to-launch-vm-to-create-more-than-1-machine-per-app-please-add-a-payment-method/12592/2).


## Bing Chat

Used Bing Chat instead of ChatGPT, since Bing Chat reportedly uses
GPT4. Bing Chat does not allow you to save or export conversations.
It was mostly used for:

- Determining the Ibis API for windowed average and median. I knew
  this feature existed, but didn't know the exact API.
- ONNX API for exporting and running a model from scikit-learn
- Composing DuckDB SQL while pursing a dead end described in the next
  section.

## Dead Ends

At one point, I was convinced that I had to write to a single
structured file. Either a Parquet file or something else.

- [Checked Parquet files do not support multiple simultaneous
  writes](https://stackoverflow.com/q/31909636/1079075).
- Tried using [OS-level file locks based on a StackOverflow
  answer](https://stackoverflow.com/a/76282117/1079075).
    - Hit [a bug with `fastparquet` and
      `append=True`](https://github.com/dask/fastparquet/issues/807).
      Was going to use `feather` instead of `parquet`, but determined
      there was no support for appending to a `feather` file.
    - Determined Prefect does not have a similar synchronization
      primitive
- Switched to a consumer, producer pattern using `multiprocessing`
  and quickly realized Prefect does not support multiple Python
  processes.
    - Could also not use `multithreading` with Ray, due to the
      producer thread exiting before Ray was finished. Could have
      possibly resolved this with further investigation, but felt the
      implementation complexity was not worth the performance
      improvement.
- Tried doing the data loading using `duckdb`, since it's intended as
  a column-based SQLite and found its performance when ingesting
  multiple CSVs in parallel to be lacking.
- Gave up when I determined querying individual Parquet files is well
  supported by PyArrow using `pyarrow.parquet.ParquetDataset`. Thus,
  there is minimal advantage to having a single Parquet file instead
  of a folder of multiple Parquet files.
