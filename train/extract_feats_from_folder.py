"""
Extract features from folder of stocks/etfs Parquet files for training a forecasting model.
"""

from pathlib import Path

import ibis
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner

ibis.set_backend("duckdb")


@task
async def extract_features(input_file: Path, output_folder: Path):
    """
    Add feature columns 'vol_moving_avg' and 'adj_close_rolling_med', to Parquet file for model training.

    Also adds 'next_day_volume' column as a training target.
    """
    table = ibis.read_parquet(input_file)

    num_rows = table.count().execute()
    # need 30 days for the window + 1 day for the next day's volume
    if num_rows < 31:
        print("Skipping", input_file, "no enough data:", num_rows)
        return

    window = ibis.trailing_window(30, order_by="Date", group_by="Symbol")

    vol_moving_avg = table["Volume"].mean().over(window)
    adj_close_rolling_med = table["Adj Close"].approx_median().over(window)

    # predict the next day's volume, but keep the current day's volume as
    # a sanity check feature
    next_day_volume = table["Volume"].lead(1).cast("int64")

    # Note: sort again by date out of caution, shouldn't be necessary
    table = table.mutate(
        vol_moving_avg=vol_moving_avg,
        adj_close_rolling_med=adj_close_rolling_med,
        next_day_volume=next_day_volume,
        cast_date=table["Date"].cast("date"),
    ).order_by("cast_date")

    # given
    # data.iloc[30]["vol_moving_avg"] == data.iloc[0:31]["Volume"].mean()
    # data.iloc[31]["vol_moving_avg"] == data.iloc[1:32]["Volume"].mean()
    # mark the first 29 rows as NaN

    # can't select index directly, so choose the value at index 30 and
    # then filter by date around it
    cutoff_date = table.limit(1, offset=30)["cast_date"].execute().values[0]

    table = table.mutate(
        vol_moving_avg=(table["cast_date"] < cutoff_date).ifelse(None, table["vol_moving_avg"]),
        adj_close_rolling_med=(table["cast_date"] < cutoff_date).ifelse(None, table["adj_close_rolling_med"]),
    )
    # Note: .execute() is necessary, otherwise to_parquet fails for some reason. This is a bug workaround.
    table.drop("cast_date").execute().to_parquet(output_folder / input_file.name)
    print("extracted features for", input_file.name)


@flow(task_runner=RayTaskRunner())
def process_folder(input_folder: Path, output_folder: Path):
    """
    For each stock/etf in input_folder, add feature and target columns.

    Saves the results for each stock/etf to output_folder.

    Parameters
    ----------
    input_folder : Path
        Folder containing Parquet files for each stock/etf.
    output_folder : Path
        Folder to save feature + target augmented Parquet files.
    """
    print("extracting features")
    extract_features.map(input_file=input_folder.glob("*.parquet"), output_folder=output_folder)
    print("done features")
