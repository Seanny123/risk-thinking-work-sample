from pathlib import Path

import ibis
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner

ibis.set_backend("duckdb")

@task
async def extract_features(input_file: Path, output_folder: Path):
    
    table = ibis.read_parquet(input_file)
    # FML, this definitly includes the current row, so I have to use rowsBetween(-31, -1) to get the last 30 days
    window = ibis.trailing_window(30, order_by="Date", group_by="Symbol")

    vol_moving_avg = table["Volume"].mean().over(window)

    adj_close_rolling_med = table["Adj Close"].approx_median().over(window)

    table.mutate(vol_moving_avg=vol_moving_avg, adj_close_rolling_med=adj_close_rolling_med).to_parquet(output_folder / input_file.name)
    print("extracted features for", input_file.name)


@flow(task_runner=RayTaskRunner())
def process_folder(input_folder: Path, output_folder: Path):
    """
    For each stock/ETF, add columns `vol_moving_avg` and `adj_close_rolling_med` with 30 day trailing windows
    save the result to parquet file
    """
    print("extracting features")
    extract_features.map(input_file=input_folder.glob("*.parquet"), output_folder=output_folder)
    print("done features")
