from pathlib import Path

import ibis
from prefect import task


@task
def extract_features(input_file: Path, output_file: Path):
    """
    For each stock/ETF, add columns `vol_moving_avg` and `adj_close_rolling_med` with 30 day trailing windows
    save the result to parquet file
    """
    print("extracting features")
    # the duckdb backend seems to work, but it also eats all my RAM and CPU
    ibis.set_backend("duckdb")

    table = ibis.read_parquet(input_file)
    # FML, this definitly includes the current row, so I have to use rowsBetween(-31, -1) to get the last 30 days
    window = ibis.trailing_window(30, order_by="Date", group_by="Symbol")

    vol_moving_avg = table["Volume"].mean().over(window)

    adj_close_rolling_med = table["Adj Close"].approx_median().over(window)

    table.mutate(vol_moving_avg=vol_moving_avg, adj_close_rolling_med=adj_close_rolling_med).to_parquet(output_file)
    print("done features")
    return True
