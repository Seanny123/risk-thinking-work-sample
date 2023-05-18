"""
Assuming the dataset from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
has already been downloaded in into the data/ folder, because otherwise I have to provide my Kaggle API key
and that's a secret!
"""

import asyncio
from pathlib import Path

from more_itertools import chunked
import pandas as pd
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import pyarrow as pa
import pyarrow.parquet as pq

expected_columns = frozenset({
    "Symbol",
    "Security Name",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
})

schema = pa.schema([
    ("Date", pa.string()),
    ("Open", pa.float64()),
    ("High", pa.float64()),
    ("Low", pa.float64()),
    ("Close", pa.float64()),
    ("Adj Close", pa.float64()),
    ("Volume", pa.int64()),
    ("Symbol", pa.string()),
    ("Security Name", pa.string())
])


@task
async def read_csv(input_file: Path, sec_name: str) -> pd.DataFrame:
    in_df = pd.read_csv(input_file, dtype_backend="pyarrow").assign(**{"Symbol": input_file.stem, "Security Name": sec_name})
    print(sec_name, in_df.shape)
    assert len(in_df) > 0
    assert set(in_df.columns) == expected_columns
    return pa.Table.from_pandas(in_df, schema=schema)


@flow(task_runner=RayTaskRunner())
async def parse_csvs(meta_input_file: Path, output_file: Path):
    """
    Read all the CSVs and add them to a parquet file
    """
    print("start parsing csvs")
    meta = pd.read_csv(meta_input_file)

    meta_path = meta_input_file.parent
    writer = pq.ParquetWriter(output_file, schema)
    meta_dict = meta.to_dict(orient="records")

    # run tasks in chunks, otherwise workflow gets stuck submitting tasks and never completes them
    for meta_chunk in chunked(meta_dict, n=100):
        futures = []

        for row in meta_chunk:
            symbol = row["NASDAQ Symbol"]
            if row["ETF"] == "Y":
                in_csv = meta_path / "etfs" / f"{symbol}.csv"
            else:
                in_csv = meta_path / "stocks" / f"{symbol}.csv"

            futures.append(read_csv(in_csv, row["Security Name"]))

        for f_i, future in enumerate(asyncio.as_completed(futures)):
            table = await future
            writer.write_table(table)
            print("Remaining", len(meta) - f_i - 1)

    print("done")
    return True
