"""
Converts CSVs from Kaggle dataset into Parquet format for feature extraction.
"""

from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect_ray.task_runners import RayTaskRunner
import pyarrow as pa
import pyarrow.parquet as pq

expected_columns = frozenset(
    {
        "Symbol",
        "Security Name",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }
)

schema = pa.schema(
    [
        ("Date", pa.string()),
        ("Open", pa.float64()),
        ("High", pa.float64()),
        ("Low", pa.float64()),
        ("Close", pa.float64()),
        ("Adj Close", pa.float64()),
        ("Volume", pa.int64()),
        ("Symbol", pa.string()),
        ("Security Name", pa.string()),
    ]
)


@task
async def csv_to_parquet(input_csv_file: Path, output_folder: Path, sec_name: str):
    """
    Convert CSV to parquet format and write/append it to the given output file.

    Also adds Symbol and Security Name columns to the dataframe.

    Parameters
    ----------
    input_csv_file : Path
        Path to the input CSV file.
    output_parquet_file : Path
        Path to the output parquet file.
    sec_name : str
        Security name to add to the dataframe's Security Name column.
    """
    # Use pyarrow backend, since this dataframe is being written to Parquet anyways
    symbol = input_csv_file.stem
    in_df = pd.read_csv(input_csv_file, dtype_backend="pyarrow").assign(**{"Symbol": symbol, "Security Name": sec_name})
    assert len(in_df) > 0
    assert set(in_df.columns) == expected_columns
    # Use pyarrow conversion, instead of the default pandas conversion, since it allows for enforcing a schema and doesn't require
    pq.ParquetWriter(output_folder / f"{symbol}.parquet", schema).write_table(
        pa.Table.from_pandas(in_df, schema=schema)
    )
    print("Wrote parquet", symbol)


@flow(task_runner=RayTaskRunner())
def process_csvs(meta_input_file: Path, output_folder: Path):
    """
    Read all CSVs defined in meta_input_file and add them to a parquet file.

    Parameters
    ----------
    meta_input_file : Path
        Path to the CSV file defining.
    output_file : Path
        Path to the output parquet file.
    """
    meta_dict = pd.read_csv("data/symbols_valid_meta.csv").to_dict(orient="records")

    meta_path = meta_input_file.parent

    sec_names = []
    input_csvs = []

    for row in meta_dict:
        symbol = row["NASDAQ Symbol"]
        if row["ETF"] == "Y":
            in_csv = meta_path / "etfs" / f"{symbol}.csv"
        elif row["ETF"] == "N":
            in_csv = meta_path / "stocks" / f"{symbol}.csv"
        else:
            raise ValueError(f"Unknown ETF value {row['ETF']} for symbol {symbol}. Expected 'Y' or 'N'.")

        input_csvs.append(in_csv)
        sec_names.append(row["Security Name"])

    csv_to_parquet.map(input_csv_file=input_csvs, output_folder=output_folder, sec_name=sec_names)
