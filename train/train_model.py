"""
Train a forecasting model to predict the next day's volume based on
the previous 30 days' volume and the previous 30 days' median
adjusted close price.
"""
from pathlib import Path

import pandas as pd
from prefect import task
import pyarrow.parquet as pq

import skl2onnx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@task
def train_model(features_folder: Path, model_file: Path):
    """
    Train a model given a folder of features + targets parquet files.

    Parameters
    ----------
    features_folder : Path
        Folder containing stock/etf parquet files for use as features + targets
    model_file : Path
        Output path to save ONNX serialized model
    """
    data = (
        pq.ParquetDataset(features_folder)
        .read()
        .to_pandas()[["Date", "Symbol", "Volume", "next_day_volume", "vol_moving_avg", "adj_close_rolling_med"]]
    )
    data.dropna(inplace=True)
    # choose 1e4 random rows for faster training
    data = data.sample(int(1e4), random_state=42).assign(**{"Date": pd.to_datetime(data["Date"])}).set_index("Date").sort_index()

    # Select features and target
    features = ["vol_moving_avg", "adj_close_rolling_med"]
    # actual target is next_day_volume, but include Volume as a
    # sanity check target used during evaluation, but not training
    target = ["next_day_volume", "Volume"]

    X = data[features]
    y = data[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model")
    # Create a RandomForestRegressor model
    model = RandomForestRegressor(random_state=42, verbose=1, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train["next_day_volume"])

    print("Evaluating model")
    # Make predictions on test data
    y_pred = model.predict(X_test)

    print("Calculating metrics")
    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test["next_day_volume"], y_pred)
    mse = mean_squared_error(y_test["next_day_volume"], y_pred)

    # Compare against naive model where the forecast of the next day's Volume is the just the current day's Volume repeated
    # As recommended by "Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices" by Hewamalage et al.
    naive_mae = mean_absolute_error(y_test["next_day_volume"], y_test["Volume"])
    naive_mse = mean_squared_error(y_test["next_day_volume"], y_test["Volume"])

    # wow, this model is terrible!
    print("model mae:", mae)
    print("mae improvement over naive: (should be positive)", naive_mae - mae)
    print("model mse:", mse)
    print("mse improvement over naive: (should be positive)", naive_mse - mse)

    onnx_model = skl2onnx.to_onnx(model, X=X_train.values.astype("float32"))

    with open(model_file, "wb+") as f:
        f.write(onnx_model.SerializeToString())
