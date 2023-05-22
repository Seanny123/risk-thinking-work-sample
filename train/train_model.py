"""

"""
from pathlib import Path

import pandas as pd
from prefect import task

import skl2onnx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@task
def train_model(features_folder: Path, model_file: Path, stock_name: str):
    feat_file = features_folder / f"{stock_name}.parquet"
    print("training model on", feat_file)
    data = pd.read_parquet(feat_file, columns=["Date", "Symbol", "Volume", "vol_moving_avg", "adj_close_rolling_med"])

    data = data.assign(**{"Date": pd.to_datetime(data["Date"])}).set_index("Date").sort_index()

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    # given
    # data.iloc[30]["vol_moving_avg"] == data.iloc[0:31]["Volume"].mean()
    # data.iloc[31]["vol_moving_avg"] == data.iloc[1:32]["Volume"].mean()
    # these are the indices to predict the next day

    X = data[features].iloc[30:-1]
    y = data[target].iloc[31:]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Compare against naive model where the forecast of the next day's Volume is the just the current day's Volume repeated
    # As recommended by "Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices" by Hewamalage et al.
    naive_y_pred = data["Volume"][X_test.index]

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    naive_mae = mean_absolute_error(y_test, naive_y_pred)
    naive_mse = mean_squared_error(y_test, naive_y_pred)

    # wow, this model is terrible!
    print("model mae:", mae)
    print("mae improvement over naive: (should be positive)", naive_mae - mae)
    print("model mse:", mse)
    print("mse improvement over naive: (should be positive)", abs(naive_mse) - abs(mse))

    onnx_model = skl2onnx.to_onnx(model, X=X_train.values.astype("float32"))

    with open(model_file, "wb+") as f:
        f.write(onnx_model.SerializeToString())
