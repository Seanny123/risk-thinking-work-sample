"""
In the instructions, the purpose of the model is not stated.

I'm assuming the most practical use case given the starter code is to
predict the Volume of the next day, given the features of the current
day for a specific stock.
"""
from pathlib import Path

import pandas as pd
from prefect import task

import skl2onnx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@task
def train_model(features_file: Path, model_file: Path):
    print("training model")
    data = pd.read_parquet(features_file, columns=["Date", "Symbol", "Volume", "vol_moving_avg", "adj_close_rolling_med"])

    # choose an arbitrary stock to evaluate on
    data = data[data["Symbol"] == "AACG"]
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
    print("mae improvement over naive: (should be positive)", naive_mae - mae)
    print("mse improvement over naive: (should be positive)", naive_mse - mse)

    # There are many options for improving these models:
    # - I could improve the input features by using features lags, which would let me use models better-suited for time-series forecasting, but that would mess up the API in the next step
    # - I could use k-fold cross-validation to better estimate model performance
    # - I could fine-tune the models further using hyper-parameter tuning, but given how poorly it is performing, this feels futile
    # Either way, given this work sample is for a data engineering
    # position, I'm not going to spend any more time on this

    onnx_model = skl2onnx.to_onnx(model, X=X_train.values.astype("float32"))

    with open(model_file, "wb+") as f:
        f.write(onnx_model.SerializeToString())
