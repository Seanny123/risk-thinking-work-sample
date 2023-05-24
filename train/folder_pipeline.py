"""
Entrypoint for model training pipeline.
"""

from pathlib import Path

from prefect import flow
import load_data_into_folder, extract_feats_from_folder, train_model


# Hardcoded paths, which could be defined as environment variables or
# CLI arguments if necessary.
root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "data"
csv_meta_file = data_dir / "symbols_valid_meta.csv"
parquet_folder = data_dir / "market_data"
parquet_folder.mkdir(exist_ok=True)
feat_folder = data_dir / "market_data_with_features"
feat_folder.mkdir(exist_ok=True)
model_file = data_dir / "rf_volume.onnx"


@flow
def train_volume_predictor():
    """
    Train a model to predict the volume of a stock/ETF
    """
    load_data_into_folder.process_csvs(meta_input_file=csv_meta_file, output_folder=parquet_folder)
    extract_feats_from_folder.process_folder(input_folder=parquet_folder, output_folder=feat_folder)
    train_model.train_model(features_folder=feat_folder, model_file=model_file)


if __name__ == "__main__":
    train_volume_predictor()
