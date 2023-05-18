
# or I guess I can upload it to my Google Drive and expect people to download it from there


from pathlib import Path

from prefect import flow
import load_data, extract_feats, train_model


root_dir = Path(__file__).resolve().parents[1]

data_dir = root_dir / "data"

csv_meta_file = data_dir / "symbols_valid_meta.csv"
merged_file = data_dir / "market_data.parquet"
feat_file = data_dir / "market_data_with_features.parquet"
model_file = root_dir / "predictor" / "rf_volume.onnx"


@flow
def train_volume_predictor():
    """
    Train a model to predict the volume of a stock/ETF
    """
    load_data.parse_csvs(meta_input_file=csv_meta_file, output_file=merged_file)
    extract_feats.extract_features(input_file=merged_file, output_file=feat_file)
    train_model.train_model(features_file=feat_file, model_file=model_file)


if __name__ == "__main__":
    train_volume_predictor()
