import os
from src.config import Config
from src.data import load_dataset
from src.train import train_model, save_artifacts

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "data/raw/social_media_engagement_dataset.csv"


def main():
    cfg = Config()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    model, feature_cols, out = train_model(df, cfg)

    save_artifacts(model, feature_cols, ARTIFACTS_DIR)

    print("Training complete")
    print("Artifacts saved to:", ARTIFACTS_DIR)
    print("Train shape:", out["meta"]["train_shape"])
    print("Test shape:", out["meta"]["test_shape"])


if __name__ == "__main__":
    main()
