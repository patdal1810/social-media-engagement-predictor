import pandas as pd
from src.config import Config
from src.predict import load_artifacts, predict_new_post

ARTIFACTS_DIR = "artifacts"


def main():
    cfg = Config()
    model, feature_cols = load_artifacts(ARTIFACTS_DIR)

    raw_post = pd.DataFrame([{
        "platform": "Instagram",
        "post_type": "Text",
        "post_length": 30,
        "follower_count": 25_000
    }])

    out = predict_new_post(raw_post, model, feature_cols, cfg)

    print("\n--- New Post Prediction ---")
    print("Predicted Label:", out["predicted_label"])
    print("Probability of High:", out["probability_high"])
    print("Decision Threshold:", out["decision_threshold"])
    print("Decision:", out["decision"])


if __name__ == "__main__":
    main()
