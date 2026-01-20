import pandas as pd
import joblib

from .config import Config
from .features import encode_new_post


def load_artifacts(artifacts_dir: str):
    model = joblib.load(f"{artifacts_dir}/model.joblib")
    cols = joblib.load(f"{artifacts_dir}/feature_columns.joblib")
    return model, pd.Index(cols)


def predict_new_post(raw_post: pd.DataFrame, model, feature_columns, cfg: Config) -> dict:
    raw_encoded = encode_new_post(raw_post, feature_columns)

    pred_is_high = int(model.predict(raw_encoded)[0])
    pred_high_proba = float(model.predict_proba(raw_encoded)[0, 1])

    decision = "HIGH" if pred_high_proba >= cfg.DECISION_THRESHOLD else "NOT_HIGH"

    return {
        "predicted_label": "High" if pred_is_high == 1 else "Not High",
        "probability_high": round(pred_high_proba, 3),
        "decision_threshold": cfg.DECISION_THRESHOLD,
        "decision": decision
    }
