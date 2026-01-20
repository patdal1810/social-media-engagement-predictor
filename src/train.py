from dataclasses import asdict
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import Config
from .features import add_binary_label, add_synthetic_followers, build_features


def train_model(df: pd.DataFrame, cfg: Config) -> Tuple[RandomForestClassifier, pd.Index, Dict]:
    df = add_binary_label(df, cfg.HIGH_THRESHOLD)
    df = add_synthetic_followers(df, cfg.SEED)

    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=cfg.SEED,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=cfg.N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=cfg.MIN_SAMPLES_LEAF,
        min_samples_split=cfg.MIN_SAMPLES_SPLIT,
        max_features=cfg.MAX_FEATURES,
        class_weight=cfg.CLASS_WEIGHT,
        random_state=cfg.SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    meta = {
        "config": asdict(cfg),
        "feature_columns": list(X.columns),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }
    return model, X.columns, {"X_test": X_test, "y_test": y_test, "meta": meta}


def save_artifacts(model, feature_columns: pd.Index, out_dir: str) -> None:
    joblib.dump(model, f"{out_dir}/model.joblib")
    joblib.dump(list(feature_columns), f"{out_dir}/feature_columns.joblib")
