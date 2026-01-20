import numpy as np
import pandas as pd
from typing import Tuple


def add_binary_label(df: pd.DataFrame, high_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["is_high"] = (df["engagement_rate"] >= high_threshold).astype(int)
    return df


def generate_followers(platform: str, rng: np.random.Generator) -> int:
    # platform-aware follower ranges (synthetic)
    if platform == "Instagram":
        return int(rng.integers(5_000, 500_000))
    if platform == "Facebook":
        return int(rng.integers(1_000, 200_000))
    # Twitter / X or others
    return int(rng.integers(500, 100_000))


def add_synthetic_followers(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["follower_count"] = df["platform"].apply(lambda p: generate_followers(p, rng))
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Uses only pre-post features:
    - platform
    - post_type
    - post_length
    - follower_count
    """
    y = df["is_high"]
    X = df[["platform", "post_type", "post_length", "follower_count"]]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def encode_new_post(raw_post: pd.DataFrame, training_columns: pd.Index) -> pd.DataFrame:
    encoded = pd.get_dummies(raw_post, drop_first=True)
    return encoded.reindex(columns=training_columns, fill_value=0)
