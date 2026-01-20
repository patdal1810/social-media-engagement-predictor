from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Label threshold: engagement_rate >= HIGH_THRESHOLD => "High"
    HIGH_THRESHOLD: float = 0.07

    # Prediction decision threshold for "ship" decision
    DECISION_THRESHOLD: float = 0.60

    # Random seed for reproducibility
    SEED: int = 42

    # Random Forest hyperparams
    N_ESTIMATORS: int = 400
    MIN_SAMPLES_LEAF: int = 10
    MIN_SAMPLES_SPLIT: int = 5
    MAX_FEATURES: str = "sqrt"
    CLASS_WEIGHT: str = "balanced"
