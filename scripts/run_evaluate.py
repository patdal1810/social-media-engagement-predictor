import os
import joblib

from src.evaluate import evaluate_classification, save_plots

ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = "reports/figures"


def main():
    model = joblib.load(f"{ARTIFACTS_DIR}/model.joblib")
    # You can also persist X_test/y_test if you want reproducible eval.
    # For now, evaluate during training OR rerun train and evaluate in one pipeline.

    print("This script expects you to evaluate right after training or extend it to reload X_test/y_test.")
    print("Suggestion: combine train + eval in a single pipeline script if you want 1-click evaluation.")


if __name__ == "__main__":
    main()
