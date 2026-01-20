import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_classification(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred, target_names=["Not High", "High"])
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "fpr_tpr": roc_curve(y_test, y_proba),
    }


def save_plots(eval_out: dict, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # Confusion Matrix
    cm = eval_out["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Not High", "Pred High"],
        yticklabels=["True Not High", "True High"]
    )
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (High vs Not High)")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/confusion_matrix.png", dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = eval_out["fpr_tpr"]
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (High vs Not High)")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/roc_curve.png", dpi=200)
    plt.close()
