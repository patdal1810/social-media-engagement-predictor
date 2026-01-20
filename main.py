import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("social_media_engagement_dataset.csv")


# Create binary engagement label
# High vs Not High is more production-realistic and usually performs better

HIGH_THRESHOLD = 0.07
df["is_high"] = (df["engagement_rate"] >= HIGH_THRESHOLD).astype(int)


# Create synthetic follower_count (platform-aware)
np.random.seed(42)

def generate_followers(platform):
    if platform == "Instagram":
        return np.random.randint(5_000, 500_000)
    elif platform == "Facebook":
        return np.random.randint(1_000, 200_000)
    else:  # Twitter
        return np.random.randint(500, 100_000)

df["follower_count"] = df["platform"].apply(generate_followers)


# Feature selection
# Only pre-post features are used

y = df["is_high"]
X = df[["platform", "post_type", "post_length", "follower_count"]]


# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)


# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train a Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=10,
    min_samples_split=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# Evaluate model (predicted labels)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 4))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Not High", "High"]))


# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Not High", "Pred High"],
            yticklabels=["True Not High", "True High"])
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.title("Confusion Matrix (High vs Not High)")
plt.show()


# Evaluate model with ROC-AUC (probability quality)
y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
#print("ROC-AUC:", round(auc, 4))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (High vs Not High)")
#plt.show()


# Test the model on a NEW POST (production-style prediction)
raw_post = pd.DataFrame([{
    "platform": "Instagram",      # must match dataset values exactly
    "post_type": "Text",         # must match dataset values exactly
    "post_length": 30,
    "follower_count": 25_000      # user provides estimate (or from profile)
}])


# One-hot encode the new post the same way
raw_post_encoded = pd.get_dummies(raw_post, drop_first=True)

# Align to training columns
raw_post_encoded = raw_post_encoded.reindex(columns=X.columns, fill_value=0)


# Predict class + probability
pred_is_high = rf.predict(raw_post_encoded)[0]
pred_high_proba = rf.predict_proba(raw_post_encoded)[0, 1]

print("\n--- New Post Prediction ---")
print("Predicted Label:", "High" if pred_is_high == 1 else "Not High")
print("Probability of High:", round(pred_high_proba, 3))


# Production decision threshold (tune this)
DECISION_THRESHOLD = 0.60

if pred_high_proba >= DECISION_THRESHOLD:
    print("✅ This post is likely to be HIGH engagement")
else:
    print("❌ This post is unlikely to be HIGH engagement (Not High)")
