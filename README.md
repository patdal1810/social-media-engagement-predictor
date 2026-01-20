# Social Media Engagement Predictor (High vs Not High)

A production-style machine learning project that predicts whether a social media post is likely to achieve **high engagement** using only **pre-post features**.

This project is designed for **portfolio presentation on GitHub**, with clean structure, reproducibility, and real-world ML design choices.

---

## Project Overview

The model predicts whether a post will be **High Engagement** or **Not High Engagement** based on:

- **Platform** (Instagram, Facebook, Twitter)
- **Post Type** (Text, Image, Video)
- **Post Length**
- **Follower Count** (synthetic, platform-aware)

A **Random Forest Classifier** is used for robust performance and interpretability.

---

## Problem Framing

Instead of predicting exact engagement values, the task is framed as a **binary classification** problem:

- **High Engagement** → `engagement_rate >= 0.07`
- **Not High Engagement** → otherwise

This mirrors how engagement decisions are made in real production systems.

---

## Repository Structure

```
social-media-engagement-predictor/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── social_media_engagement_dataset.csv
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── scripts/
│   ├── run_train.py
│   ├── run_predict.py
│   └── make_sample_data.py
│
└── reports/
    └── figures/
```

---

## ⚙️ Setup Instructions

### Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Requirements

Place your dataset at:
```
data/raw/social_media_engagement_dataset.csv
```

Required columns:
- `platform`
- `post_type`
- `post_length`
- `engagement_rate`

### Optional: Generate Sample Dataset
```bash
python scripts/make_sample_data.py
```

---

## Train the Model
```bash
python scripts/run_train.py
```

Artifacts saved:
- `artifacts/model.joblib`
- `artifacts/feature_columns.joblib`

---

## Predict on a New Post
```bash
python scripts/run_predict.py
```

Example Output:
```
Predicted Label: High
Probability of High: 0.73
Decision: HIGH
```

---

## Evaluation Metrics
- Accuracy
- Classification Report
- Confusion Matrix
- ROC-AUC Curve

Saved under:
```
reports/figures/
```

---

## Design Decisions
- Binary threshold labeling improves stability
- Uses only pre-post features (no leakage)
- Synthetic follower count simulates real-world constraints
- Modular, production-ready code layout

---

## Future Improvements
- Replace synthetic followers with real data
- Hyperparameter tuning & calibration
- Model explainability (SHAP)
- Streamlit / FastAPI deployment
- CI testing pipeline

---

## 📄 License
MIT License

---

## 👤 Author
Built as a **machine learning portfolio project** demonstrating:
- Feature engineering
- Model training & evaluation
- Production-style ML workflow
