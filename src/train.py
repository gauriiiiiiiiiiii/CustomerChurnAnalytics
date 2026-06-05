# src/train.py
# End-to-end training pipeline for Customer Churn Analytics.
#
# Run from the project root:
#   python src/train.py
#
# Pipeline steps:
#   1.  Load & clean data          (src/preprocess.py)
#   2.  EDA -- print key patterns
#   3.  Feature engineering        (src/features.py)
#   4.  Train / test split (80/20 stratified)
#   5.  Cross-validation -- compare 3 models
#   6.  Fit best model on full training set
#   7.  Evaluate on held-out test set
#   8.  Threshold sensitivity analysis
#   9.  Save Pipeline -> models/best_model.joblib
#
# Output: models/best_model.joblib
#   Contains: ColumnTransformer (OneHotEncoder + StandardScaler) + trained classifier
#   Loaded at startup by src/api.py to serve predictions.

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package resolves correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH
from src.features import add_features
from src.preprocess import CAT_COLS, NUM_COLS, build_preprocessor, run_cleaning_pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


# ── EDA ───────────────────────────────────────────────────────────────────────

def print_eda(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 2 -- EDA: KEY CHURN PATTERNS")
    print("=" * 60)

    # Contract type is the strongest categorical predictor
    print("\nChurn rate by Contract:")
    print(
        df.groupby("Contract")["Churn"]
        .mean().sort_values(ascending=False)
        .map("{:.1%}".format).to_string()
    )

    print("\nChurn rate by Internet service:")
    print(
        df.groupby("InternetService")["Churn"]
        .mean().sort_values(ascending=False)
        .map("{:.1%}".format).to_string()
    )

    print("\nChurn rate by Payment method:")
    print(
        df.groupby("PaymentMethod")["Churn"]
        .mean().sort_values(ascending=False)
        .map("{:.1%}".format).to_string()
    )

    # Tenure is the strongest numeric predictor
    print("\nChurn rate by Tenure bucket:")
    tmp = df.copy()
    tmp["_bucket"] = pd.cut(
        tmp["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "12-24 months", "24-48 months", "48-72 months"],
        include_lowest=True,
    )
    print(
        tmp.groupby("_bucket")["Churn"]
        .mean().sort_values(ascending=False)
        .map("{:.1%}".format).to_string()
    )

    print("\nMean numeric values -- Churners vs Non-churners:")
    print(
        df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]]
        .mean().round(2).to_string()
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3 -- FEATURE ENGINEERING")
    print("=" * 60)

    df = add_features(df)

    new_cols = [
        "tenure_months", "tenure_years", "monthly_charges", "total_charges",
        "avg_monthly_charges", "recency", "frequency", "monetary",
        "engagement_score", "complaint_proxy",
    ]
    print(f"New columns added: {new_cols}")
    print(f"\nStats on engineered features:")
    print(df[new_cols].describe().round(2).to_string())
    return df


# ── Train / evaluate ──────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> Pipeline:

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 -- TRAIN / TEST SPLIT  (80 / 20, stratified)")
    print("=" * 60)

    X = df[CAT_COLS + NUM_COLS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train : {X_train.shape}  -- churn rate {y_train.mean():.1%}")
    print(f"Test  : {X_test.shape}   -- churn rate {y_test.mean():.1%}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 -- CROSS-VALIDATION  (5-fold, scoring = ROC-AUC)")
    print("=" * 60)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_pipe = None, -1, None

    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", clf),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        print(f"  {name:30s}  AUC = {mean_auc:.4f}  ±{scores.std():.4f}")

        if mean_auc > best_score:
            best_score, best_name, best_pipe = mean_auc, name, pipe

    print(f"\n  Best -> {best_name}  (CV AUC = {best_score:.4f})")

    # ── Fit on full training set ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 6 -- FIT {best_name} ON FULL TRAINING SET")
    print("=" * 60)
    best_pipe.fit(X_train, y_train)
    print("  Done.")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 -- TEST SET EVALUATION  (threshold = 0.50)")
    print("=" * 60)

    proba = best_pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, proba)
    recall    = recall_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds)

    print(f"\n  ROC-AUC   : {auc:.4f}")
    print(f"  Recall    : {recall:.4f}  <- fraction of actual churners correctly caught")
    print(f"  Precision : {precision:.4f}  <- fraction of flagged customers who truly churn")
    print(f"  F1        : {f1:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, preds)
    print(f"  Confusion Matrix:")
    print(f"    True Negative  (correctly kept)    : {cm[0][0]}")
    print(f"    False Positive (wrong alarm)        : {cm[0][1]}  <- wasted retention spend")
    print(f"    False Negative (missed churner)     : {cm[1][0]}  <- most costly miss")
    print(f"    True Positive  (correctly caught)   : {cm[1][1]}")

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 -- THRESHOLD SENSITIVITY")
    print("=" * 60)
    print(f"  {'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p  = (proba >= t).astype(int)
        r  = recall_score(y_test, p, zero_division=0)
        pr = precision_score(y_test, p, zero_division=0)
        f  = f1_score(y_test, p, zero_division=0)
        tag = ""
        if t == 0.3: tag = "<- catch more churners, more false alarms"
        if t == 0.5: tag = "<- default used in api.py"
        if t == 0.7: tag = "<- fewer false alarms, miss more churners"
        print(f"  {t:>10.1f}  {r:>8.4f}  {pr:>10.4f}  {f:>8.4f}  {tag}")

    return best_pipe


# ── Save ──────────────────────────────────────────────────────────────────────

def save(pipeline: Pipeline):
    print("\n" + "=" * 60)
    print("STEP 9 -- SAVE MODEL")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"  Saved to : {MODEL_PATH}")
    print(f"  Size     : {size_kb:.1f} KB")
    print(f"  Contents : ColumnTransformer (OHE + StandardScaler) + trained classifier")
    print(f"\n  Next steps:")
    print(f"    API       : uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print(f"    Dashboard : streamlit run src/dashboard.py")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_cleaning_pipeline(DATA_PATH)
    print_eda(df)
    df = engineer(df)
    best_model = train(df)
    save(best_model)
