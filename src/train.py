import json
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED, METRICS_PATH, MODEL_PATH, TARGET_COL


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    if TARGET_COL in categorical:
        categorical.remove(TARGET_COL)
    if TARGET_COL in numeric:
        numeric.remove(TARGET_COL)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return preprocessor, categorical, numeric


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
    }


def _train_all_models(df: pd.DataFrame) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, _, _ = build_preprocessor(df)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
        ),
    }

    results = []
    trained_models: Dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    return trained_models, metrics_df


def train_best_model(df: pd.DataFrame, save_artifacts: bool = False) -> Pipeline:
    trained_models, metrics_df = _train_all_models(df)
    best_name = metrics_df.iloc[0]["model"]

    if save_artifacts:
        joblib.dump(trained_models[best_name], MODEL_PATH)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(METRICS_PATH, index=False)
        with open(METRICS_PATH.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    return trained_models[best_name]


def train_and_compare(df: pd.DataFrame) -> pd.DataFrame:
    trained_models, metrics_df = _train_all_models(df)
    best_name = metrics_df.iloc[0]["model"]

    joblib.dump(trained_models[best_name], MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    with open(METRICS_PATH.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    return metrics_df


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED)
    metrics_df = train_and_compare(df)
    print(metrics_df)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
