from typing import Dict

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import DATA_PROCESSED, EXPLANATIONS_PATH, MODEL_PATH, SHAP_IMPORTANCE_PATH, SHAP_SUMMARY_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def compute_shap_values(model, X: pd.DataFrame):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values


def save_global_importance(shap_values, feature_names):
    importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    importance_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)


def save_summary_plot(shap_values, X: pd.DataFrame):
    shap.summary_plot(shap_values, X, show=False)
    plt.gcf().savefig(SHAP_SUMMARY_PATH, bbox_inches="tight")
    plt.close()


def explain_example(shap_values, X: pd.DataFrame, idx: int) -> str:
    row = X.iloc[idx]
    values = shap_values[idx].values
    feature_names = X.columns
    top_idx = np.argsort(np.abs(values))[-5:][::-1]
    reasons = [f"{feature_names[i]}={row.iloc[i]} (impact {values[i]:.3f})" for i in top_idx]
    return "Likely churn drivers: " + ", ".join(reasons)


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED)
    model = load_model()

    # Use the model pipeline's preprocessor to keep feature names aligned
    preprocessor = model.named_steps["preprocessor"]
    X = df.drop(columns=["Churn"])
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    shap_values = compute_shap_values(model.named_steps["model"], X_df)

    save_global_importance(shap_values, feature_names)
    save_summary_plot(shap_values, X_df)

    explanation = explain_example(shap_values, X_df, idx=0)
    EXPLANATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPLANATIONS_PATH, "w", encoding="utf-8") as f:
        f.write(explanation + "\n")

    print("Saved SHAP outputs")


if __name__ == "__main__":
    main()
