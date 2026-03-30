from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "churn_features.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "metrics.csv"
SHAP_IMPORTANCE_PATH = BASE_DIR / "reports" / "shap_importance.csv"
SHAP_SUMMARY_PATH = BASE_DIR / "reports" / "shap_summary.png"
EXPLANATIONS_PATH = BASE_DIR / "reports" / "example_explanations.txt"

TARGET_COL = "Churn"
ID_COL = "customerID"
