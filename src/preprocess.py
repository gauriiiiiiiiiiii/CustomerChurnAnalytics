# src/preprocess.py
# Data Cleaning & Preprocessing for the IBM Telco Customer Churn dataset.
#
# Raw CSV: data/raw/telco_churn.csv  (7,043 rows, 21 columns)
#
# CLEANING STEPS (in order):
#   1. Load raw CSV -- inspect shape, dtypes, missing values
#   2. Fix TotalCharges -- stored as string, blank for new customers
#   3. Fill any remaining missing categoricals with "No"
#   4. Encode target column -- Churn: "Yes"->1, "No"->0
#   5. Drop customerID -- identifier, not a feature
#   6. Validate category vocabulary -- catch unexpected values early
#
# COLUMN DEFINITIONS:
#   CAT_COLS -> OneHotEncoded by the ColumnTransformer
#   NUM_COLS -> StandardScaled by the ColumnTransformer
#   These names MUST match what add_features() produces (src/features.py)
#   and MUST use transformer names 'cat' / 'num' -- api.py reads these names
#   from the saved model via _extract_columns() at inference time.

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Column definitions ────────────────────────────────────────────────────────

# Categorical columns -- raw dataset values, passed to OneHotEncoder
CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric columns -- includes BOTH original numerics AND engineered features from features.py
# Note: raw 'tenure', 'MonthlyCharges', 'TotalCharges' are NOT here directly --
# they appear under their engineered aliases (tenure_months, monthly_charges, etc.)
NUM_COLS = [
    "SeniorCitizen",        # raw: 0/1 binary -- senior citizen flag
    "tenure_months",        # engineered: same as tenure (explicit rename)
    "tenure_years",         # engineered: tenure / 12
    "monthly_charges",      # engineered: same as MonthlyCharges
    "total_charges",        # engineered: TotalCharges (after fixing blanks)
    "avg_monthly_charges",  # engineered: total_charges / (tenure_months + 1)
    "recency",              # engineered: max_tenure - tenure_months (how new the customer is)
    "frequency",            # engineered: count of "Yes" across 8 service columns
    "monetary",             # engineered: total_charges (RFM monetary value)
    "engagement_score",     # engineered: frequency*2 + tenure_months/6
    "complaint_proxy",      # engineered: TechSupport==Yes + OnlineSecurity==No + DeviceProtection==No
]

# Expected vocabulary for each categorical column
# Used in validate_categories() to catch data quality issues early
EXPECTED_CATEGORIES = {
    "gender":            {"Male", "Female"},
    "Partner":           {"Yes", "No"},
    "Dependents":        {"Yes", "No"},
    "PhoneService":      {"Yes", "No"},
    "MultipleLines":     {"Yes", "No", "No phone service"},
    "InternetService":   {"DSL", "Fiber optic", "No"},
    "OnlineSecurity":    {"Yes", "No", "No internet service"},
    "OnlineBackup":      {"Yes", "No", "No internet service"},
    "DeviceProtection":  {"Yes", "No", "No internet service"},
    "TechSupport":       {"Yes", "No", "No internet service"},
    "StreamingTV":       {"Yes", "No", "No internet service"},
    "StreamingMovies":   {"Yes", "No", "No internet service"},
    "Contract":          {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling":  {"Yes", "No"},
    "PaymentMethod":     {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    },
}


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV and print a basic inspection report."""
    df = pd.read_csv(path)

    print(f"[Load] File   : {path.name}")
    print(f"[Load] Shape  : {df.shape}  ({len(df):,} rows, {len(df.columns)} columns)")
    print(f"[Load] Columns: {list(df.columns)}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[Load] Missing: none")
    else:
        print(f"[Load] Missing:\n{missing.to_string()}")

    print(f"[Load] TotalCharges dtype: {df['TotalCharges'].dtype}  <- stored as string in CSV")
    return df


# ── Step 2: Fix TotalCharges ──────────────────────────────────────────────────

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is a string column in the raw CSV.
    New customers (tenure = 0) have a blank TotalCharges -- they haven't been billed yet.

    Fix:
      1. Coerce to numeric -> blanks become NaN
      2. Fill NaN with MonthlyCharges × tenure (best estimate: what they would owe)
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blanks = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    print(f"[Fix] TotalCharges  : string -> float64")
    print(f"[Fix] Blank values  : {n_blanks} filled with MonthlyCharges × tenure")
    return df


# ── Step 3: Fill remaining missing categoricals ───────────────────────────────

def fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    After fixing TotalCharges, check for any remaining NaN in categorical columns.
    Fill with "No" -- a safe neutral value meaning 'service not subscribed'.
    This rarely triggers on the Telco dataset but acts as a safety net.
    """
    df = df.copy()
    filled = 0
    for col in CAT_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna("No")
                filled += n
    print(f"[Fill] Categorical NaNs filled with 'No': {filled}")
    return df


# ── Step 4: Encode target ─────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Churn from text ("Yes"/"No") to binary (1/0).
    Required by sklearn classifiers which expect numeric targets.
    Also prints class distribution -- Telco dataset is imbalanced (~27% churn).
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    n_churn    = int(df["Churn"].sum())
    n_no_churn = len(df) - n_churn
    rate       = n_churn / len(df)

    print(f"[Target] Churn = 1 (churned) : {n_churn:,}  ({rate:.1%})")
    print(f"[Target] Churn = 0 (stayed)  : {n_no_churn:,}  ({1-rate:.1%})")
    print(f"[Target] Class imbalance: use ROC-AUC and Recall, not Accuracy")
    return df


# ── Step 5: Drop non-feature columns ─────────────────────────────────────────

def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop customerID -- it is a unique identifier and must never be a model feature.
    Including it would cause data leakage: the model would memorize IDs
    rather than learning generalizable churn patterns.
    """
    df = df.copy()
    drop = [c for c in ["customerID"] if c in df.columns]
    df = df.drop(columns=drop)
    print(f"[Drop] Removed: {drop}")
    return df


# ── Step 6: Validate category vocabulary ─────────────────────────────────────

def validate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify all categorical columns contain only expected values.
    Unexpected values at training time = data quality issue.
    At inference, OneHotEncoder(handle_unknown='ignore') handles them silently.
    """
    issues = []
    for col, expected in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - expected
        if unexpected:
            issues.append(f"  {col}: {unexpected}")

    if issues:
        print("[Validate] WARNING -- unexpected values found:")
        for msg in issues:
            print(msg)
    else:
        print("[Validate] All category vocabularies are clean")
    return df


# ── Sklearn preprocessor ──────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies:
      - OneHotEncoder on CAT_COLS  (transformer name: 'cat')
      - StandardScaler  on NUM_COLS (transformer name: 'num')

    IMPORTANT: transformer names 'cat' and 'num' MUST stay as-is.
    api.py reads these names from the saved model via _extract_columns()
    to reconstruct CAT_COLS and NUM_COLS at inference time.

    handle_unknown='ignore': unseen categories at inference -> all-zero row (no crash).
    sparse_output=False: dense arrays for GradientBoosting compatibility.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_cleaning_pipeline(path: Path) -> pd.DataFrame:
    """
    Run all cleaning steps in order.
    Returns a cleaned DataFrame with customerID dropped and Churn encoded.
    Ready for add_features() in src/features.py.
    """
    print("\n" + "=" * 60)
    print("STEP 1 -- DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    df = load_raw(path)
    print()
    df = fix_total_charges(df)
    df = fill_missing_categoricals(df)
    print()
    df = encode_target(df)
    print()
    df = drop_non_features(df)
    df = validate_categories(df)

    print(f"\n[Done] Cleaned shape: {df.shape}")
    return df
