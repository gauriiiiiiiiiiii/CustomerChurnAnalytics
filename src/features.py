# src/features.py
# Feature engineering for the IBM Telco Customer Churn dataset.
#
# Input : cleaned DataFrame (output of preprocess.run_cleaning_pipeline)
#         Columns present: all 19 original feature columns (customerID and Churn removed)
#
# Output: same DataFrame + 11 new engineered columns
#         These new columns must match the names in NUM_COLS in preprocess.py exactly.
#
# CRITICAL: This same function is called at BOTH training and inference time.
#   Training : src/train.py calls add_features() before fitting the model
#   Inference : src/api.py calls add_features() before predict_proba()
# Any difference between the two = silent mispredictions.
#
# Engineered features:
#   Tenure     -> tenure_months, tenure_years
#   Monetary   -> monthly_charges, total_charges, avg_monthly_charges
#   RFM proxy  -> recency, frequency, monetary
#   Engagement -> engagement_score
#   Risk       -> complaint_proxy

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy -- never mutate the caller's DataFrame
    df = df.copy()

    # ── Tenure features ───────────────────────────────────────────────────────
    # tenure_months: explicit alias so the model receives a clearly named feature.
    # Low tenure = newer customer = higher churn risk (especially Month-to-month).
    df["tenure_months"] = df["tenure"].fillna(0)
    df["tenure_years"]  = (df["tenure_months"] / 12).round(2)

    # ── Monetary features ─────────────────────────────────────────────────────
    # avg_monthly_charges: smoothed spend rate -- avoids penalising new customers
    # with low TotalCharges who simply haven't been subscribed long enough.
    df["monthly_charges"]     = df["MonthlyCharges"].fillna(0)
    df["total_charges"]       = df["TotalCharges"].fillna(0)
    df["avg_monthly_charges"] = (
        df["total_charges"] / (df["tenure_months"] + 1)
    ).round(2)

    # ── RFM-style proxies (adapted from e-commerce RFM to telecom) ────────────
    # Recency: how new is this customer relative to the longest-tenured customer?
    # Higher recency = newer = less loyal = higher churn risk.
    # NOTE: when a single customer is scored at inference, max_tenure = that customer's
    # own tenure, so recency = 0. This is an acceptable simplification for real-time use.
    max_tenure     = df["tenure_months"].max()
    df["recency"]  = (max_tenure - df["tenure_months"]).fillna(0)

    # Frequency: count of services actively subscribed to.
    # More services = deeper product adoption = harder to leave.
    df["frequency"] = df[
        [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    ].apply(lambda row: (row == "Yes").sum(), axis=1)

    # Monetary: total lifetime spend -- standard RFM monetary dimension.
    df["monetary"] = df["total_charges"]

    # ── Engagement score ──────────────────────────────────────────────────────
    # Combines service breadth (frequency) and loyalty (tenure).
    # Higher = more engaged = stickier = lower churn probability.
    # Frequency weighted 2× because service adoption is a stronger signal than raw tenure.
    df["engagement_score"] = (df["frequency"] * 2 + df["tenure_months"] / 6).round(2)

    # ── Complaint proxy ───────────────────────────────────────────────────────
    # Proxy for customer dissatisfaction using service quality indicators:
    #   +1 if TechSupport == "Yes"      -> customer needs support (has a problem)
    #   +1 if OnlineSecurity == "No"    -> missing security = vulnerability / dissatisfaction
    #   +1 if DeviceProtection == "No"  -> missing protection = value gap
    # Score range: 0 (satisfied) -> 3 (highly dissatisfied)
    df["complaint_proxy"] = (
        (df["TechSupport"]       == "Yes").astype(int)
        + (df["OnlineSecurity"]  == "No").astype(int)
        + (df["DeviceProtection"]== "No").astype(int)
    )

    return df
