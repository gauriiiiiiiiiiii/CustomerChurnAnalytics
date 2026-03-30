import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure features
    df["tenure_months"] = df["tenure"].fillna(0)
    df["tenure_years"] = (df["tenure_months"] / 12).round(2)

    # Monetary features
    df["monthly_charges"] = df["MonthlyCharges"].fillna(0)
    df["total_charges"] = df["TotalCharges"].fillna(0)
    df["avg_monthly_charges"] = (df["total_charges"] / (df["tenure_months"] + 1)).round(2)

    # RFM-style features (proxy from subscription data)
    max_tenure = df["tenure_months"].max()
    df["recency"] = (max_tenure - df["tenure_months"]).fillna(0)
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
    df["monetary"] = df["total_charges"]

    # Engagement score (more services + longer tenure)
    df["engagement_score"] = (df["frequency"] * 2 + df["tenure_months"] / 6).round(2)

    # Complaint proxy (support-related and service quality indicators)
    df["complaint_proxy"] = (
        (df["TechSupport"] == "Yes").astype(int)
        + (df["OnlineSecurity"] == "No").astype(int)
        + (df["DeviceProtection"] == "No").astype(int)
    )

    return df
