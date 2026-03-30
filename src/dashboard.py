import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import shap
import streamlit as st

# Ensure project root is on sys.path when running via streamlit.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import clean_data, load_raw_data
from src.features import add_features
from src.train import train_best_model

st.set_page_config(page_title="Churn Dashboard", layout="wide")

API_URL = "https://churn-api-y0y2.onrender.com"
CUSTOMER_FIELDS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]

raw_df = load_raw_data()
base_df = clean_data(raw_df)
feature_df = add_features(base_df)
model = train_best_model(feature_df, save_artifacts=False)

st.title("Customer Churn Analytics Dashboard")

st.subheader("Filters")
plan_filter = st.multiselect(
    "Contract", options=sorted(base_df["Contract"].dropna().unique().tolist())
)
if plan_filter:
    base_df = base_df[base_df["Contract"].isin(plan_filter)]
    feature_df = feature_df.loc[base_df.index]

proba = model.predict_proba(feature_df.drop(columns=["Churn"]))[:, 1]
high_risk = (proba >= 0.5).sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(base_df))
with col2:
    churn_rate = (base_df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate %", f"{churn_rate:.1f}")
with col3:
    st.metric("High-risk customers", int(high_risk))

st.subheader("Feature importance")
X = feature_df.drop(columns=["Churn"])
preprocessor = model.named_steps["preprocessor"]
X_transformed = preprocessor.transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_transformed, columns=feature_names)
explainer = shap.Explainer(model.named_steps["model"], X_df)
shap_values = explainer(X_df)
importance = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
importance_df = importance_df.sort_values(by="importance", ascending=False).head(15)

fig = px.bar(importance_df, x="importance", y="feature", orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Customer-level prediction")
customer_id = st.selectbox("Customer ID", base_df["customerID"].unique())
row = base_df[base_df["customerID"] == customer_id].iloc[0]
payload = {"records": [{k: row.get(k) for k in CUSTOMER_FIELDS}]}

try:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    response.raise_for_status()
    prediction = response.json()["predictions"][0]
    st.write(f"Churn probability: {prediction['churn_probability']:.2f}")
    st.write("Insights:")
    for item in prediction["insights"]:
        st.write(f"- {item}")
except requests.RequestException as exc:
    st.error(f"API request failed: {exc}")
