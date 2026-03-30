import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

# Ensure project root is on sys.path when running via streamlit.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep import clean_data, load_raw_data
from src.features import add_features
from src.insights import generate_insights
from src.train import train_best_model

st.set_page_config(page_title="Churn Dashboard", layout="wide")

raw_df = load_raw_data()
df = clean_data(raw_df)
df = add_features(df)
model = train_best_model(df, save_artifacts=False)

st.title("Customer Churn Analytics Dashboard")

st.subheader("Filters")
plan_filter = st.multiselect("Contract", options=sorted(df["Contract"].dropna().unique().tolist()))
if plan_filter:
    df = df[df["Contract"].isin(plan_filter)]

proba = model.predict_proba(df.drop(columns=["Churn"]))[:, 1]
high_risk = (proba >= 0.5).sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate %", f"{churn_rate:.1f}")
with col3:
    st.metric("High-risk customers", int(high_risk))

st.subheader("Feature importance")
X = df.drop(columns=["Churn"])
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
customer_id = st.selectbox("Customer ID", df["customerID"].unique())
row = df[df["customerID"] == customer_id].iloc[0]
row_df = pd.DataFrame([row])
row_df = add_features(row_df)

proba = model.predict_proba(row_df.drop(columns=["Churn"]))[:, 1][0]
insights = generate_insights(row_df.iloc[0].to_dict(), float(proba))

st.write(f"Churn probability: {proba:.2f}")
st.write("Insights:")
for item in insights["insights"]:
    st.write(f"- {item}")
