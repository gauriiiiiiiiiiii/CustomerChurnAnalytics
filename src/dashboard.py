import requests
import streamlit as st

st.set_page_config(page_title="Churn Dashboard", layout="wide")

API_URL = "https://churn-api-y0y2.onrender.com"

st.title("Customer Churn Analytics Dashboard")
st.info("This dashboard sends inputs to the deployed API for predictions.")

st.subheader("Customer input")
with st.form("prediction_form"):
    customer_id = st.text_input("Customer ID", value="0000-TEST")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)
    phone_service = st.selectbox("PhoneService", ["Yes", "No"])
    multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
    payment_method = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, value=75.0)
    total_charges = st.number_input("TotalCharges", min_value=0.0, value=1200.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "records": [
            {
                "customerID": customer_id,
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": float(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges),
            }
        ]
    }

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
