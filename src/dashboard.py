# src/dashboard.py
# Professional Streamlit dashboard for the Customer Churn Analytics system.
#
# Layout:
#   Sidebar  → customer profile form (all 20 inputs in logical groups)
#   Main     → prediction results: probability gauge, risk badge, insights cards
#
# This file contains NO model, NO training logic, NO dataset.
# It only builds a JSON payload and POSTs to the FastAPI service.
#
# Run locally (API must be running first):
#   streamlit run src/dashboard.py
#
# To test against local API, change API_URL to: http://localhost:8000

import requests
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8fafc; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem; color: #1e3a5f; }

    /* Main title */
    .main-title { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0; }
    .main-subtitle { color: #64748b; margin-top: 0; font-size: 0.95rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Risk badges */
    .badge-high   { background:#fee2e2; color:#dc2626; border-radius:6px; padding:6px 14px; font-weight:600; display:inline-block; }
    .badge-medium { background:#fef9c3; color:#d97706; border-radius:6px; padding:6px 14px; font-weight:600; display:inline-block; }
    .badge-low    { background:#dcfce7; color:#16a34a; border-radius:6px; padding:6px 14px; font-weight:600; display:inline-block; }

    /* Insight cards */
    .insight-card {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }

    /* Section divider */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 1rem 0 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── API configuration ─────────────────────────────────────────────────────────
# Change to http://localhost:8000 for local testing
API_URL = "http://localhost:8000"


# ── Sidebar — Customer Profile Form ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Profile")
    st.divider()

    # Personal details
    st.markdown('<p class="section-header">Personal</p>', unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID", value="0000-TEST", placeholder="e.g. 7590-VHVEG")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")

    col3, col4 = st.columns(2)
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

    # Phone services
    st.markdown('<p class="section-header">Phone Service</p>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "Yes", "No"],
        disabled=(phone_service == "No"),
    )
    if phone_service == "No":
        multiple_lines = "No phone service"

    # Internet services
    st.markdown('<p class="section-header">Internet Service</p>', unsafe_allow_html=True)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    no_internet = internet_service == "No"
    internet_default = "No internet service" if no_internet else None

    online_security  = st.selectbox("Online Security",   ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)
    online_backup    = st.selectbox("Online Backup",     ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)
    device_protection= st.selectbox("Device Protection", ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)
    tech_support     = st.selectbox("Tech Support",      ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)
    streaming_tv     = st.selectbox("Streaming TV",      ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)
    streaming_movies = st.selectbox("Streaming Movies",  ["Yes", "No"] if not no_internet else ["No internet service"], disabled=no_internet)

    if no_internet:
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"

    # Contract & billing
    st.markdown('<p class="section-header">Contract & Billing</p>', unsafe_allow_html=True)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    col5, col6 = st.columns(2)
    with col5:
        monthly_charges = st.number_input("Monthly ($)", min_value=0.0, max_value=200.0, value=75.0, step=0.05)
    with col6:
        total_charges = st.number_input("Total ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * tenure), step=1.0)

    st.divider()
    submitted = st.button("Predict Churn Risk", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📊 Customer Churn Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Predict churn probability and get targeted retention recommendations</p>', unsafe_allow_html=True)
st.divider()

if submitted:
    # Build the JSON payload — field names must match CustomerRecord in schemas.py
    payload = {
        "records": [{
            "customerID":       customer_id,
            "gender":           gender,
            "SeniorCitizen":    int(senior),
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           float(tenure),
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protection,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   float(monthly_charges),
            "TotalCharges":     float(total_charges),
        }]
    }

    with st.spinner("Sending to API..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            prediction = response.json()["predictions"][0]
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            st.info("Make sure the API is running. See the README for instructions.")
            st.stop()

    prob     = prediction["churn_probability"]
    label    = prediction["churn_label"]
    insights = prediction["insights"]

    # ── Risk classification ───────────────────────────────────────────────────
    if prob >= 0.70:
        risk_level = "High"
        risk_html  = f'<span class="badge-high">High Risk</span>'
        alert_fn   = st.error
        alert_msg  = f"High churn risk detected — immediate action recommended"
    elif prob >= 0.40:
        risk_level = "Medium"
        risk_html  = f'<span class="badge-medium">Medium Risk</span>'
        alert_fn   = st.warning
        alert_msg  = f"Moderate churn risk — monitor and consider proactive outreach"
    else:
        risk_level = "Low"
        risk_html  = f'<span class="badge-low">Low Risk</span>'
        alert_fn   = st.success
        alert_msg  = f"Low churn risk — customer appears stable"

    # ── Layout: metrics + insights ────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader(f"Results for Customer {customer_id}")
        alert_fn(alert_msg)

        # Probability gauge
        st.markdown(f"**Churn Probability** &nbsp; {risk_html}", unsafe_allow_html=True)
        st.progress(prob, text=f"{prob:.1%}")
        st.caption("0% = definitely stays &nbsp;&nbsp; 100% = definitely churns")
        st.divider()

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability",  f"{prob:.1%}")
        m2.metric("Label",        label)
        m3.metric("Risk Level",   risk_level)

        st.divider()

        # Key risk indicators for this customer
        st.subheader("Customer Summary")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"**Contract**  \n{contract}")
            st.markdown(f"**Tenure**  \n{tenure} months")
            st.markdown(f"**Internet**  \n{internet_service}")
        with s2:
            st.markdown(f"**Monthly Charge**  \n${monthly_charges:.2f}")
            st.markdown(f"**Payment**  \n{payment_method}")
            st.markdown(f"**Tech Support**  \n{tech_support}")

    with right:
        st.subheader("Retention Recommendations")
        st.caption("Actions recommended by the system based on this customer's risk profile")
        st.markdown("")

        for insight in insights:
            st.markdown(
                f'<div class="insight-card">💡 {insight}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Explanation of churn drivers
        st.subheader("Why This Risk Level?")
        drivers = []
        if contract == "Month-to-month":
            drivers.append("Month-to-month contract — no long-term commitment")
        if tenure < 12:
            drivers.append(f"Short tenure ({tenure} months) — still in early loyalty phase")
        if internet_service == "Fiber optic":
            drivers.append("Fiber optic — higher price sensitivity / more competition")
        if payment_method == "Electronic check":
            drivers.append("Electronic check — lower billing friction to cancel")
        if online_security in ("No", "No internet service") and not no_internet:
            drivers.append("No online security — unmet service need")
        if tech_support in ("No", "No internet service") and not no_internet:
            drivers.append("No tech support — potential unresolved issues")

        if drivers:
            for d in drivers:
                st.markdown(f"- {d}")
        else:
            st.markdown("No strong individual risk factors identified.")

else:
    # ── Welcome state ─────────────────────────────────────────────────────────
    st.info("Fill in the customer profile in the sidebar and click **Predict Churn Risk** to get started.")

    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**1. Enter Customer Data**")
        st.markdown("Fill in the customer's demographics, services, and billing details in the sidebar.")

    with col_b:
        st.markdown("**2. API Runs the Model**")
        st.markdown("The form data is sent to the FastAPI backend which runs a trained Gradient Boosting model.")

    with col_c:
        st.markdown("**3. Get Actionable Insights**")
        st.markdown("Churn probability and tailored retention recommendations are returned instantly.")

    st.divider()
    st.markdown("### Model Information")
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Dataset",   "IBM Telco Churn")
    info2.metric("Records",   "7,043")
    info3.metric("ROC-AUC",   "0.8465")
    info4.metric("Algorithm", "Logistic Regression")
