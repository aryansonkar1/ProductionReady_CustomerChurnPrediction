import streamlit as st
import joblib
import yaml
import pandas as pd

# Load config & model
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = joblib.load(config["model"]["save_path"])
threshold = config["evaluation"]["threshold"]

# ── Page title ────────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.write("Fill in the customer details below and click **Predict** to get a churn risk assessment.")
st.divider()

# ── Inputs ────────────────────────────────────────────────────
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=85.5, step=0.5)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

# ── Build DataFrame (all columns the pipeline expects) ────────
input_data = pd.DataFrame([{
    "gender":           "Male",
    "SeniorCitizen":    0,
    "Partner":          "No",
    "Dependents":       "No",
    "tenure":           tenure,
    "PhoneService":     "Yes",
    "MultipleLines":    "No phone service",
    "InternetService":  internet_service,
    "OnlineSecurity":   online_security,
    "OnlineBackup":     "No",
    "DeviceProtection": "No",
    "TechSupport":      tech_support,
    "StreamingTV":      "No",
    "StreamingMovies":  "No",
    "Contract":         contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod":    payment_method,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     str(monthly_charges * tenure),
}])

# ── Predict ────────────────────────────────────────────────────
if st.button("Predict", type="primary", use_container_width=True):
    try:
        prob = float(model.predict_proba(input_data)[:, 1][0])
        pred = int(prob >= threshold)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.divider()
    st.subheader("Result")

    if pred == 0:
        st.success(f"✅ Customer is likely to **Stay**  (churn probability: {prob:.1%})")
    else:
        st.error(f"⚠️ Customer is likely to **Churn**  (churn probability: {prob:.1%})")

    st.metric("Churn Probability", f"{prob:.1%}")
    st.metric("Decision Threshold", f"{threshold:.0%}")
    st.progress(prob)
