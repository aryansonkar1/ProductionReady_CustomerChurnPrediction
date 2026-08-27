import streamlit as st
import joblib
import yaml
import pandas as pd
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnPredict AI",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load config & model ────────────────────────────────────────
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = joblib.load(config["model"]["save_path"])
threshold = config["evaluation"]["threshold"]

# ── Styling ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }

    h1 { color: #a78bfa; font-weight: 700; font-size: 2.2rem; }
    h3 { color: #c4b5fd; }

    .result-stay {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ecfdf5;
        margin-top: 1rem;
    }
    .result-churn {
        background: linear-gradient(135deg, #7f1d1d, #b91c1c);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fef2f2;
        margin-top: 1rem;
    }
    .metric-box {
        background: rgba(167,139,250,0.15);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        font-size: 1rem;
    }
    div[data-testid="stSlider"] > div { color: #c4b5fd; }
    .stSelectbox label, .stSlider label { color: #d8b4fe !important; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────
st.markdown('<h1>📊 ChurnPredict AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#a78bfa;margin-top:-0.5rem;">Real-time customer churn risk assessment</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input form ─────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 👤 Customer Profile")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 85.5, step=0.5)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

col3, col4 = st.columns(2)
with col3:
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
with col4:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Build input DataFrame with defaults ────────────────────────
full_df = pd.read_csv(config["data"]["raw_path"])
column_order = full_df.columns
default_row = full_df.iloc[0].copy()

user_inputs = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
}

input_data = pd.DataFrame([user_inputs])
for col in column_order:
    if col not in input_data.columns:
        input_data[col] = default_row[col]
input_data = input_data[column_order]

# ── Predict ────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True):
    with st.spinner("Analysing customer profile..."):
        try:
            prob = float(model.predict_proba(input_data)[:, 1][0])
            pred = int(prob >= threshold)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### 🔎 Prediction Result")

    if pred == 0:
        st.markdown(
            f'<div class="result-stay">🟢 Customer Will Stay</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-churn">🔴 Customer Will Churn</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="metric-box">📈 <b>Churn Probability:</b> {prob:.1%}</div>
        <div class="metric-box">⚖️ <b>Decision Threshold:</b> {threshold:.0%}</div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(prob)
