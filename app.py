"""
Fintech Credit Risk & Lending Decision System
Deployment-safe Streamlit application
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Fintech Credit Risk Decision System",
    layout="centered"
)

st.title("Fintech Credit Risk & Lending Decision System")
st.caption("Interactive demo for credit approval and risk-based pricing")

# ---------------- Load artifacts ----------------
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- User inputs ----------------
st.subheader("Enter Borrower Details")

MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, step=1000.0)
DebtRatio = st.number_input("Debt Ratio", min_value=0.0, step=0.05)
CreditUtilization = st.number_input(
    "Credit Utilization (0–1)", min_value=0.0, max_value=1.0, step=0.05
)
age = st.slider("Age", 18, 80, value=30)
past_due = st.selectbox("Any Past Delinquency?", ["No", "Yes"])

# ---------------- Prediction ----------------
if st.button("Evaluate Credit Risk"):

    any_past_due = 1 if past_due == "Yes" else 0

    # IMPORTANT: create NUMPY array (not DataFrame)
    X_input = [[
        MonthlyIncome,
        DebtRatio,
        CreditUtilization,
        age,
        any_past_due
    ]]

    # Scale and predict
    X_scaled = scaler.transform(X_input)
    default_prob = model.predict_proba(X_scaled)[0][1]

    # Decision logic
    if default_prob < 0.10:
        decision = "APPROVE"
        rate = "12%"
    elif default_prob < 0.25:
        decision = "REVIEW"
        rate = "18%"
    else:
        decision = "REJECT"
        rate = "25%"

    # ---------------- Output ----------------
    st.markdown("### Credit Decision")
    st.write(f"**Default Probability:** {default_prob:.2%}")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Indicative Interest Rate:** {rate}")
