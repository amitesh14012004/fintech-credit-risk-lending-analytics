"""
Fintech Credit Risk & Lending Decision System

Interactive Streamlit application that simulates how fintech lenders
assess borrower credit risk, make approval decisions, and price loans
based on default probability.
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fintech Credit Risk Decision System",
    layout="centered"
)

st.title("Decision System")
st.caption("Interactive demo for fintech-style credit approval and pricing")

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Feature schema (MUST match training order)
# -----------------------------
FEATURE_COLUMNS = [
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "any_past_due"
]

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter Borrower Details")

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=0.0,
    step=1000.0
)

DebtRatio = st.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.0,
    step=0.05
)

CreditUtilization = st.number_input(
    "Credit Utilization (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=30
)

past_due = st.selectbox(
    "Any Past Delinquency?",
    ["No", "Yes"]
)

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("Evaluate Credit Risk"):

    any_past_due = 1 if past_due == "Yes" else 0

    input_df = pd.DataFrame(
        [[
            MonthlyIncome,
            DebtRatio,
            CreditUtilization,
            age,
            any_past_due
        ]],
        columns=FEATURE_COLUMNS
    )

    input_scaled = scaler.transform(input_df)
    default_prob = model.predict_proba(input_scaled)[0][1]

    # Decision rules
    if default_prob < 0.10:
        decision = "APPROVE"
        rate = "12%"
    elif default_prob < 0.25:
        decision = "REVIEW"
        rate = "18%"
    else:
        decision = "REJECT"
        rate = "25%"

    # -------------------------
    # Display results
    # -------------------------
    st.markdown("### Results")
    st.write(f"**Default Probability:** {default_prob:.2%}")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Indicative Interest Rate:** {rate}")
