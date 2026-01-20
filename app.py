"""
Fintech Credit Risk & Lending Decision System

Streamlit application demonstrating how fintech lenders
estimate default risk and make lending decisions using
an interpretable credit risk model.
"""

import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Fintech Credit Risk Decision System",
    layout="centered"
)

st.title("Fintech Credit Risk & Lending Decision System")
st.caption("Interactive demo for credit approval and risk-based pricing")

# -------------------------------------------------
# Load trained model and scaler
# -------------------------------------------------
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------
# Feature order (MUST match training exactly)
# -------------------------------------------------
FEATURES = [
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "any_past_due"
]

# -------------------------------------------------
# User input section
# -------------------------------------------------
st.subheader("Enter Borrower Details")

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0.0,
    step=1000.0,
    help="Borrower's monthly income"
)

DebtRatio = st.number_input(
    "Debt Ratio",
    min_value=0.0,
    step=0.05,
    help="Total monthly debt divided by income"
)

CreditUtilization = st.number_input(
    "Credit Utilization (0–1)",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="Proportion of available credit currently used"
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=30
)

past_due = st.selectbox(
    "Any Past Delinquency?",
    ["No", "Yes"],
    help="Whether the borrower has any past repayment delinquency"
)

# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
if st.button("Evaluate Credit Risk"):

    any_past_due = 1 if past_due == "Yes" else 0

    # Create input DataFrame in correct feature order
    input_df = pd.DataFrame(
        [[
            MonthlyIncome,
            DebtRatio,
            CreditUtilization,
            age,
            any_past_due
        ]],
        columns=FEATURES
    )

    # Scale input and predict
    input_scaled = scaler.transform(input_df)
    default_prob = model.predict_proba(input_scaled)[0][1]

    # Decision thresholds
    if default_prob < 0.10:
        decision = "APPROVE"
        interest_rate = "12%"
    elif default_prob < 0.25:
        decision = "REVIEW"
        interest_rate = "18%"
    else:
        decision = "REJECT"
        interest_rate = "25%"

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.markdown("### Credit Decision")
    st.write(f"**Default Probability:** {default_prob:.2%}")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Indicative Interest Rate:** {interest_rate}")
