import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fintech Credit Risk App", layout="centered")

st.title("Fintech Credit Risk & Lending Decision System")
st.caption("Interactive demo for fintech-style credit approval and pricing")

# Load trained artifacts
model = joblib.load("dashboard/credit_model.pkl")
scaler = joblib.load("dashboard/scaler.pkl")

st.subheader("Enter Borrower Details")

MonthlyIncome = st.number_input("Monthly Income", min_value=0.0)
DebtRatio = st.number_input("Debt Ratio", min_value=0.0)
RevolvingUtilizationOfUnsecuredLines = st.number_input(
    "Credit Utilization (0–1)", min_value=0.0, max_value=1.0
)
age = st.slider("Age", 18, 80)
past_due = st.selectbox("Any Past Delinquency?", ["No", "Yes"])

if st.button("Evaluate Credit Risk"):
    past_due_flag = 1 if past_due == "Yes" else 0

    input_df = pd.DataFrame([[
        RevolvingUtilizationOfUnsecuredLines,
        age,
        past_due_flag,
        DebtRatio,
        MonthlyIncome,
        0, 0, 0, 0, 0
    ]], columns=model.feature_names_in_)

    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    if prob < 0.10:
        decision = "APPROVE"
        rate = "12%"
    elif prob < 0.25:
        decision = "REVIEW"
        rate = "18%"
    else:
        decision = "REJECT"
        rate = "25%"

    st.markdown("### Results")
    st.write(f"**Default Probability:** {prob:.2%}")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Indicative Interest Rate:** {rate}")
