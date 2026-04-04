import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.title("💳 AI Credit Risk Decision System")
st.markdown("Predict default risk and understand **why** using AI + SHAP")

# ================================
# LOAD MODEL + SCALER
# ================================
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================================
# INPUT SECTION
# ================================
st.sidebar.header("📥 Enter Applicant Details")

age = st.sidebar.slider("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 1000000, 20000)
debt = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)

late90 = st.sidebar.slider("90 Days Late", 0, 20, 0)
late60 = st.sidebar.slider("60-89 Days Late", 0, 20, 0)
late30 = st.sidebar.slider("30-59 Days Late", 0, 20, 0)

credit_lines = st.sidebar.slider("Credit Lines", 0, 50, 5)
dependents = st.sidebar.slider("Dependents", 0, 10, 0)
util = st.sidebar.slider("Credit Utilization", 0.0, 5.0, 0.5)

# ================================
# CREATE DATAFRAME
# ================================
data = pd.DataFrame({
    "age": [age],
    "MonthlyIncome": [income],
    "DebtRatio": [debt],
    "NumberOfTimes90DaysLate": [late90],
    "NumberOfTime30-59DaysPastDueNotWorse": [late30],
    "NumberOfTime60-89DaysPastDueNotWorse": [late60],
    "NumberOfOpenCreditLinesAndLoans": [credit_lines],
    "NumberOfDependents": [dependents],
    "RevolvingUtilizationOfUnsecuredLines": [util]
})

# ================================
# FEATURE ENGINEERING (SAME AS MODEL)
# ================================
data["income_per_person"] = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)

data["total_late"] = (
    data["NumberOfTimes90DaysLate"] +
    data["NumberOfTime30-59DaysPastDueNotWorse"] +
    data["NumberOfTime60-89DaysPastDueNotWorse"]
)

data["late_severity"] = (
    data["NumberOfTimes90DaysLate"] * 3 +
    data["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
    data["NumberOfTime30-59DaysPastDueNotWorse"]
)

data["debt_to_income"] = data["DebtRatio"] / (data["MonthlyIncome"] + 1)

data["credit_per_line"] = data["RevolvingUtilizationOfUnsecuredLines"] / (
    data["NumberOfOpenCreditLinesAndLoans"] + 1
)

data["risk_score"] = data["late_severity"] * data["DebtRatio"]
data["stress_score"] = data["total_late"] / (data["MonthlyIncome"] + 1)

# ================================
# SCALE
# ================================
data_scaled = scaler.transform(data)

# ================================
# PREDICTION
# ================================
prob = model.predict_proba(data_scaled)[0][1]

# Use your optimized threshold
threshold = 0.78  

prediction = "⚠️ High Risk (Likely Default)" if prob > threshold else "✅ Low Risk"

# ================================
# DISPLAY RESULT
# ================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Prediction")
    st.metric("Default Probability", f"{prob:.2f}")

    if prob > threshold:
        st.error(prediction)
    else:
        st.success(prediction)

with col2:
    st.subheader("🧾 Input Summary")
    st.write(data.T)

# ================================
# SHAP EXPLANATION
# ================================
st.subheader("🔍 Why this prediction? (SHAP Explainability)")

explainer = shap.Explainer(model, data_scaled)
shap_values = explainer(data_scaled)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# ================================
# GLOBAL INFO (OPTIONAL)
# ================================
st.markdown("---")
st.markdown("### 🧠 Model Insights")

st.write("""
- High **late payments** → increases default risk  
- High **debt ratio** → increases risk  
- Higher **income stability** → reduces risk  
- Model uses AI + Explainability (SHAP)
""")
