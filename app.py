import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Credit Risk AI",
    layout="wide"
)

st.title("💳 AI Credit Risk Decision System")
st.markdown(
    "Predict default risk and understand **why** using AI + SHAP"
)

# ==========================================
# LOAD MODEL
# ==========================================
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# ==========================================
# INPUT SECTION
# ==========================================
st.sidebar.header("📥 Applicant Information")

age = st.sidebar.slider("Age", 18, 100, 30)

income = st.sidebar.number_input(
    "Monthly Income",
    1000,
    1000000,
    20000
)

debt = st.sidebar.slider(
    "Debt Ratio",
    0.0,
    5.0,
    0.5
)

late90 = st.sidebar.slider(
    "90 Days Late",
    0,
    20,
    0
)

late60 = st.sidebar.slider(
    "60-89 Days Late",
    0,
    20,
    0
)

late30 = st.sidebar.slider(
    "30-59 Days Late",
    0,
    20,
    0
)

credit_lines = st.sidebar.slider(
    "Credit Lines",
    0,
    50,
    5
)

dependents = st.sidebar.slider(
    "Dependents",
    0,
    10,
    0
)

util = st.sidebar.slider(
    "Credit Utilization",
    0.0,
    5.0,
    0.5
)

threshold = st.sidebar.slider(
    "Risk Threshold",
    0.0,
    1.0,
    0.78,
    0.01
)

# ==========================================
# CREATE INPUT DATA
# ==========================================
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

# ==========================================
# FEATURE ENGINEERING
# ==========================================
data["income_per_person"] = (
    data["MonthlyIncome"]
    /
    (data["NumberOfDependents"] + 1)
)

data["total_late"] = (
    data["NumberOfTimes90DaysLate"]
    +
    data["NumberOfTime30-59DaysPastDueNotWorse"]
    +
    data["NumberOfTime60-89DaysPastDueNotWorse"]
)

data["late_severity"] = (
    data["NumberOfTimes90DaysLate"] * 3
    +
    data["NumberOfTime60-89DaysPastDueNotWorse"] * 2
    +
    data["NumberOfTime30-59DaysPastDueNotWorse"]
)

data["debt_to_income"] = (
    data["DebtRatio"]
    /
    (data["MonthlyIncome"] + 1)
)

data["credit_per_line"] = (
    data["RevolvingUtilizationOfUnsecuredLines"]
    /
    (
        data["NumberOfOpenCreditLinesAndLoans"]
        + 1
    )
)

data["risk_score"] = (
    data["late_severity"]
    *
    data["DebtRatio"]
)

data["stress_score"] = (
    data["total_late"]
    /
    (data["MonthlyIncome"] + 1)
)

# ==========================================
# MATCH TRAINING FEATURES
# ==========================================
for col in feature_names:

    if col not in data.columns:
        data[col] = 0

data = data[feature_names]

# ==========================================
# SCALE
# ==========================================
data_scaled = scaler.transform(data)

# ==========================================
# PREDICTION
# ==========================================
prob = model.predict_proba(
    data_scaled
)[0][1]

prediction = (
    "⚠️ High Risk (Likely Default)"
    if prob > threshold
    else
    "✅ Low Risk"
)

# ==========================================
# DISPLAY RESULT
# ==========================================
col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Prediction")

    st.metric(
        "Default Probability",
        f"{prob*100:.2f}%"
    )

    if prob > threshold:
        st.error(prediction)
    else:
        st.success(prediction)

with col2:

    st.subheader("🧾 Input Summary")

    st.dataframe(
        data.T,
        use_container_width=True
    )

# ==========================================
# SHAP SECTION
# ==========================================
st.subheader(
    "🔍 Why this prediction? (SHAP Explainability)"
)

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        data_scaled
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=data.iloc[0],
            feature_names=list(data.columns)
        ),
        show=False
    )

    st.pyplot(fig)

    # ======================================
    # KEY RISK DRIVERS
    # ======================================
    st.subheader("🧠 Key Risk Drivers")

    driver_df = pd.DataFrame({
        "Feature": data.columns,
        "SHAP Impact": shap_values[0]
    })

    driver_df["Abs Impact"] = np.abs(
        driver_df["SHAP Impact"]
    )

    driver_df = driver_df.sort_values(
        "Abs Impact",
        ascending=False
    )

    top3 = driver_df.head(3)

    for _, row in top3.iterrows():

        feature = row["Feature"]
        impact = row["SHAP Impact"]

        if impact > 0:

            st.write(
                f"🔺 **{feature}** increased default risk "
                f"(SHAP = {impact:.3f})"
            )

        else:

            st.write(
                f"🔻 **{feature}** reduced default risk "
                f"(SHAP = {impact:.3f})"
            )

    st.markdown(
        "### 📋 Top 10 Feature Contributions"
    )

    st.dataframe(
        driver_df[
            ["Feature", "SHAP Impact"]
        ].head(10),
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"SHAP explanation could not be generated: {e}"
    )

# ==========================================
# MODEL INSIGHTS
# ==========================================
st.markdown("---")

st.markdown("### 🧠 Model Insights")

st.write("""
• Late payments strongly increase default probability

• High debt burden increases financial stress

• High credit utilization is a warning signal

• Stable income generally reduces risk

• Model uses XGBoost + SHAP explainability

• Engineered features capture borrower behaviour more effectively than raw variables
""")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption(
    "Built with XGBoost, Feature Engineering, SHAP and Streamlit"
)
