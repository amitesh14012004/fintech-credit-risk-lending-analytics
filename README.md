# fintech-credit-risk-lending-analytics
End-to-end fintech credit risk and lending decision system using Python, interpretable models, and risk-based pricing
# Fintech Credit Risk & Lending Analytics

This project builds an end-to-end **fintech-style credit risk and lending decision system** using borrower-level financial and behavioral data.

The objective is to simulate how fintech companies and banks assess credit risk, approve or reject loans, and price credit based on borrower risk.

---

## Problem Statement

Fintech lenders process a large volume of loan applications and must make fast, data-driven decisions while controlling default risk.

This project addresses three key questions:
- Which borrowers are likely to default?
- How should lending decisions be made based on predicted risk?
- How can interest rates be adjusted to reflect borrower risk?

---

## Dataset

The project uses an anonymized retail credit dataset that closely mirrors real-world lending portfolios.

Each observation represents a borrower, with features capturing:
- Income and debt burden
- Credit utilization
- Past repayment behavior and delinquencies
- Demographic and household factors

The target variable indicates whether the borrower experienced **serious delinquency (default)** within a two-year horizon.

---

## Methodology

The project follows a structured credit analytics pipeline:

1. **Data Understanding and Cleaning**
   - Interpreted variables using a credit data dictionary
   - Handled missing income values using median imputation
   - Removed unrealistic borrower records
   - Applied light winsorization to extreme delinquency values

2. **Exploratory Risk Analysis**
   - Analyzed default patterns across income, debt ratio, age, and repayment history
   - Identified past delinquency as the strongest predictor of default
   - Validated economic intuition before modeling

3. **Credit Risk Modeling**
   - Built an interpretable logistic regression model
   - Estimated borrower-level default probabilities
   - Evaluated performance using ROC–AUC
   - Analyzed model coefficients to understand key risk drivers

4. **Credit Decision Framework**
   - Converted default probabilities into lending decisions (Approve / Review / Reject)
   - Introduced borrower-level explainability for credit decisions
   - Simulated risk-based loan pricing using probability thresholds

---

## Key Features

- Interpretable credit risk model aligned with banking practices
- Risk-based loan approval thresholds
- Explainable borrower-level credit decisions
- Risk-adjusted interest rate simulation
- Business-oriented analytics rather than black-box modeling

---

## Results and Insights

- Past repayment behavior is the strongest driver of default risk
- High debt ratios and credit utilization significantly increase default probability
- Risk-based decision rules effectively separate low-risk and high-risk borrowers
- Linking interest rates to risk improves the risk–return tradeoff

---

## Project Structure


---

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Applications

This project is directly relevant for roles in:
- Fintech lending and payments
- Credit risk and analytics teams
- Banking and NBFC risk management
- Business and consulting analytics

---

## Author

**Amitesh Srivastava**  
BS–MS Economics, IISER Bhopal  

---

## Note

The dataset used in this project is anonymized and publicly available.  
All analysis is conducted for educational and analytical purposes, following standard industry practices.
