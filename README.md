cat >> README.md <<'EOF'
# Credit Card Fraud Decision Engine

A banking-oriented fraud detection and decisioning system designed around extreme class imbalance, temporal validation, cost-sensitive decisioning, investigator capacity, explainability, and model drift monitoring.

## Project Overview

Credit card fraud detection is not simply a classification problem.

A bank does not ultimately need:

> "Is this transaction fraud?"

It needs:

> "Given the estimated fraud risk, customer friction, financial loss, and limited investigator capacity, what action should we take?"

This project develops a fraud decision engine that moves from transaction-level prediction to operational decisioning.

```text
Transaction
     ↓
Fraud Detection Model
     ↓
Fraud Probability
     ↓
Decision Engine
     ↓
APPROVE / REVIEW
     ↓
Investigator Queue
     ↓
SHAP Reason Codes
     ↓
PSI Monitoring### 5. SMOTE Comparison

SMOTE was evaluated as an alternative to class weighting.

Validation PR-AUC:

| Model | Validation PR-AUC |
|---|---:|
| Class-Weighted XGBoost | 0.8618 |
| SMOTE XGBoost | 0.8475 |

Class-weighted XGBoost was selected as the final model.

### 6. Threshold Optimization

Fraud probability is converted into an operational decision.

```text
P(Fraud) >= 0.56 → REVIEW
P(Fraud) <  0.56 → APPROVE
## Final Out-of-Time Test Results

| Metric | Result |
|---|---:|
| Validation PR-AUC | 0.8618 |
| Test PR-AUC | 0.7684 |
| Test ROC-AUC | 0.9797 |
| Test Precision | 86.67% |
| Test Recall | 75.00% |
| Test F1 | 80.41% |
| Review Alerts | 45 |
| True Positives | 39 |
| False Positives | 6 |
| False Negatives | 13 |
| True Negatives | 42,664 |

At the frozen threshold of 0.56, the model identifies 39 of the 52 fraudulent transactions in the out-of-time test period while sending 45 transactions for investigation.

The test set was not used for model or threshold selection.

## Monitoring Results

### Model Score Stability

```text
Train → Validation PSI = 0.0383
Status = Stable

Train → Test PSI = 0.0865
Status = Stable
## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, SciPy, matplotlib, seaborn and Jupyter.

## Author

**Sreerag M.**

M.A. Environmental Economics  
Madras School of Economics

Interests:

- Quantitative Finance
- Risk Analytics
- Optimization
- Machine Learning
- Financial Economics
- Climate and Environmental Economics
