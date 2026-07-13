# 🛡️ LoanShield — Lending Fraud & Default Detection Engine

> Can an ensemble model flag high-risk loan applicants before 
> disbursement — and quantify exactly how much it saves the lender?

---

## Problem Statement
Banks and lending institutions lose billions annually to loan defaults.
Most ML approaches optimise for accuracy — but a False Negative 
(approving a bad loan) costs ~$18,500 in lost principal, while a 
False Positive (rejecting a good borrower) costs only ~$1,200 in 
missed interest. This project treats fraud detection as a 
**business cost minimisation problem**, not an accuracy contest.

---

## Dataset
- **Source:** [HMEQ — Home Equity Loan Default (Kaggle)](https://www.kaggle.com/datasets/ajay1735/hmeq-data)
- **Size:** 5,960 rows × 13 features
- **Target:** `BAD` — 1 = defaulted / severely delinquent, 0 = repaid
- **Note:** Raw data not committed. Download `hmeq.csv` from Kaggle 
  and place in `data/` before running notebooks.

---

## Approach

**Act 1 — Fraud Landscape (Notebook 01)**
- Exploratory data analysis — class imbalance, feature distributions
- Class-conditional median imputation (not global median — preserves 
  the distributional difference between fraud and legit borrowers)
- Stratified train/test split

**Act 2 — Ensemble Showdown (Notebook 02)**
- Decision Tree baseline — exposes high variance problem
- Random Forest (Bagging) — 200 bootstrap trees, √n feature subsets, 
  uncorrelated errors collapse variance
- XGBoost (Boosting) — sequential residual correction, 
  `scale_pos_weight` derived from actual class ratio

**Act 3 — Business Verdict (Notebook 02)**
- Cost matrix: FN = $18,500 loss / FP = $1,200 loss
- 90-threshold sweep to find cost-minimising decision boundary
- Feature importance comparison: RF (MDI) vs XGBoost (Gain)

---

## Key Results


| Model | ROC-AUC | CV Mean AUC | CV Std |
|-------|---------|-------------|--------|
| Decision Tree | 0.9005 | 0.9209 | 0.0222 |
| Random Forest | 0.9810 | 0.9814 | 0.0026 |
| **XGBoost** | **0.9757** | **0.9747** | **0.0052** |


> Fill in your actual numbers after running the notebook.

**Business impact:** Threshold optimisation at `t=X.XX` reduced 
projected lending losses by $X,XXX vs the default 0.50 cutoff.

---

## Dashboard Previews

**Act 1 — Fraud Landscape**
![Fraud Landscape](outputs/figures/act1_fraud_landscape.png)

**Act 3 — Business Verdict**
![Business Verdict](outputs/figures/act3_business_verdict.png)

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/loanshield-fraud-detection
cd loanshield-fraud-detection
pip install -r requirements.txt
```

1. Download `hmeq.csv` from Kaggle (link above)
2. Place it in the `data/` folder
3. Run `notebooks/01_eda_fraud_landscape.ipynb`
4. Run `notebooks/02_modeling_ensemble_verdict.ipynb`

---

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`

---

## LinkedIn Posts
- [Mid-build post — EDA & Cleaning](https://www.linkedin.com/posts/prajwal-kapnoor-50042730a_datascience-machinelearning-fintech-ugcPost-7479943391854788608-5ivi/)
- [Completion post — Full Results]()
