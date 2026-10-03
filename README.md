# Real-Time UPI Fraud Detection

An end-to-end machine learning system that scores UPI/card transactions for fraud risk in milliseconds, with a cost-based decision threshold, explainability and a demo dashboard.

## Run it (about 5 minutes)

```
pip install -r requirements.txt
python fraud_data.py        # 1. creates upi_transactions.csv and upi_features.csv
python train.py             # 2. trains models, writes model.joblib, metrics.json, plots
streamlit run dashboard.py  # 3. demo dashboard
```

## Results (time-based test set, 25,000 transactions, ~1.2% fraud)

| Model | PR-AUC |
|---|---|
| Random Forest (best) | 0.821 |
| Gradient Boosting | 0.820 |
| XGBoost | 0.816 |
| Logistic Regression | 0.808 |
| Isolation Forest (unsupervised) | 0.653 |

At the cost-optimal threshold (0.06): recall 96.2%, precision 19.9%, and estimated fraud loss reduced by about 92% (Rs 14.65 lakh to Rs 1.12 lakh) under stated cost assumptions.

**Note:** Results are on simulated data with fraud patterns I designed, so they demonstrate the pipeline rather than real-world performance. Next step: validate on real datasets such as Kaggle Credit Card Fraud or PaySim.

---

## How it works?

### Stage 1: Data (`fraud_data.py`)
**What:** Generated 100,000 UPI-style transactions with ~1.2% fraud. Fraud is injected with realistic behaviour: much larger amounts than the user's norm, night-time activity, new devices, new beneficiaries.
**Why:** Real bank data is confidential, and public datasets are anonymised (their features are PCA outputs, so they have no UPI meaning). Synthetic data lets us build features that make business sense, and we control the fraud patterns.

### Stage 2: Feature engineering (`fraud_data.py`)
**What:** Built behavioural features: amount relative to the user's own average, transaction count in the last 1h and 24h (velocity), time since last transaction, night and weekend flags, new device, new beneficiary, category.
**Why:** Fraud is defined by deviation from a user's normal behaviour. A Rs 5,000 payment is normal for one person and suspicious for another. The raw amount alone says little, which is why `amt_vs_avg` ends up as the top feature.
**Leakage care:** the user's average uses only past transactions (`shift()` before the expanding mean). Including the current transaction would leak the answer.

### Stage 3: Handling imbalance (`train.py`)
**What:** Class weights (`balanced`, `scale_pos_weight`).
**Why:** With 1% fraud, a model that always says "genuine" gets 99% accuracy and catches nothing. Weighting makes each fraud example count more during training. Alternatives are SMOTE (synthetic oversampling) and undersampling.

### Stage 4: Models (`train.py`)
**What:** Logistic Regression (simple, interpretable baseline), Random Forest, Gradient Boosting (and XGBoost if installed), plus Isolation Forest (unsupervised).
**Why:** Always start with a baseline. Tree ensembles capture non-linear interactions (for example, large amount AND new device AND night). The unsupervised model shows the value of labelled data and is useful when labels are scarce, since new fraud types appear before they are labelled.

### Stage 5: Evaluation (`train.py`)
**What:** Time-based split (train on the first 75% of time, test on the last 25%) and PR-AUC as the main metric.
**Why:** Accuracy and ROC-AUC look good on imbalanced data even for poor models. Precision-recall focuses on the rare class. A time-based split mimics deployment (predict the future from the past), while a random split leaks and inflates scores.

### Stage 6: Cost-based threshold (`train.py`)
**What:** Chose the probability cutoff that minimises total loss, assuming Rs 5,000 lost per missed fraud and Rs 50 per false alarm.
**Why:** The default 0.5 threshold ignores business reality. Missing fraud costs far more than a blocked genuine payment, so the optimal threshold is low. This is the most "banker-like" part of the project: it connects the model to money. The cost figures are assumptions; change them and the threshold adapts.

### Stage 7: Explainability (`train.py`)
**What:** Permutation importance (shuffle a feature, measure the PR-AUC drop). Optional: SHAP for per-transaction reasons.
**Why:** Banks are regulated and must justify decisions to customers, auditors and the RBI. A black-box block is hard to defend.

### Stage 8: Serving (`api.py`, `dashboard.py`)
**What:** A FastAPI `/score` endpoint returning probability, decision and latency, plus a Streamlit dashboard.
**Why:** "Real-time" means a payment must be scored in milliseconds before approval. The API shows the model can be deployed; the dashboard makes the demo visual.

---

## Next steps: validating on real data
1. **Kaggle: Credit Card Fraud Detection** (mlg-ulb/creditcardfraud). 284,807 real European card transactions, 0.17% fraud. Best for proving the pipeline on real data.
2. **Kaggle: PaySim synthetic mobile money** (ealaxi/paysim1). Mobile-money transfers, closest to UPI behaviour.
3. **Kaggle: IEEE-CIS Fraud Detection**. Large, rich e-commerce fraud data, harder and more realistic.

---

## Limitations
- Synthetic data, so results are optimistic.
- Cost figures are assumptions.
- No concept drift handling: fraudsters adapt, so models need monitoring and retraining.
- In production, user history features would come from a low-latency feature store, not be recomputed per request.
- Real systems combine ML with rules, device fingerprinting and graph analysis, and send borderline cases to human review.


---

