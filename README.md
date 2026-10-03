# Real-Time UPI Fraud Detection

An end-to-end machine learning system that scores UPI/card transactions for fraud risk in milliseconds, with a cost-based decision threshold, explainability, a REST API and a demo dashboard.

## Run it (about 5 minutes)

```
pip install -r requirements.txt
python fraud_data.py        # 1. creates upi_transactions.csv and upi_features.csv
python train.py             # 2. trains models, writes model.joblib, metrics.json, plots
uvicorn api:app --reload    # 3. API docs at http://127.0.0.1:8000/docs
streamlit run dashboard.py  # 4. demo dashboard
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

**Important:** these numbers come from synthetic data where I designed the fraud patterns, so they show the pipeline works, not real-world performance. Say this openly in interviews. To strengthen the project, rerun the same pipeline on a real dataset (see below).

---

## What we did, and why

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

## Datasets to strengthen the project (download yourself)

1. **Kaggle: Credit Card Fraud Detection** (mlg-ulb/creditcardfraud). 284,807 real European card transactions, 0.17% fraud. Best for proving the pipeline on real data. Skip the engineered features and use the columns directly.
2. **Kaggle: PaySim synthetic mobile money** (ealaxi/paysim1). Mobile-money transfers, closest to UPI behaviour.
3. **Kaggle: IEEE-CIS Fraud Detection**. Large, rich e-commerce fraud data, harder and more realistic.

---

## Limitations (mention these, they show maturity)
- Synthetic data, so results are optimistic.
- Cost figures are assumptions.
- No concept drift handling: fraudsters adapt, so models need monitoring and retraining.
- In production, user history features would come from a low-latency feature store, not be recomputed per request.
- Real systems combine ML with rules, device fingerprinting and graph analysis, and send borderline cases to human review.

## Resume bullet (edit to match what you actually ran)
> Built an end-to-end UPI fraud detection system in Python on 100K simulated transactions (1.2% fraud): engineered behavioural features (velocity, amount-vs-user-average, device/beneficiary risk), compared Logistic Regression, Random Forest and Gradient Boosting (PR-AUC 0.82), tuned a cost-based decision threshold (96% recall, ~92% estimated loss reduction), and deployed a FastAPI scoring service (~10 ms per transaction) with a Streamlit dashboard.

---

# Interview questions and answers

**1. Explain your project in 30 seconds.**
I built a system that scores each UPI transaction for fraud risk in real time. I simulated transaction data, engineered behavioural features such as how much a transaction deviates from the user's normal spend and how many transactions they made in the last hour, trained and compared several models, chose a decision threshold based on the financial cost of missed fraud versus false alarms, and exposed it through an API with a dashboard.

**2. Why not use accuracy?**
Fraud is about 1% of transactions, so a model that predicts "genuine" every time scores 99% accuracy while catching zero fraud. I used PR-AUC, precision and recall, which focus on the rare class.

**3. How did you handle class imbalance?**
Class weights, so errors on fraud are penalised more. Other options are SMOTE and undersampling. I chose weights because they don't create artificial data and are simple to explain. SMOTE can also hurt when applied before splitting, because it leaks information.

**4. Why a time-based split instead of a random one?**
In production the model predicts future transactions from past ones. A random split lets the model see future patterns and the same user's behaviour on both sides, which inflates results.

**5. What is data leakage and where did you avoid it?**
Leakage is when information from outside the prediction moment enters the features. My user average uses only earlier transactions (shifted before averaging), so the current transaction can't influence its own feature.

**6. Which features mattered most and why?**
Amount relative to the user's average, new beneficiary, new device and hour of day. This matches how fraud works: a stolen account is used differently from its owner's usual behaviour.

**7. How did you choose the threshold?**
By minimising expected cost: a missed fraud costs about Rs 5,000 and a false alarm about Rs 50. That gave a low threshold (0.06), so we catch 96% of fraud at the cost of low precision. These costs are assumptions, and a bank would plug in its real numbers. Low precision is acceptable if flagged cases get a cheap step-up check such as an OTP or PIN instead of a hard block.

**8. Your precision is only about 20%. Isn't that bad?**
It's a deliberate trade-off. Each false alarm costs little, while each missed fraud costs a lot. I'd also route mid-risk scores to step-up authentication instead of blocking, and high-risk ones to a block. The two-tier policy improves the customer experience.

**9. Why Random Forest or Gradient Boosting over Logistic Regression?**
They capture interactions (large amount AND new device AND night) without manual feature crossing, and performed better on PR-AUC. Logistic Regression remains useful as an interpretable baseline. The gap here is small, which is itself a finding: the features carry most of the signal.

**10. How would you explain a blocked transaction to a customer or regulator?**
Use feature attributions, for example SHAP values, to give reasons such as "unusually large amount, new device, new payee, late night". I used permutation importance globally and can use SHAP per transaction.

**11. What is the difference between supervised and unsupervised here? Why include Isolation Forest?**
Supervised models learn from labelled fraud. Isolation Forest finds statistical outliers with no labels, which helps with new fraud types that aren't labelled yet. It scored lower (0.65 vs 0.82), showing the value of labels, but it's useful as a complement.

**12. How would you make this truly real-time?**
Keep per-user stats (averages, 1h and 24h counts) in a low-latency store such as Redis, updated by a streaming pipeline (Kafka). The model loads once in memory and the API scores in milliseconds. My endpoint already reports latency.

**13. What happens when fraud patterns change?**
Concept drift. I'd monitor precision, recall and feature distributions, retrain on a schedule, and use champion-challenger testing before replacing the live model. Because fraud labels arrive late (chargebacks and complaints), I'd account for label delay.

**14. What are the limitations of your project?**
It uses synthetic data, so absolute numbers are optimistic. Cost values are assumptions. There are no graph features (shared devices and mule accounts) and no human-in-the-loop review. With real data, I'd validate on a public dataset like Kaggle's credit card fraud data.

**15. What would you do next?**
Add graph features to detect mule accounts, SHAP explanations per transaction, drift monitoring, and test on real datasets such as PaySim or the Kaggle credit card data.

**16. Why is this relevant to a bank like ICICI or Axis?**
Fraud directly affects losses, customer trust and regulatory compliance. The project shows I can connect ML to business outcomes: choosing metrics, thresholds and explanations based on cost and regulation, not just model accuracy.

**17. What is the difference between precision and recall in this context?**
Recall is the share of actual frauds we catch. Precision is the share of flagged transactions that are really fraud. High recall protects the bank's money, and high precision protects customer experience. The threshold balances the two.

**18. What is PR-AUC?**
The area under the precision-recall curve. It summarises performance across all thresholds and is more informative than ROC-AUC when the positive class is rare.
