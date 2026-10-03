"""
Step 7b: Demo dashboard.
Run:  streamlit run dashboard.py
"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
art = joblib.load("model.joblib")
model, FEATURES, THRESHOLD = art["model"], art["features"], art["threshold"]
metrics = json.load(open("metrics.json"))

st.title("Real-Time UPI Fraud Detection")
tab1, tab2 = st.tabs(["Score a transaction", "Model performance"])

with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (Rs)", 1.0, 500000.0, 2500.0)
    avg = c1.number_input("User's usual average (Rs)", 1.0, 100000.0, 500.0)
    hour = c1.slider("Hour of day", 0, 23, 2)
    cat = c1.selectbox("Category", ["grocery", "food", "bills", "shopping", "travel", "p2p", "gaming", "electronics"], 5)
    new_dev = c2.checkbox("New / unrecognised device", True)
    new_ben = c2.checkbox("New beneficiary", True)
    c1h = c2.slider("Transactions by user in last 1 hour", 0, 15, 1)
    c24 = c2.slider("Transactions by user in last 24 hours", 0, 40, 2)

    row = dict.fromkeys(FEATURES, 0)
    row.update(amount=amount, hour=hour, is_night=int(hour <= 5), user_avg_amt=avg, amt_vs_avg=amount / avg,
               new_device=int(new_dev), new_beneficiary=int(new_ben), txn_count_1h=c1h,
               txn_count_24h=c24, secs_since_last=3600)
    if f"category_{cat}" in row:
        row[f"category_{cat}"] = 1
    p = model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1]

    st.metric("Fraud probability", f"{p:.1%}")
    if p >= THRESHOLD:
        st.error("BLOCK / ask for step-up authentication")
    else:
        st.success("ALLOW")
    st.caption(f"Decision threshold {THRESHOLD:.2f}, chosen to minimise expected financial loss.")

with tab2:
    a, b, c, d = st.columns(4)
    a.metric("Best model", metrics["best_model"])
    b.metric("Recall", metrics["recall"])
    c.metric("Precision", metrics["precision"])
    d.metric("Loss reduction", f"{metrics['savings_pct']}%")
    st.image("pr_curve.png", caption="Precision-Recall curves")
    st.image("feature_importance.png", caption="Top features")
