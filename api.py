"""
Step 7a: Real-time scoring API.
Run:  uvicorn api:app --reload      then open http://127.0.0.1:8000/docs to try it.
"""
import time
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

art = joblib.load("model.joblib")
model, FEATURES, THRESHOLD = art["model"], art["features"], art["threshold"]

app = FastAPI(title="UPI Fraud Detection API")


class Txn(BaseModel):
    amount: float
    hour: int                      # 0-23
    category: str = "p2p"          # grocery, food, bills, shopping, travel, p2p, gaming, electronics
    new_device: int = 0
    new_beneficiary: int = 0
    user_avg_amt: float = 500.0    # in production: looked up from a feature store
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    secs_since_last: float = 86400
    is_weekend: int = 0


def to_row(t: Txn) -> pd.DataFrame:
    row = dict.fromkeys(FEATURES, 0)
    row.update(amount=t.amount, hour=t.hour, is_night=int(0 <= t.hour <= 5), is_weekend=t.is_weekend,
               new_device=t.new_device, new_beneficiary=t.new_beneficiary, user_avg_amt=t.user_avg_amt,
               amt_vs_avg=t.amount / max(t.user_avg_amt, 1), txn_count_1h=t.txn_count_1h,
               txn_count_24h=t.txn_count_24h, secs_since_last=t.secs_since_last)
    col = f"category_{t.category}"
    if col in row:
        row[col] = 1
    return pd.DataFrame([row])[FEATURES]


@app.post("/score")
def score(t: Txn):
    start = time.perf_counter()
    p = float(model.predict_proba(to_row(t))[0, 1])
    decision = "BLOCK / STEP-UP AUTH" if p >= THRESHOLD else "ALLOW"
    return {"fraud_probability": round(p, 4), "decision": decision,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}
