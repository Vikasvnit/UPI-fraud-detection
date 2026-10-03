"""
Step 1-2: Synthetic UPI transaction data + feature engineering.
Run:  python fraud_data.py   ->  creates upi_transactions.csv and upi_features.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_USERS = 2000
N_TXN = 100_000
FRAUD_RATE = 0.012
CATEGORIES = ["grocery", "food", "bills", "shopping", "travel", "p2p", "gaming", "electronics"]


def generate(n_txn=N_TXN, n_users=N_USERS):
    # Per-user profile: typical spend and usual device
    avg_amt = rng.lognormal(6.0, 0.6, n_users)  # ~Rs 400 median
    user_id = rng.integers(0, n_users, n_txn)
    start = pd.Timestamp("2026-01-01")
    ts = start + pd.to_timedelta(rng.integers(0, 90 * 24 * 3600, n_txn), unit="s")

    df = pd.DataFrame({
        "txn_id": np.arange(n_txn),
        "user_id": user_id,
        "timestamp": ts,
        "amount": rng.lognormal(np.log(avg_amt[user_id]), 0.7),
        "category": rng.choice(CATEGORIES, n_txn, p=[.2, .2, .15, .15, .08, .12, .05, .05]),
        "new_device": (rng.random(n_txn) < 0.03).astype(int),
        "new_beneficiary": (rng.random(n_txn) < 0.10).astype(int),
        "is_fraud": 0,
    })

    # Inject fraud with realistic patterns
    n_fraud = int(n_txn * FRAUD_RATE)
    idx = rng.choice(n_txn, n_fraud, replace=False)
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "amount"] *= rng.uniform(3, 12, n_fraud)
    df.loc[idx, "new_device"] = (rng.random(n_fraud) < 0.6).astype(int)
    df.loc[idx, "new_beneficiary"] = (rng.random(n_fraud) < 0.8).astype(int)
    night = rng.random(n_fraud) < 0.6  # most fraud at night
    new_hours = np.where(night, rng.integers(0, 5, n_fraud), df.loc[idx, "timestamp"].dt.hour)
    df.loc[idx, "timestamp"] = (
        df.loc[idx, "timestamp"].dt.normalize() + pd.to_timedelta(new_hours, unit="h")
        + pd.to_timedelta(rng.integers(0, 3600, n_fraud), unit="s")
    )
    df["amount"] = df["amount"].round(2)
    return df.sort_values("timestamp").reset_index(drop=True)


def engineer_features(df):
    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    g = df.groupby("user_id")
    # Past-only stats (shift avoids leaking the current transaction)
    df["user_avg_amt"] = g["amount"].transform(lambda s: s.shift().expanding().mean())
    df["amt_vs_avg"] = df["amount"] / df["user_avg_amt"]
    df["secs_since_last"] = g["timestamp"].diff().dt.total_seconds()

    # Velocity: transactions by this user in the previous 1 hour / 24 hours
    def velocity(group, window):
        t = group.set_index("timestamp")["amount"]
        return t.rolling(window).count().sub(1).values

    df["txn_count_1h"] = np.concatenate([velocity(x, "1h") for _, x in g])
    df["txn_count_24h"] = np.concatenate([velocity(x, "24h") for _, x in g])

    df = df.fillna({"amt_vs_avg": 1.0, "secs_since_last": 1e6, "user_avg_amt": df["amount"].median()})
    df = pd.get_dummies(df, columns=["category"], dtype=int)
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    raw = generate()
    raw.to_csv("upi_transactions.csv", index=False)
    feats = engineer_features(raw)
    feats.to_csv("upi_features.csv", index=False)
    print(f"Transactions: {len(raw):,} | Fraud: {raw.is_fraud.sum():,} ({raw.is_fraud.mean():.2%})")
    print("Feature columns:", [c for c in feats.columns if c not in ("txn_id", "user_id", "timestamp", "is_fraud")])
