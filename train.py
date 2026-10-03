"""
Steps 3-6: imbalance handling, model comparison, evaluation, cost-based threshold, explainability.
Run:  python train.py     (needs upi_features.csv from fraud_data.py)
Outputs: model.joblib, metrics.json, pr_curve.png, feature_importance.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------- 1. Load + time-based split ----------
df = pd.read_csv("upi_features.csv", parse_dates=["timestamp"]).sort_values("timestamp")
DROP = ["txn_id", "user_id", "timestamp", "is_fraud"]
FEATURES = [c for c in df.columns if c not in DROP]

# Train on the past, test on the future (random split would leak patterns and flatter the score)
cut = df["timestamp"].quantile(0.75)
train, test = df[df.timestamp <= cut], df[df.timestamp > cut]
X_tr, y_tr = train[FEATURES], train["is_fraud"]
X_te, y_te = test[FEATURES], test["is_fraud"]
print(f"Train: {len(train):,} (fraud {y_tr.mean():.2%}) | Test: {len(test):,} (fraud {y_te.mean():.2%})")

# ---------- 2. Imbalance handling: class weights ----------
# Fraud is ~1% of data, so we make a missed fraud "cost" more during training.
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                            min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                      scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=42)

# ---------- 3. Train + evaluate with PR-AUC ----------
results, scores = {}, {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    scores[name] = p
    results[name] = {"PR_AUC": round(average_precision_score(y_te, p), 4)}

# Unsupervised baseline: no labels used. Shows what supervised learning adds.
iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()), random_state=42).fit(X_tr)
iso_score = -iso.score_samples(X_te)
results["Isolation Forest (unsupervised)"] = {"PR_AUC": round(average_precision_score(y_te, iso_score), 4)}

best_name = max((k for k in models), key=lambda k: results[k]["PR_AUC"])
best, best_p = models[best_name], scores[best_name]
print("\nPR-AUC by model:")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]["PR_AUC"]):
    print(f"  {k:35s} {v['PR_AUC']}")
print(f"\nBest model: {best_name}")

# ---------- 4. Cost-based threshold ----------
# Business logic: missing a fraud costs far more than annoying a genuine user.
# Assumed costs (state these as assumptions in your README):
COST_MISSED_FRAUD = 5000   # Rs lost per undetected fraud (approx. average fraud amount)
COST_FALSE_ALARM = 50      # Rs cost per blocked genuine transaction (support + customer friction)

thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    pred = (best_p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    costs.append(fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM)
best_t = float(thresholds[int(np.argmin(costs))])
pred = (best_p >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
no_model_cost = int(y_te.sum() * COST_MISSED_FRAUD)
model_cost = int(fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM)

summary = {
    "best_model": best_name,
    "pr_auc_by_model": results,
    "threshold": round(best_t, 2),
    "precision": round(precision_score(y_te, pred), 3),
    "recall": round(recall_score(y_te, pred), 3),
    "f1": round(f1_score(y_te, pred), 3),
    "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "cost_without_model_rs": no_model_cost,
    "cost_with_model_rs": model_cost,
    "savings_pct": round(100 * (1 - model_cost / no_model_cost), 1),
    "features": FEATURES,
}
json.dump(summary, open("metrics.json", "w"), indent=2)
print(f"\nThreshold {best_t:.2f} -> precision {summary['precision']}, recall {summary['recall']}, F1 {summary['f1']}")
print(f"Confusion: {summary['confusion']}")
print(f"Estimated loss: Rs {no_model_cost:,} without model vs Rs {model_cost:,} with model ({summary['savings_pct']}% lower)")

# ---------- 5. Plots ----------
plt.figure(figsize=(6, 4.5))
for name, p in scores.items():
    pr, rc, _ = precision_recall_curve(y_te, p)
    plt.plot(rc, pr, label=f"{name} ({results[name]['PR_AUC']})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("pr_curve.png", dpi=150); plt.close()

# ---------- 6. Explainability ----------
# Permutation importance: shuffle one feature and see how much PR-AUC drops.
# (If you install shap, you can also do: shap.TreeExplainer(best).shap_values(X_te) for per-transaction reasons.)
sample = X_te.sample(min(5000, len(X_te)), random_state=1)
imp = permutation_importance(best, sample, y_te.loc[sample.index], scoring="average_precision",
                             n_repeats=3, random_state=1, n_jobs=-1)
order = np.argsort(imp.importances_mean)[-10:]
plt.figure(figsize=(6, 4.5))
plt.barh(np.array(FEATURES)[order], imp.importances_mean[order])
plt.title("Top features (permutation importance)"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150); plt.close()
print("\nTop features:", [FEATURES[i] for i in order[::-1][:5]])

joblib.dump({"model": best, "features": FEATURES, "threshold": best_t}, "model.joblib")
print("Saved model.joblib, metrics.json, pr_curve.png, feature_importance.png")
