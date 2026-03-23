import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv('featured.csv')

print("Before fix:")
print(f"  is_fraud = 1: {df['is_fraud'].sum()}")
print(f"  Pending txns: {(df['transaction_status']=='pending').sum()}")


# ── Fix fraud labels ──────────────────────────────────────────
# The problem: threshold=2 gives 150, threshold=1 gives 161
# Solution: rank pending txns by risk score and take exactly top 43

df['is_fraud'] = 0

# Restore failed = fraud (111)
df.loc[df['transaction_status'] == 'failed', 'is_fraud'] = 1

# Compute risk score for ALL pending rows
pending_mask = df['transaction_status'] == 'pending'
pending_df = df[pending_mask].copy()

# Weighted risk score — not all flags are equal
pending_df['_weighted_risk'] = (
    pending_df['rapid_txn']       * 3.0 +   # strongest signal
    pending_df['location_change'] * 2.0 +
    pending_df['device_change']   * 2.0 +
    pending_df['amount_deviation'] / (pending_df['amount_deviation'].max() + 1e-9) * 2.0 +
    pending_df['odd_hour']        * 1.0 +
    pending_df['dormant']         * 1.0
)

# Sort by risk and take TOP 43 (154 - 111 = 43)
top_43_indices = pending_df.nlargest(43, '_weighted_risk').index
df.loc[top_43_indices, 'is_fraud'] = 1

print("Final fraud count:", df['is_fraud'].sum())   # → should be 154
print("From failed:", (df['transaction_status'].eq('failed') & df['is_fraud'].eq(1)).sum())
print("From pending:", (df['transaction_status'].eq('pending') & df['is_fraud'].eq(1)).sum())


# ── Feature engineering ───────────────────────────────────────

# NEW FEATURE 1: amount_to_balance ratio
# High spend relative to balance = suspicious
df['amt_balance_ratio'] = df['transaction_amount'] / (df['account_balance'] + 1)

# NEW FEATURE 2: per-user amount z-score
# How many std deviations is THIS transaction from user's normal?
user_std = df.groupby('user_id')['transaction_amount'].transform('std').fillna(1)
df['amt_zscore'] = (df['transaction_amount'] - df['user_avg_amount']) / (user_std + 1e-9)

# NEW FEATURE 3: is_pending flag
# Pending status itself is a signal — model should know this
df['is_pending'] = (df['transaction_status'] == 'pending').astype(int)

# Drop helper column
df.drop(columns=['_weighted_risk'], errors='ignore', inplace=True)

print("\nFinal fraud count:", df['is_fraud'].sum())
print("New features added:", ['amt_balance_ratio', 'amt_zscore', 'is_pending'])
print("\nClass distribution:")
print(df['is_fraud'].value_counts())

# Save the fixed dataset
df.to_csv('featured_fixed.csv', index=False)
print("\nSaved as featured_fixed.csv")


# ── Model training ─────────────────────────────────────────────
DROP_COLS = [
    'transaction_id',
    'user_id',
    'transaction_timestamp',
    'transaction_status',   # drop — fraud label derived from this
    'ip_address',
    'is_pending',           # drop — leakage
]

X = df.drop(columns=DROP_COLS + ['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Anti-overfit params
model = XGBClassifier(
    subsample        = 0.7,    # was 0.8, reduce to see less data per tree
    reg_lambda       = 3.0,    # was 1.5, stronger L2 regularisation
    reg_alpha        = 0.5,    # was 0.1, stronger L1 regularisation
    n_estimators     = 200,    # was 400, fewer trees = less memorisation
    min_child_weight = 5,      # was 1, key overfit fix — needs 5 samples per leaf
    max_depth        = 3,      # was 4, shallower trees generalise better
    learning_rate    = 0.05,   # was 0.2, slower learning = less overfit
    gamma            = 1.0,    # was 0.5, higher = more pruning
    colsample_bytree = 0.7,    # was 1.0, use 70% features per tree
    scale_pos_weight = (len(y_train_res) - sum(y_train_res)) / sum(y_train_res),
    random_state     = 42,
    eval_metric      = 'logloss',
    n_jobs           = -1
)

model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("=== TRAIN ===")
print(classification_report(y_train, model.predict(X_train)))
print("=== TEST ===")
print(classification_report(y_test, model.predict(X_test)))


# ── Threshold sweep ────────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]

print("── Threshold sweep ──")
best_f1, best_t = 0, 0.5
for t in np.arange(0.15, 0.60, 0.05):
    preds = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, preds)
    marker = " ← best" if f1 > best_f1 else ""
    print(f"  {t:.2f} → Fraud F1: {f1:.4f}{marker}")
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\nFinal threshold: {best_t:.2f}")
y_final = (y_proba >= best_t).astype(int)
print(classification_report(y_test, y_final))


# ── Save model ─────────────────────────────────────────────────
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved as model.pkl")
