from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

EXPECTED_FEATURES = list(getattr(model, "feature_names_in_", []))


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def parse_numeric_series(series):
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_timestamp_series(series):
    as_text = series.astype(str).str.strip()
    parsed_default = pd.to_datetime(as_text, errors="coerce")

    compact = as_text.str.replace(r"[^0-9]", "", regex=True)
    compact_14 = compact.where(compact.str.len() == 14)
    parsed_compact = pd.to_datetime(compact_14, format="%Y%m%d%H%M%S", errors="coerce")

    numeric = pd.to_numeric(as_text, errors="coerce")
    valid_epoch_seconds = numeric.where((numeric >= 0) & (numeric <= 4102444800))
    parsed_epoch = pd.to_datetime(valid_epoch_seconds, unit="s", errors="coerce")

    return parsed_default.fillna(parsed_compact).fillna(parsed_epoch)


def safe_col(df, name, default=0):
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def build_feature_frame(df):
    work = normalize_columns(df)

    amount_source = safe_col(work, "transaction_amount", safe_col(work, "amount", 0))
    work["transaction_amount"] = parse_numeric_series(amount_source).fillna(0.0)

    balance_source = safe_col(work, "account_balance", 0)
    work["account_balance"] = parse_numeric_series(balance_source).fillna(0.0)

    timestamp_source = safe_col(work, "transaction_timestamp", None)
    work["transaction_timestamp"] = parse_timestamp_series(timestamp_source)

    if "user_id" not in work.columns:
        work["user_id"] = "unknown"

    work = work.sort_values(by=["user_id", "transaction_timestamp"], na_position="last")

    time_diff = work.groupby("user_id")["transaction_timestamp"].diff().dt.total_seconds().fillna(0)
    work["rapid_txn"] = (time_diff < 60).astype(int)
    work["dormant"] = (time_diff > 86400).astype(int)

    if "user_location" in work.columns:
        prev_loc = work.groupby("user_id")["user_location"].shift()
        work["location_change"] = (
            work["user_location"].astype(str).str.lower().str.strip() != prev_loc.astype(str).str.lower().str.strip()
        ).astype(int)
    else:
        work["location_change"] = 0

    if "device_id" in work.columns:
        prev_device = work.groupby("user_id")["device_id"].shift()
        work["device_change"] = (
            work["device_id"].astype(str).str.lower().str.strip() != prev_device.astype(str).str.lower().str.strip()
        ).astype(int)
    else:
        work["device_change"] = 0

    work["hour"] = work["transaction_timestamp"].dt.hour.fillna(-1).astype(int)
    work["odd_hour"] = ((work["hour"] >= 0) & ((work["hour"] <= 5) | (work["hour"] >= 22))).astype(int)

    work["user_avg_amount"] = work.groupby("user_id")["transaction_amount"].transform("mean").fillna(work["transaction_amount"])
    work["amount_deviation"] = (work["transaction_amount"] - work["user_avg_amount"]).abs()
    work["txn_count"] = work.groupby("user_id")["user_id"].transform("count").fillna(1)

    std_amount = work.groupby("user_id")["transaction_amount"].transform("std").fillna(1.0)
    work["amt_zscore"] = (work["transaction_amount"] - work["user_avg_amount"]) / (std_amount + 1e-9)
    work["amt_balance_ratio"] = work["transaction_amount"] / (work["account_balance"] + 1.0)

    for col in ["rapid_txn", "location_change", "device_change", "odd_hour", "dormant"]:
        if col in work.columns:
            work[col] = parse_numeric_series(work[col]).fillna(0).astype(int)

    if EXPECTED_FEATURES:
        X = pd.DataFrame(index=work.index)
        for feature in EXPECTED_FEATURES:
            if feature in work.columns:
                X[feature] = parse_numeric_series(work[feature]).fillna(0.0)
            else:
                X[feature] = 0.0
    else:
        fallback = [
            "transaction_amount",
            "rapid_txn",
            "location_change",
            "device_change",
            "odd_hour",
            "dormant",
            "amount_deviation",
            "txn_count",
        ]
        X = work[fallback].copy()

    return work, X


@app.route("/")
def home():
    return "Fraud Detection API Running ✅"


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "model_loaded": model is not None,
            "feature_count": len(EXPECTED_FEATURES) if EXPECTED_FEATURES else None,
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file provided"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as ex:
        return jsonify({"error": f"Failed to parse CSV: {str(ex)}"}), 400

    if df.empty:
        return jsonify({"total_transactions": 0, "fraud_detected": 0})

    try:
        _, X = build_feature_frame(df)
        preds = model.predict(X)
    except Exception as ex:
        return jsonify({"error": f"Prediction failed: {str(ex)}"}), 500

    fraud_count = int(preds.sum())
    total_rows = int(len(df))

    return jsonify(
        {
            "total_transactions": total_rows,
            "fraud_detected": fraud_count,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)