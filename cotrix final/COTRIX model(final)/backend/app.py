
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "ai4i2020.csv")
METRICS_PATH = os.path.join(BASE, "model_metrics.json")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

FEATURES = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Temp difference", "Type_L", "Type_M"
]
NUMERIC = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Temp difference"
]
MODEL_NAMES = [
    "Logistic Regression", "SVM", "Decision Tree",
    "Random Forest", "Gradient Boosting", "XGBoost", "ANN"
]

app = Flask(__name__)
CORS(app)

models = {}
metrics = {}

def prepare_data():
    raw = pd.read_csv(DATA_PATH)
    df = raw.rename(columns={
        "Air temperature [K]": "Air temperature",
        "Process temperature [K]": "Process temperature",
        "Rotational speed [rpm]": "Rotational speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool wear",
        "Machine failure": "Machine failure",
    }).copy()

    # Drop identifiers and failure-mode columns because those columns leak the
    # target and should not be used as production inputs.
    drop_cols = [c for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df["Temp difference"] = df["Process temperature"] - df["Air temperature"]
    df = pd.get_dummies(df, columns=["Type"], prefix="Type", drop_first=True)

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])
    X_test_s[NUMERIC] = scaler.transform(X_test[NUMERIC])

    # Guarantee exact column order.
    X_train_s = X_train_s.reindex(columns=FEATURES, fill_value=0)
    X_test_s = X_test_s.reindex(columns=FEATURES, fill_value=0)

    joblib.dump(scaler, SCALER_PATH)
    return X_train_s, X_test_s, y_train, y_test

def train_models(X_train, y_train):
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, learning_rate_init=0.001
        )
    }

def build_models_and_metrics():
    global models, metrics
    X_train, X_test, y_train, y_test = prepare_data()
    candidates = train_models(X_train, y_train)
    new_metrics = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        models[name] = model
        try:
            joblib.dump(model, os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".pkl"))
        except Exception:
            pass

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        new_metrics[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
        }

    metrics = new_metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def load_or_train():
    # For reproducibility and version safety, Cotrix trains from the supplied
    # project data/configuration at startup. This also avoids stale .pkl files
    # created by a different scikit-learn version.
    build_models_and_metrics()

def make_input(payload):
    machine_type = str(payload.get("machine_type", "M")).upper()
    air = float(payload["air_temperature"])
    process = float(payload["process_temperature"])
    rpm = float(payload["rotational_speed"])
    torque = float(payload["torque"])
    wear = float(payload["tool_wear"])

    row = pd.DataFrame([{
        "Air temperature": air,
        "Process temperature": process,
        "Rotational speed": rpm,
        "Torque": torque,
        "Tool wear": wear,
        "Temp difference": process - air,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }], columns=FEATURES)

    scaler = joblib.load(SCALER_PATH)
    row[NUMERIC] = scaler.transform(row[NUMERIC])
    return row

@app.get("/health")
def health():
    return jsonify({
        "status": "online",
        "project": "COTRIX",
        "models": MODEL_NAMES
    })

@app.get("/api/metrics")
def api_metrics():
    return jsonify(metrics)

@app.post("/api/predict")
def api_predict():
    try:
        row = make_input(request.get_json(force=True))
        probabilities = {}
        for name in MODEL_NAMES:
            probabilities[name] = float(models[name].predict_proba(row)[:, 1][0] * 100)

        ensemble = float(np.mean(list(probabilities.values())))
        health_score = round(100 - ensemble, 1)

        if ensemble >= 60:
            risk = "Critical"
            recommendation = "Stop or inspect the machine immediately and schedule maintenance."
        elif ensemble >= 30:
            risk = "Warning"
            recommendation = "Schedule inspection and monitor the machine closely."
        else:
            risk = "Normal"
            recommendation = "Continue operation and routine monitoring."

        return jsonify({
            "models": probabilities,
            "ensemble": round(ensemble, 1),
            "health_score": health_score,
            "risk": risk,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No CSV file supplied."}), 400
    try:
        df = pd.read_csv(f)
        return jsonify({
            "success": True,
            "filename": f.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        })
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

if __name__ == "__main__":
    print("Starting COTRIX with 7 predictive-maintenance models...")
    load_or_train()
    print("Models ready:", ", ".join(MODEL_NAMES))
    print("COTRIX API running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
