from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
from typing import List
import logging
from datetime import datetime

app = FastAPI(title="SentinelNet NIDS API")

# ────────────────────────────────────────────────────────────────
#  Paths relative to backend/ folder (models/ is sibling folder)
# ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")             # one level up → models/

MODEL_PATH = os.path.join(MODELS_DIR, "best_xgboost_tuned.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")            # or "standard_scaler.pkl"
FEATURES_PATH = os.path.join(MODELS_DIR, "selected_features.json")
ISO_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")

# Debug: print paths so you can see what's being looked for
print("Base dir (backend):", BASE_DIR)
print("Models dir:", MODELS_DIR)
print("Model file exists?", os.path.exists(MODEL_PATH))
print("Scaler file exists?", os.path.exists(SCALER_PATH))
print("Features file exists?", os.path.exists(FEATURES_PATH))

# ────────────────────────────────────────────────────────────────
#  Load artifacts
# ────────────────────────────────────────────────────────────────

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
except Exception as e:
    print("Error loading artifacts:", str(e))
    raise

# Optional: Isolation Forest for hybrid mode
iso_forest = None
if os.path.exists(ISO_FOREST_PATH):
    try:
        iso_forest = joblib.load(ISO_FOREST_PATH)
        print("Isolation Forest loaded → hybrid mode enabled")
    except Exception as e:
        print("Could not load Isolation Forest:", str(e))

# ────────────────────────────────────────────────────────────────
#  Logging / Alerts
# ────────────────────────────────────────────────────────────────

# Log alerts to models/alerts.log (same folder as artifacts)
LOG_PATH = os.path.join(MODELS_DIR, "alerts.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def generate_alert(flow_data, prediction, prob, model_name='XGBoost'):
    if prediction == 1:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = (
            f"ALERT: Potential Attack Detected! "
            f"Model: {model_name} | "
            f"Probability: {prob:.4f} | "
            f"Time: {timestamp} | "
            f"Flow sample: {flow_data[:5]}..."
        )
        logging.info(msg)
        print(msg)  # also show in console during dev

# ────────────────────────────────────────────────────────────────
#  API Models
# ────────────────────────────────────────────────────────────────

class FlowInput(BaseModel):
    features: List[float]  # must be exactly 30 values

# ────────────────────────────────────────────────────────────────
#  Endpoints
# ────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(flow: FlowInput):
    if len(flow.features) != len(selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {len(selected_features)} features, got {len(flow.features)}"
        )

    try:
        X = np.array(flow.features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Supervised prediction (XGBoost)
        pred_sup = model.predict(X_scaled)[0]
        prob_sup = float(model.predict_proba(X_scaled)[0][1])

        # Optional hybrid with Isolation Forest
        if iso_forest:
            pred_unsup = iso_forest.predict(X_scaled)[0]  # -1 = anomaly
            pred_unsup_bin = 1 if pred_unsup == -1 else 0
            final_pred = 1 if (pred_sup == 1 or pred_unsup_bin == 1) else 0
            # Rough combined probability
            anomaly_score = -iso_forest.decision_function(X_scaled)[0]
            final_prob = max(prob_sup, anomaly_score)
            model_used = "Hybrid (XGBoost + Isolation Forest)"
        else:
            final_pred = pred_sup
            final_prob = prob_sup
            model_used = "XGBoost (tuned)"

        label = "Attack" if final_pred == 1 else "Benign"

        # Generate alert if attack
        generate_alert(flow.features, final_pred, final_prob, model_used)

        return {
            "prediction": label,
            "attack_probability": final_prob,
            "model_used": model_used,
            "features_count": len(selected_features)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


# ────────────────────────────────────────────────────────────────
#  START THE SERVER (this must be at the very end of main.py)
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")