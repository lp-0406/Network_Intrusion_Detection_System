# app.py
import streamlit as st
import joblib
import numpy as np
import json
import pandas as pd
import os
import requests
from datetime import datetime

# ────────────────────────────────────────────────────────────────
# Page config – professional look for presentation
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelNet – Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=90)
with col2:
    st.title("SentinelNet")
    st.markdown("**Advanced Hybrid Network Intrusion Detection System**")
    st.caption("Tuned XGBoost + Isolation Forest | CIC-IDS-2017 Dataset")

# Sidebar
with st.sidebar:
    st.header("SentinelNet")
    st.markdown("""
    Detects **Benign** vs **Attack** network flows using:
    - Tuned XGBoost (supervised)
    - Isolation Forest (unsupervised anomaly detection)
    - Hybrid mode for maximum recall
    """)
    st.markdown("---")
    st.info("Dataset: CIC-IDS-2017 (~1.91M cleaned flows)")
    st.markdown("""
    **Links**  
    - [GitHub Repository](https://github.com/lp-0406/Network_Intrusion_Detection_System)  
    """)
    st.markdown("Built with ❤️ by Prashasthi G")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict Single Flow", "Batch Upload (CSV)", "Model Details & Metrics"])

# ────────────────────────────────────────────────────────────────
# Load models (cached + download from HF if missing)
# ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    HF_REPO = "zealcatz/sentinelnet-models"  # ← your actual repo

    def download(url, path):
        if not os.path.exists(path):
            with st.spinner(f"Downloading {os.path.basename(path)}..."):
                r = requests.get(url, allow_redirects=True)
                r.raise_for_status()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(r.content)

    download(f"https://huggingface.co/{HF_REPO}/resolve/main/best_xgboost_tuned.pkl", "models/best_xgboost_tuned.pkl")
    download(f"https://huggingface.co/{HF_REPO}/resolve/main/scaler.pkl", "models/scaler.pkl")
    download(f"https://huggingface.co/{HF_REPO}/resolve/main/selected_features.json", "models/selected_features.json")
    download(f"https://huggingface.co/{HF_REPO}/resolve/main/isolation_forest.pkl", "models/isolation_forest.pkl")

    model = joblib.load("models/best_xgboost_tuned.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/selected_features.json", 'r') as f:
        features = json.load(f)
    iso_forest = joblib.load("models/isolation_forest.pkl") if os.path.exists("models/isolation_forest.pkl") else None

    return model, scaler, features, iso_forest

model, scaler, selected_features, iso_forest = load_models()

# ────────────────────────────────────────────────────────────────
# Alert function
# ────────────────────────────────────────────────────────────────
def generate_alert(features_list, prediction, prob, model_name='XGBoost'):
    if prediction == 1:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"🚨 ALERT: Attack Detected! Model: {model_name} | Prob: {prob:.4f} | Time: {timestamp}"
        st.error(msg)
        with open("models/alerts.log", "a") as f:
            f.write(f"{timestamp} - {msg}\n")

# ────────────────────────────────────────────────────────────────
# Tab 1: Single Flow Prediction
# ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Single Flow Prediction")
    st.write("Paste 30 comma-separated feature values below (from test_samples.txt)")

    features_input = st.text_area(
        "Features (comma-separated)",
        height=120,
        placeholder="80.0,2.0,0.0,40.0,... (exactly 30 values)"
    )

    if st.button("Run Prediction", type="primary"):
        try:
            feat_list = [float(x.strip()) for x in features_input.split(',') if x.strip()]
            if len(feat_list) != 30:
                st.error(f"Exactly 30 values required. You entered {len(feat_list)}.")
            else:
                X = np.array(feat_list).reshape(1, -1)
                X_scaled = scaler.transform(X)

                pred_sup = model.predict(X_scaled)[0]
                prob_sup = float(model.predict_proba(X_scaled)[0][1])

                if iso_forest:
                    pred_unsup = iso_forest.predict(X_scaled)[0]
                    pred_unsup_bin = 1 if pred_unsup == -1 else 0
                    final_pred = 1 if pred_sup or pred_unsup_bin else 0
                    anomaly_score = -iso_forest.decision_function(X_scaled)[0]
                    final_prob = max(prob_sup, anomaly_score)
                    model_used = "Hybrid (XGBoost + Isolation Forest)"
                else:
                    final_pred = pred_sup
                    final_prob = prob_sup
                    model_used = "XGBoost (tuned)"

                label = "Attack" if final_pred == 1 else "Benign"

                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", label, delta_color="inverse" if label == "Attack" else "normal")
                col2.metric("Attack Probability", f"{final_prob:.4f}", delta_color="inverse" if label == "Attack" else "normal")
                col3.metric("Model", model_used)

                generate_alert(feat_list, final_pred, final_prob, model_used)

                st.progress(final_prob)
                st.caption(f"Attack confidence: {final_prob:.1%}")

                result = {
                    "prediction": label,
                    "attack_probability": final_prob,
                    "model_used": model_used,
                    "features": feat_list
                }
                st.download_button(
                    "Download result as JSON",
                    json.dumps(result, indent=2),
                    "prediction_result.json",
                    "application/json"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ────────────────────────────────────────────────────────────────
# Tab 2: Batch Prediction (CSV)
# ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction (CSV Upload)")
    st.write("Upload a CSV file with **exactly 30 columns** (no header needed)")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] != 30:
                st.error(f"CSV must have exactly 30 columns. Found {df.shape[1]}.")
            else:
                st.success(f"Loaded {len(df)} flows. Predicting...")

                X_batch = scaler.transform(df.values)

                pred_sup = model.predict(X_batch)
                prob_sup = model.predict_proba(X_batch)[:, 1]

                if iso_forest:
                    pred_unsup = iso_forest.predict(X_batch)
                    pred_unsup_bin = np.where(pred_unsup == -1, 1, 0)
                    final_pred = np.logical_or(pred_sup, pred_unsup_bin).astype(int)
                    anomaly_scores = -iso_forest.decision_function(X_batch)
                    final_prob = np.maximum(prob_sup, anomaly_scores)
                    model_used = "Hybrid"
                else:
                    final_pred = pred_sup
                    final_prob = prob_sup
                    model_used = "XGBoost"

                results_df = pd.DataFrame({
                    "Flow #": range(1, len(df) + 1),
                    "Prediction": ["Attack" if p == 1 else "Benign" for p in final_pred],
                    "Probability": [f"{p:.4f}" for p in final_prob]
                })

                st.subheader("Prediction Results")
                st.dataframe(results_df.style.apply(
                    lambda x: ['background-color: #ffcccc' if v == "Attack" else '' for v in x],
                    axis=0,
                    subset=["Prediction"]
                ))

                attacks = (final_pred == 1).sum()
                st.metric("Attacks Detected", attacks, delta=f"{(attacks / len(df)) * 100:.1f}% of flows")

                if attacks > 0:
                    st.error(f"🚨 {attacks} potential attacks detected! Check alerts.log for details.")
                    for i, p in enumerate(final_pred):
                        if p == 1:
                            generate_alert(df.iloc[i].tolist(), 1, final_prob[i], model_used)

                # Export results as CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# ────────────────────────────────────────────────────────────────
# Tab 3: Model Details & Metrics
# ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Overview & Performance")

    # Class Imbalance
    st.markdown("### Class Imbalance in Dataset")
    st.write("""
    The CIC-IDS-2017 dataset is highly imbalanced:
    - **Benign flows**: ~88%
    - **Attack flows**: ~12%
    
    This imbalance can bias models toward predicting Benign.
    **Solution used**: SMOTE oversampling on the training set only (50/50 balance after resampling).
    """)

    # Class distribution image
    if os.path.exists("images/class_distribution.png"):
        st.image("images/class_distribution.png", caption="Class Distribution Before & After SMOTE", use_column_width=True)
    else:
        st.info("Add 'class_distribution.png' to images/ folder for visualization")

    # All models table
    st.markdown("### All Models Evaluated")
    st.write("Performance on the test set (focus on Attack class):")

    data = {
        "Model": ["Decision Tree", "Random Forest", "LinearSVC", "Logistic Regression", "XGBoost (untuned)", "XGBoost (tuned)"],
        "Accuracy": [0.9977, 0.9964, 0.9159, 0.9198, 0.9967, 0.9968],
        "Precision (Attack)": [0.9948, 0.9752, 0.5938, 0.6049, 0.9749, 0.9761],
        "Recall (Attack)": [0.9865, 0.9953, 0.9730, 0.9774, 0.9983, 0.9979],
        "F1 (Attack)": [0.9906, 0.9851, 0.7375, 0.7473, 0.9865, 0.9869],
        "Train Time (s)": ["~26", "~109", "~127", "~43", "~23", "~23"]
    }

    df_metrics = pd.DataFrame(data)
    st.dataframe(
        df_metrics.style
        .highlight_max(subset=["F1 (Attack)", "Recall (Attack)"], color="#d4edda")
        .format(precision=4)
    )

    st.markdown("**Best Model**: Tuned XGBoost — highest recall on attacks (critical for IDS) with strong F1 score.")

    # Visuals gallery
    st.subheader("Evaluation Visuals")

    visuals = [
        ("confusion_matrix_tuned.png", "Confusion Matrix (Tuned XGBoost)"),
        ("feature_importance_tuned.png", "Top 15 Feature Importances"),
        ("roc_pr_tuned.png", "ROC & Precision-Recall Curves"),
        ("class_distribution.png", "Class Distribution Before & After SMOTE")
    ]

    cols = st.columns(2)
    for idx, (filename, caption) in enumerate(visuals):
        path = f"images/{filename}"
        if os.path.exists(path):
            with cols[idx % 2]:
                st.image(path, caption=caption, use_column_width=True)
        else:
            with cols[idx % 2]:
                st.info(f"Image '{filename}' not found – add to images/ folder")

