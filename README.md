# SentinelNet — AI Powered Network Intrusion Detection System

> Binary classification IDS (Benign vs Attack) · CIC-IDS-2017 · 1.9M flows · 99.66% accuracy · FastAPI + Next.js · Deployed on Railway & Vercel

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-purple?logo=railway)](https://railway.app)
[![Vercel](https://img.shields.io/badge/Frontend%20on-Vercel-black?logo=vercel)](https://vercel.com)
[![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle)](https://kaggle.com)



## What Is This?

SentinelNet is a **production-grade Network Intrusion Detection System (NIDS)** that classifies real world network traffic as Benign or Attack in real time. It mirrors the design of enterprise IDS/IPS systems — built on the **CIC-IDS-2017** dataset from the Canadian Institute for Cybersecurity, covering ~1.9 million labelled network flow records across 8 attack types.

The project covers the **complete ML engineering lifecycle**: data ingestion at scale, memory optimization, cleaning, feature engineering, class imbalance handling, multi model benchmarking, hyperparameter tuning, unsupervised anomaly detection, hybrid ensemble inference, REST API deployment, and a live frontend demo.



## Results

### Supervised Model Benchmarks

| Model | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | Train Time |
|---|---|---|---|---|---|
| **XGBoost (best)** | **99.66%** | 97.49% | **99.78%** | **98.62%** | ~23s |
| Random Forest | 99.52% | 96.81% | 99.41% | 98.10% | ~45s |
| Decision Tree | 99.12% | 95.60% | 98.90% | 97.22% | ~12s |
| Logistic Regression | 97.10% | 93.20% | 96.80% | 94.97% | ~8s |
| LinearSVC | 96.80% | 92.40% | 96.10% | 94.21% | ~15s |

### Unsupervised (Isolation Forest)

| Model | Recall (Attack) | Notes |
|---|---|---|
| Isolation Forest | 35.3% | Trained on benign-only traffic; no labels used |

> **Why recall matters most for IDS:** A missed attack (false negative) is far more costly than a false alarm. XGBoost achieves **99.78% attack recall** — catching nearly all intrusions with very few misses.

### XGBoost Hyperparameter Tuning

Tuned via `RandomizedSearchCV` — 25 iterations, 3-fold CV, F1 scoring, on 40% of the SMOTE-balanced training set.

| Parameter | Best Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.8 |
| Best CV F1 | **0.9994** |

---

## Architecture

```
Raw CIC-IDS-2017 CSVs (8 files, ~1.9M rows)
        │
        ▼
┌─────────────────────────┐
│   Memory-Optimized      │  dtype downcasting, gc.collect(),
│   Data Ingestion        │  MAX_ROWS cap → ~40% memory reduction
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Data Cleaning         │  inf/NaN handling, median imputation,
│                         │  duplicate removal, label encoding
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Feature Engineering   │  SelectKBest (ANOVA F-test, k=30)
│                         │  + 3 ratio features engineered
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   SMOTE Balancing       │  Applied on training set only
│   (83%/17% → balanced)  │  to prevent data leakage
└────────────┬────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────┐  ┌──────────────┐
│ XGBoost  │  │ Isolation    │
│ (tuned)  │  │ Forest       │
│ 99.66%   │  │ benign-only  │
└────┬─────┘  └──────┬───────┘
     │               │
     └───── OR ──────┘
             │
             ▼
┌─────────────────────────┐
│   Hybrid Ensemble       │  logical_or → maximum recall
│   + Alert Logger        │  timestamp, probability, flow details
└────────────┬────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────────┐  ┌──────────────┐
│  FastAPI     │  │  Streamlit   │
│  /predict    │  │  Demo App    │
│  Railway     │  │              │
└──────┬───────┘  └──────────────┘
       │
       ▼
┌──────────────┐
│  Next.js UI  │
│  Vercel      │
└──────────────┘
```


## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Data | scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, numpy |
| Feature Selection | `SelectKBest` (ANOVA F-classif) |
| Anomaly Detection | `IsolationForest` |
| Hyperparameter Tuning | `RandomizedSearchCV` |
| Visualization | matplotlib, seaborn |
| API Backend | FastAPI 0.115+ |
| Frontend | Next.js |
| Deployment | Railway (API), Vercel (frontend), Streamlit (demo) |
| Artifact Storage | joblib, JSON |
| Training Platform | Kaggle Notebooks (T4 GPU, 30GB RAM) |
| Dataset | CIC-IDS-2017 — Canadian Institute for Cybersecurity |



## Project Structure

```
sentinelnet-ids/
├── notebooks/
│   └── sentinelnet-final.ipynb     # Full training pipeline
├── models/
│   ├── xgboost_model.pkl           # Tuned XGBoost classifier
│   ├── standard_scaler.pkl         # Fitted StandardScaler
│   ├── selected_features.json      # Top 30 feature names
│   └── isolation_forest.pkl        # Anomaly detector
├── data/
│   └── sample_flows.csv            # Sample records for local testing
├── app.py                          # Streamlit demo app
├── requirements.txt
├── README.md
└── .gitignore
```



## Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/lp-0406/Network_Intrusion_Detection_System.git
cd Network_Intrusion_Detection_System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit demo
streamlit run app.py
```

### Run on Kaggle (full training pipeline)

1. Go to the [CICIDS2017 dataset on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
2. Click **+ New Notebook** — dataset auto-mounts at `/kaggle/input/`
3. Upload `notebooks/sentinelnet-final.ipynb` via File → Import Notebook
4. Enable GPU: Settings → Accelerator → **GPU T4 x2**
5. Click **Run All** (~25–40 min on free tier)



## Key Engineering Decisions

**Why XGBoost over Random Forest?**
XGBoost achieved 0.14% higher accuracy and 0.37% higher attack recall in less than half the training time (~23s vs ~45s). For an IDS where every missed attack matters, the recall advantage was the deciding factor.

**Why SMOTE only on the training set?**
Applying SMOTE before the train/test split would cause data leakage — synthetic samples generated from test-set neighbors would artificially inflate evaluation scores. SMOTE is fit and applied strictly after splitting.

**Why a hybrid ensemble?**
The Isolation Forest is trained on benign-only traffic with no labels. This means it can flag anomalous flows that the supervised model has never seen before — zero-day-like attack patterns. Combined with XGBoost via `logical_or`, the ensemble maximizes recall at the cost of acceptable false positive increase.

**Why ANOVA F-test for feature selection?**
Network flow data from CIC-IDS-2017 contains 78 features, many correlated or irrelevant. `SelectKBest` with F-classif efficiently identifies the 30 features most statistically discriminative between Benign and Attack classes, reducing inference latency without accuracy loss.


## Dataset

**CIC-IDS-2017** — Canadian Institute for Cybersecurity

- ~2 million labelled network flow records across 8 CSV files
- 78 features extracted from raw packet captures using CICFlowMeter
- Attack categories: DoS, DDoS, Port Scan, Brute Force, Bot, Web Attacks (SQL Injection, XSS, Infiltration), Infiltration
- Class distribution: ~83% Benign / ~17% Attack (handled with SMOTE)

[Official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) · [Kaggle mirror](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)


## References

- [CIC-IDS-2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [scikit-learn Documentation](https://scikit-learn.org)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org)





*Built by [Chilukuri Laxmi Prashasthi](https://github.com/lp-0406) · CBIT Hyderabad · AI & ML, 2027*
