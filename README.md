<div align="center">

# SentinelNet – Network Intrusion Detection System

**Binary classification IDS** (Benign vs Attack) using **CIC-IDS-2017** dataset  
Modern, production-like ML pipeline with API backend & frontend demo

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-purple?logo=railway)](https://railway.app)
[![Vercel](https://img.shields.io/badge/Frontend%20on-Vercel-black?logo=vercel)](https://vercel.com)

</div>

## ✨ Features

- Trained on **~1.9M cleaned CIC-IDS-2017 flows**
- **99.66% Accuracy** | **99.78% Recall (attacks)** | **very low false negatives** (critical for IDS)
- Feature selection (ANOVA F-test → top 30 features)
- SMOTE balancing + memory optimization
- **FastAPI** inference API (production-ready)
- Simple **Next.js** frontend demo
- Deployed: **Railway** (backend) + **Vercel** (frontend)

## 🚀 Live Demo

- **API** (Swagger docs): https://your-railway-app.railway.app/docs
- **Frontend** (try it!): https://your-vercel-frontend.vercel.app

## 🏗️ Project Structure
sentinelnet-ids/
├── backend/                  # FastAPI inference service
│   ├── main.py
│   ├── requirements.txt
│   └── ...
├── frontend/                 # Next.js / React demo UI
│   ├── app/
│   └── ...
├── notebooks/                # Kaggle development notebook
│   └── sentinelnet-final.ipynb
├── models/                   # Saved artifacts
│   ├── xgboost_model.pkl
│   ├── standard_scaler.pkl
│   └── selected_features.json
├── data/                     # sample_flows.csv (5-10 rows only!)
├── README.md
└── .gitignore


## 📊 Results Summary

| Model          | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | Train Time |
|----------------|----------|---------------------|------------------|-------------|------------|
| XGBoost (best) | 99.66%   | 97.49%             | **99.78%**      | 98.62%      | ~23s      |
| RandomForest   | 99.52%   | 96.81%             | 99.41%          | 98.10%      | ~45s      |
| Isolation Forest (unsupervised) | - | - | 35.3% | - | ~14s |

## 🛠️ Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
cd frontend
npm install
npm run dev
