<div align="center" style="border: 2px solid #0d1b3e; padding: 20px; border-radius: 12px; width: 80%; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.15);">
    <img
        width="180"
        height="220"
        alt="Logo - SURE ProEd"
        src="https://github.com/user-attachments/assets/88fa5098-24b1-4ece-87df-95eb920ea721"
        style="border-radius: 10px;"
    />

  <h1 align="center" style="font-family: Arial; font-weight: 600; margin-top: 15px;">SURE ProEd (formerly SURE Trust)</h1>
  <h2 style="color: #2b6cb0; font-family: Arial;">Skill Upgradation for Rural Youth Empowerment Trust</h2>
</div>


<div style="padding: 20px; border: 2px solid #ddd; border-radius: 12px; width: 90%; margin: auto; background: #fafafa; font-family: Arial;">

<h2 style="color:#333;">Student Details</h2>
<div align="left" style="margin: 20px; font-size: 16px;">
    <p><strong>Name:</strong> Chilukuri Laxmi Prashasthi </p>
    <p><strong>Email ID:</strong> laxmiprashasthi@gmail.com </p>
    <p><strong>College Name:</strong> Chaitanya Bharathi Institute of Technology </p>
    <p><strong>Branch/Specialization:</strong> Artificial Intelligence and Machine Learning </p>
    <p><strong>College ID:</strong>160123729303 </p>
</div>



<h2 style="color:#333;">Course Details</h2>
<div align="left" style="margin: 20px; font-size: 16px;">
    <p><strong>Course Opted:</strong>G35 Python</p>
    <p><strong>Instructor Name:</strong> Gaurav Patel  </p>
    <p><strong>Duration:</strong> 6 months (July to January) </p>
</div>


<h2 style="color:#333;">Trainer Details</h2>
<div align="left" style="margin: 20px; font-size: 16px;">
    <p><strong>Trainer Name:</strong> Gaurav Patel </p>
    <p><strong>Trainer Email ID:</strong> gaurav.patel.gpp@gmail.com</p>
    <p><strong>Trainer Designation:</strong>  Data engineer </p>
</div>

</div>


## Table of Contents

- [Overall Learning](#overall-learning)
- [Projects Completed](#projects-completed)
- [Project Introduction](#project-introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Results Summary](#results-summary)
- [Roles and Responsibilities](#roles-and-responsibilities)
- [Local Development](#local-development)
- [Live Demo](#live-demo)
- [References](#references)
- [Learnings from LST & SST](#learnings-from-lst--sst)
- [Community Services](#community-services)
- [Certificate](#certificate)
- [Acknowledgments](#acknowledgments)


## Overall Learning

During this course, I worked on building an end to end AI powered Network Intrusion Detection System using real world cybersecurity data. I gained hands on experience with the full machine learning pipeline from raw data ingestion and cleaning to model training, hyperparameter tuning, and deployment. I strengthened my understanding of supervised and unsupervised learning, class imbalance handling with SMOTE, feature selection using ANOVA F-tests, and ML deployment using streamlit. I also developed skills in writing clean, memory efficient Python code for large datasets.



## Projects Completed

**[Project 1: SentinelNet — AI-Powered Network Intrusion Detection System](#project-introduction)**


## Project Introduction

<div align="center">

# SentinelNet — Network Intrusion Detection System

**Binary classification IDS** (Benign vs Attack) using the **CIC-IDS-2017** dataset
Modern, production-like ML pipeline with API backend & frontend demo

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-purple?logo=railway)](https://railway.app)
[![Vercel](https://img.shields.io/badge/Frontend%20on-Vercel-black?logo=vercel)](https://vercel.com)
[![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle)](https://kaggle.com)

</div>

SentinelNet is an end-to-end AI-powered Network Intrusion Detection System (NIDS) built to classify real-world network traffic as **Benign** or **Attack** in real time. It was developed entirely on **Kaggle** using the publicly available **CIC-IDS-2017** dataset from the Canadian Institute for Cybersecurity, which contains ~2 million labelled network flow records across 8 attack categories including DoS, DDoS, Port Scan, Brute Force, Bot, Web Attacks, and Infiltration.

The project covers the complete ML pipeline — data ingestion, memory optimization, cleaning, feature selection, SMOTE balancing, training of multiple classifiers, hyperparameter tuning, unsupervised anomaly detection, and real-time alert generation. The best model (XGBoost) was then exported and used to deploy on streamlit.

### ✨ Key Features

- Trained on **~1.9M cleaned CIC-IDS-2017 flows** across 8 CSV files
- **99.66% Accuracy** | **99.78% Recall (attacks)** | very low false negatives — critical for IDS
- Memory-optimized loading: `float64→float32`, `int64→int32`, `MAX_ROWS` cap, `gc.collect()` after each file
- Feature selection via **ANOVA F-test → top 30 features** using `SelectKBest`
- **SMOTE** oversampling applied on training set only to fix 83%/17% class imbalance
- **Isolation Forest** for unsupervised anomaly detection (trained on benign-only traffic)
- **Hybrid ensemble**: XGBoost + Isolation Forest combined via `logical_or` for maximum recall
- **FastAPI** inference API with `/predict` endpoint and Swagger docs (production-ready)
- Simple **Next.js** frontend demo for manual flow input and live prediction
- Deployed: **Streamlit**
- Real-time **alert logging** to `alerts.log` with timestamp, probability, and flow details

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **ML / Data** | scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Feature Selection** | `SelectKBest` (ANOVA F-classif) |
| **Anomaly Detection** | `IsolationForest` |
| **Hyperparameter Tuning** | `RandomizedSearchCV` |
| **Artifact Storage** | joblib, JSON |
| **Deployment** | Streamlit |
| **Platform** | Kaggle Notebooks (T4 GPU, 30GB RAM) |
| **Dataset** | CIC-IDS-2017 (Canadian Institute for Cybersecurity) |


## Project Structure

```
sentinelnet-ids/
├── notebooks/                     
│   └── sentinelnet-final.ipynb     
├── models/                        
│   ├── xgboost_model.pkl          
│   ├── standard_scaler.pkl         
│   ├── selected_features.json     
│   └── isolation_forest.pkl        
├── data/
│   └── sample_flows.csv           
├── README.md
└── .gitignore
```

## Results Summary

### Supervised Models

| Model | Accuracy | Precision (Attack) | Recall (Attack) | F1 (Attack) | Train Time |
|---|---|---|---|---|---|
| **XGBoost (best)**  | **99.66%** | 97.49% | **99.78%** | **98.62%** | ~23s |
| Random Forest | 99.52% | 96.81% | 99.41% | 98.10% | ~45s |
| Decision Tree | 99.12% | 95.60% | 98.90% | 97.22% | ~12s |
| Logistic Regression | 97.10% | 93.20% | 96.80% | 94.97% | ~8s |
| LinearSVC | 96.80% | 92.40% | 96.10% | 94.21% | ~15s |

### Unsupervised (Isolation Forest)

| Model | Recall (Attack) | Train Time | Notes |
|---|---|---|---|
| Isolation Forest | 35.3% | ~14s | Trained on benign-only; no labels used |

> **Why Recall matters most for IDS:** A missed attack (false negative) is far more costly than a false alarm. XGBoost achieves **99.78% attack recall**, meaning it catches nearly all intrusions with very few misses.

### Hyperparameter Tuning (XGBoost)

Tuning was performed using `RandomizedSearchCV` with 25 iterations, 3-fold CV, F1 scoring, on 40% of the SMOTE-balanced training set for speed.

**Best parameters found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.8 |
| Best CV F1 | **0.9994** |



## Roles and Responsibilities

As the sole developer of this project, I was responsible for the complete end-to-end pipeline:

- **Data Engineering** — Loaded and concatenated 8 CICIDS2017 CSV files on Kaggle; implemented `reduce_mem_usage()` for dtype downcasting; applied `MAX_ROWS` cap and `gc.collect()` for memory safety
- **Data Cleaning** — Handled infinite values, NaN imputation (median for rate columns), duplicate removal, label encoding
- **Feature Engineering** — Applied `SelectKBest` (ANOVA F-test, k=30); built new ratio features (`Fwd_Bwd_Ratio`, `Bytes_Ratio`, `Avg_Fwd_IAT`)
- **Imbalance Handling** — Analyzed 83%/17% class split; applied SMOTE on training data only; used `class_weight='balanced'` across all sklearn models
- **Model Training** — Trained 5 supervised classifiers (DT, RF, LinearSVC, LR, XGBoost) and evaluated with classification reports, confusion matrices, ROC-AUC
- **Anomaly Detection** — Implemented Isolation Forest on benign-only traffic; built hybrid XGBoost + IsoForest ensemble using `logical_or`
- **Hyperparameter Tuning** — Ran `RandomizedSearchCV` (25 iter, 3-fold, F1) on XGBoost with 8-parameter grid
- **Alert System** — Built `generate_alert()` function for real-time intrusion logging with timestamps and probability scores
- **Artifact Management** — Saved all model artifacts (`best_xgboost_tuned.pkl`, `standard_scaler.pkl`, `selected_features.json`, `isolation_forest.pkl`)
- **Backend Deployment** — Built FastAPI `/predict` endpoint; deployed on Railway
- **Frontend Deployment** — Built Next.js demo UI; deployed on Vercel


## Local Development
- Clone Repository
    bash
    ```
    git clone https://github.com/lp-0406/Network_Intrusion_Detection_System.git
    cd Network_Intrusion_Detection_System
    ```
- create virtual environment
  bash
  ```
  python -m venv .venv
  .venv\Scripts\activate
  ```
- install dependencies
        bash
        ```
        pip install -r requirements.txt
        ```      
- run application
  bash
  ```
  streamlit run app.py
  ```

### Kaggle Notebook

The full training pipeline is in `notebooks/sentinelnet-final.ipynb`.

To run on Kaggle:
1. Go to [kaggle.com/datasets/chethuhn/network-intrusion-dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
2. Click **+ New Notebook** — dataset auto-mounts at `/kaggle/input/`
3. Upload `sentinelnet-final.ipynb` via File → Import Notebook
4. Enable GPU: Settings → Accelerator → **GPU T4 x2**
5. Click **Run All** (~25–40 min on free tier)

## References

- [CIC-IDS-2017 Dataset — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [scikit-learn Documentation](https://scikit-learn.org)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org)
- [CICIDS2017 on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)



## Learnings from LST & SST

The LST and SST sessions helped me understand how to approach real-world ML projects end-to-end rather than just running models in isolation. I learned the importance of data quality over model complexity — most of my performance gains came from proper cleaning and SMOTE balancing, not from switching algorithms. The sessions also gave me exposure to deployment workflows, API design, and how to communicate technical results clearly to non-technical audiences.



## Community Services

During my internship period, I participated in multiple community-oriented activities that helped develop a sense of social responsibility alongside my technical skills.

### Activities Involved

- **Blood Donation** — Donated blood and supported basic assistance tasks during the camp.
- **Tree Plantation Drive** — Participated by planting trees and contributing to environmental improvement.
- **Helping Elder Citizens** — Assisted elderly individuals with simple daily tasks and provided support where needed.

### Impact / Contribution

- Helped create a supportive environment during the blood donation camp.
- Actively participated in promoting greener and cleaner surroundings.
- Offered personal assistance to elder citizens, strengthening community bonds.
- Improved skills in communication, coordination, and social responsibility.


## Certificate

The internship certificate serves as an official acknowledgment of the successful completion of my training period at SURE ProEd. It is issued upon fulfilling all required milestones — including project completion, documentation, community service, and final presentation. The certificate validates the skills, experience, and real-world contributions made during the internship, including the development of SentinelNet.


## Acknowledgments

- [Prof. Radhakumari Challa](https://www.linkedin.com/in/prof-radhakumari-challa-a3850219b), Executive Director and Founder — [SURE Trust](https://www.suretrustforruralyouth.com/)
- <!--- Add your Trainer Name and LinkedIn/profile link --->
- The Canadian Institute for Cybersecurity for making the CIC-IDS-2017 dataset publicly available
- The open-source communities behind scikit-learn, XGBoost, FastAPI, and imbalanced-learn
