***

# PrometheusML 

***

## **MLOps System: Train -> Serve -> Monitor -> Retrain**

---

PrometheusML is a complete machine learning lifecycle constructed to demonstrate authentic MLOps practices. 

It trains a payment risk model, serves it via and API, logs live inferences, monitors drift and performance, retrains models automatically when warranted, and reloads the deployed model. This is all accomplished locally, with repoducibility.

The primary focus is on system correctness and lifecycle orchestration, with less attention to state-of-the-art model accuracy, and serves as an exploration of what happens after deployment emulating how ML operates in practice.

---

## What PrometheusML Demonstrates

- Reproducible model training with experiment tracking
- Versioned model registry and promotion logic
- Online inference via FastAPI
- Feature and prediction logging
- Delayed labels and feedback handling
- Drift detection (feature and prediction)
- Automated retraining decisions
- Safe promotion rules (no model churn)
- Hot reload of deployed models
- One-command end-to-end demo

--- 

# System Architecture

```text
              ┌──────────────┐
              │ Raw / Drifted│
              │  Payment Data│
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Training     │
              │ Pipeline     │
              │ (sklearn)    │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ MLflow       │
              │ Model        │
              │ Registry     │
              └──────┬───────┘
                     │
        ┌────────────▼────────────┐
        │ FastAPI Inference API   │
        │ /predict /feedback     │
        └────────────┬────────────┘
                     │
              ┌──────▼───────┐
              │ DuckDB       │
              │ Inference +  │
              │ Feedback Log │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Monitoring   │
              │ Drift + Perf │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Retraining   │
              │ + Promotion  │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ API Reload   │
              │ (Token Auth) │
              └──────────────┘
```

---

## Problem Setting

**Binary classification: payment chargeback risk**

- Input: tabular payment features
- Output: probability of chargeback
- Labels arrive after inference (simulated delayed feedback)
- Class imbalance is intentional

**Note:** Data is synthetic, PrometheusML exists to validate MLOps mechanics only

---

## Tech Stack

- Python 3.11
- scikit-learn
- FastAPI + Uvicorn
- MLflow (SQLite backend)
- DuckDB (inference and feedback logs)
- Requests
- Pandas / NumPy

---

## Repository Structure

```text

Prometheus/
├── api/
│   └── main.py                # FastAPI service
├── scripts/
│   ├── train.py               # Initial training
│   ├── monitor.py             # Drift & performance monitoring
│   ├── retrain.py             # Retrain + promotion logic
│   ├── ops_tick.py            # Monitor → retrain → reload orchestration
│   ├── simulate_drift_traffic.py
│   └── demo_full_loop.py      # One-command demo
├── src/prometheus/
│   ├── data/                  # Synthetic data generation
│   ├── training/              # Pipelines & features
│   ├── monitoring/            # Drift logic
│   └── utils/                 # Config, helpers
├── data/
│   └── prometheus.duckdb      # Local inference store
├── reports/
│   └── drift_*.md             # Monitoring reports
├── .env
└── README.md


```

---

## Setup

### Create Environment

```bash
conda create -n prometheus python=3.11
conda activate prometheus
pip install -r requirements.txt
```

### Configure Environment

Create `.env` in repo root:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
RELOAD_TOKEN=change_me_local_dev
```

---

## Quickstart

### Generate Data

```bash
python scripts/make_payment_data.py
```

### Train and Register Initial Model

```bash
python scripts/train.py
```

### Start API

```bash
uvicorn api.main:app --reload
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "amount_gbp": 42.5, "hour": 2, "is_weekend": 1,
        "merchant_category": 5, "payment_method": 0,
        "account_age_days": 12, "txn_count_24h": 4,
        "ip_country_mismatch": 1, "device_new": 1,
        "device_risk": 0.72 }'
```

---

## Full Demo (Single Command)

```bash
python scripts/demo_full_loop.py --n 1200
```

**What happens:**

- Drifted traffic with labels is generated
- Monitoring detects drift and writes a report
- Retraining runs automatically
- Promotion logic decides whether to upgrade the model
- API hot-reloads the champion model securely
- /model confirms the active version

**Example output:**

```bash
Retrain triggered status: True
PROMOTED_FLAG=0
Reloaded API: model_version=13
```

---

## Monitoring

**Monitoring produces readable reports in `reports/`:**

- Feature drift (PSI)
- Prediction drift
- Performance on labeled feedback
- Retraining decision

**Example:**

```bash
reports/drift_20260127_2114.md
```

--- 

## Retraining and Promotion Rules

**Retrain triggers depend on:**

- feature drift exceeds a threshold
- prediction distribution shifts
- sufficient new labeled data exists

**Promotion only occurs if:**

- candidate model beats champion model by margin
- otherwise, champion model remains

---

## Security Note

`/reload_model` is prodected via header:

```bash
X-Reload-Token
```

- Token is read from `.env`

---

