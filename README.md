# Production-Ready Customer Churn Prediction System

### 🌐 [Live Demo → churn-predictor-0w5i.onrender.com](https://churn-predictor-0w5i.onrender.com)

This repository demonstrates a complete, production-style Machine Learning pipeline for predicting customer churn. It upgrades a traditional Jupyter Notebook experimentation environment into a scalable, deployable system with rigorous software engineering practices, complete with an interactive Glassmorphism UI.

## 📌 Architecture

```text
                 ┌───────────────┐
                 │ Raw Customer  │
                 │     Data      │
                 └───────┬───────┘
                         ↓
                 Data Validation
                         ↓
               Feature Engineering
                         ↓
                Model Training
                         ↓
                     MLflow (Model Registry)
                         ↓
                    FastAPI (Inference Server)
                     │     │
              Frontend     Prometheus (Telemetry)
                     │     │
                     Docker (Containerization)
                         ↓
                  CI/CD Pipeline (GitHub Actions)
                         ↓
                      Cloud (Render)
```

## ✨ Engineering Features

*   **Modular Pipeline:** Data ingestion, preprocessing, and model training logic decoupled into strict software `src/` modules.
*   **Encapsulated Data Transformations:** Prevented Data Leakage during SMOTE by utilizing `imblearn.pipeline.Pipeline` with `ColumnTransformer` (StandardScaler + OneHotEncoder).
*   **Experiment Tracking:** Integrates **MLflow** for robust parameter tracking, hyperparameter tuning, and threshold evaluation (`src/train.py`).
*   **Automated Model Selection:** Dynamically evaluated Logistic Regression, Random Forest, KNN, XGBoost, and Stacking/Voting Classifiers. The final pipeline logs the best performer using threshold optimization.
*   **Web API (Serving):** Lightweight local deployment via **FastAPI** (`api/main.py`), strictly typing incoming JSON payload schemas with **Pydantic**.
*   **Interactive UI:** A highly aesthetic, Glassmorphism-themed frontend (`frontend/`) served natively through FastAPI via `StaticFiles`.
*   **Monitoring:** Live SLA tracking and endpoint metrics ingestion integrated via **Prometheus** (`prometheus-fastapi-instrumentator`).
*   **Containerization:** Extensible `Dockerfile` packaging the ML Model, the Backend API, and the Frontend UI completely encapsulated for robust deployment.
*   **Testing:** End-to-end `pytest` coverage verifying ML behavior bounded between 0-1 probabilities, strict DataFrame validation, and FastAPI Endpoint correctness (`tests/`).
*   **CI/CD:** Integrated **GitHub Actions** checking strict environments on push and validating the Docker Build (`.github/workflows/ci.yml`).

## 🧠 Machine Learning Insights

Based on extensive Exploratory Data Analysis, traditional baseline evaluation probability thresholds (0.50 cutoff) resulted in high False Positive churn warnings. 

By analyzing the Precision-Recall dependencies within our Voting Classifier (Random Forest + Gradient Boost models), **the pipeline dynamically evaluates and applies a custom target threshold (optimum `0.55`)**. This mathematically improves business Precision out to 61% (saving retention budgets) while maintaining a dominant 70% Recall on customers at genuine risk.

## 🚀 Local Setup & Execution

### 1. Initialize Virtual Environment
```bash
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\Activate.ps1
# Activate on Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate the Pipeline (Training)
This script simulates the MLOps pipeline, validating data, processing splits, testing multiple classifiers, tracking metrics via MLflow, and yielding the serialized `.pkl` artifact.
```bash
python src/train.py
```

### 3. Spin up the Production Server
```bash
uvicorn api.main:app --reload --port 8000
```
- **Interactive UI Dashboard:** Open your browser to `http://127.0.0.1:8000/`
- **Swagger API Docs:** `http://127.0.0.1:8000/docs`
- **Live Prometheus Telemetry:** `http://127.0.0.1:8000/metrics`

## 🐋 Docker Build (Deployment Ready!)

To quickly serve the isolated network:
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## 🌐 Live Deployment

The app is deployed on **Render** and accessible at:

🔗 **https://churn-predictor-0w5i.onrender.com**

The repository is configured to deploy instantly via Render, natively connecting to GitHub pushes. Every `git push` to `main` triggers an automatic redeployment.
