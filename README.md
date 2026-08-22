# Production-Ready Customer Churn Prediction System

This repository demonstrates a complete, production-style Machine Learning pipeline for predicting customer churn. It upgrades a traditional Jupyter Notebook experimentation environment into a scalable, deployable system with rigorous software engineering practices.

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
                     MLflow
                         ↓
                Model Evaluation
                         ↓
                 Model Registry
                         ↓
                    FastAPI
                         ↓
                     Docker
                         ↓
                  CI/CD Pipeline
                         ↓
                     Cloud
                         ↓
                   Monitoring
                         ↓
                  Drift Detection
```

## ✨ Engineering Features

*   **Modular Pipeline:** Data ingestion, preprocessing, and model training logic decoupled into `src/` modules.
*   **Encapsulated Data Transformations:** Scikit-Learn `ColumnTransformer` handles scaling and encoding dynamically.
*   **Experiment Tracking:** Integrating **MLflow** for robust parameter, threshold, and artifact tracking (`src/train.py`).
*   **Automated Model Selection:** Evaluates LogReg, Random Forest, XGBoost, and an Ensemble Voting logic dynamically logging the best performer.
*   **Web API (Serving):** Lightweight local deployment via **FastAPI** (`api/main.py`) strictly typing schemas with **Pydantic**.
*   **Containerization:** Extensible `Dockerfile` ready for scale and cloud deployment execution.
*   **Testing:** End-to-end `pytest` coverage verifying ML behavior, Data validation, and Endpoint correctness (`tests/`).
*   **CI/CD:** Integrated **GitHub Actions** checking environments on push and validating the Docker Build (`.github/workflows/ci.yml`).

## 🚀 Quick Start (Local Setup)

1.  **Install requirements and initialize the environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Run the Training Pipeline (Generates Model & MLflow outputs)**:
    ```bash
    python src/train.py
    ```

3.  **Spin up the API locally**:
    ```bash
    uvicorn api.main:app --reload
    ```

4.  **Test the inference endpoint**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'Content-Type: application/json' \
      -d '{
        "tenure": 12,
        "MonthlyCharges": 85.5,
        "Contract": "Month-to-month"
      }'
    ```

## 🐋 Docker Build (Deployment Ready!)

To quickly serve the isolated network:
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## 📊 Next Steps (Roadmap)
- [ ] Connect FastAPI tracking directly to **Prometheus** for endpoint SLA tracking (Latency, Requests/Sec).
- [ ] Incorporate **Evidently** evaluating data drift over live inference volumes (Input distributions vs Data distributions).
- [ ] Push to container registries (AWS ECR / DockerHub).
