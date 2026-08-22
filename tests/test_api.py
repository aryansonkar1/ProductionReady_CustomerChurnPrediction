from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "1.0"}

def test_predict_endpoint_success():
    payload = {
        "tenure": 12,
        "monthly_charges": 85.5,
        "contract": "Month-to-month"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert "threshold" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1

def test_predict_endpoint_validation_error():
    payload = {
        "tenure": "invalid_type",
        "monthly_charges": 85.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
