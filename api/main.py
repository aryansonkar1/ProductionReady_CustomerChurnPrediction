from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import yaml

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["model"]["save_path"]
threshold = config["evaluation"]["threshold"]

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Warning: Model could not be loaded. Error: {e}")


class CustomerRequest(BaseModel):
    tenure: int
    monthly_charges: float = Field(alias="MonthlyCharges")
    contract: str = Field(alias="Contract")
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    TotalCharges: str = "0.0"

    class Config:
        populate_by_name = True


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy", "model_version": "1.0"}


@app.post("/predict")
def predict_churn(customer: CustomerRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    data_dict = customer.model_dump(by_alias=True)
    df = pd.DataFrame([data_dict])

    try:
        prob = model.predict_proba(df)[0][1]
        prediction = int(prob >= threshold)
        return {
            "prediction": prediction,
            "churn_probability": round(float(prob), 4),
            "threshold": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


Instrumentator().instrument(app).expose(app)
