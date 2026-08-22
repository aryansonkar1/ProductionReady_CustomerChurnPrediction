import joblib
import pandas as pd
import yaml
import pytest

@pytest.fixture
def model():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    try:
        return joblib.load(config["model"]["save_path"])
    except:
        return None

def test_model_loads(model):
    assert model is not None, "Model failed to load"

def test_model_prediction(model):
    if model is None:
        pytest.skip("Model not available")
        
    # Dummy data ensuring same columns as training
    columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    data = [['Male', 0, 'No', 'No', 12, 'Yes', 'No', 'DSL', 'No', 'No', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes', 'Electronic check', 85.5, 1026.0]]
    df = pd.DataFrame(data, columns=columns)
    
    prob = model.predict_proba(df)[:,1]
    
    assert len(prob) == 1
    assert 0 <= prob[0] <= 1
