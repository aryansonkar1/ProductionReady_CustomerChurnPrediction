import pandas as pd
import yaml
import os

def test_data_exists():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert os.path.exists(config["data"]["raw_path"]), "Dataset file not found!"

def test_data_schema():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config["data"]["raw_path"])
    
    # Check for some essential required columns
    required_columns = ['tenure', 'MonthlyCharges', 'Contract', 'Churn']
    for col in required_columns:
        assert col in df.columns, f"Missing essential column: {col}"

    assert not df.empty, "Dataset is empty"

def test_data_datatypes():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
        
    df = pd.read_csv(config["data"]["raw_path"])
    assert pd.api.types.is_numeric_dtype(df['tenure']), "tenure must be numeric"
    assert pd.api.types.is_numeric_dtype(df['MonthlyCharges']), "MonthlyCharges must be numeric"
