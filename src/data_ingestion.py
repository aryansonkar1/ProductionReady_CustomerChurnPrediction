import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    df = df.drop(columns='customerID', errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def get_data_splits(data_path, test_size=0.2, random_state=42):
    df = load_data(data_path)
    df = clean_data(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
