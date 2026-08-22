import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sys

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_ingestion import get_data_splits
from src.preprocessing import get_preprocessor

def train_model():
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data_path = config["data"]["raw_path"]
    X_train, X_test, y_train, y_test = get_data_splits(
        data_path, 
        test_size=config["data"]["test_size"], 
        random_state=config["base"]["random_state"]
    )
    
    preprocessor = get_preprocessor(X_train)
    
    # Define models
    models_to_train = {
        "LogisticRegression": LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42),
    }
    
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    best_f1 = 0
    best_pipeline = None
    best_model_name = ""
    
    threshold = config["evaluation"]["threshold"]
    primary_metric = config["evaluation"]["primary_metric"]
    
    trained_estimators = []
    
    # Train individual models
    for name, model in models_to_train.items():
        with mlflow.start_run(run_name=name):
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            # Store the trained model for the voting classifier
            trained_estimators.append((name, pipeline.named_steps['model']))
            
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            
            metrics = {
                "roc_auc": roc_auc_score(y_test, y_prob),
                "f1_score": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }
            
            mlflow.log_params({"model_type": name, "threshold": threshold})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
            
            if metrics[primary_metric] > best_f1:
                best_f1 = metrics[primary_metric]
                best_pipeline = pipeline
                best_model_name = name

    # Train Voting Classifier
    with mlflow.start_run(run_name="VotingClassifier"):
        voting_clf = VotingClassifier(
            estimators=trained_estimators,
            voting='soft'
        )
        
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', voting_clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
        
        mlflow.log_params({"model_type": "VotingClassifier", "threshold": threshold})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
        
        if metrics[primary_metric] > best_f1:
            best_f1 = metrics[primary_metric]
            best_pipeline = pipeline
            best_model_name = "VotingClassifier"
            
    print(f"Best model based on {primary_metric}: {best_model_name} with score: {best_f1:.4f}")
    
    # Save the best pipeline
    os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)
    joblib.dump(best_pipeline, config["model"]["save_path"])
    print(f"Best pipeline saved to {config['model']['save_path']}")

if __name__ == "__main__":
    train_model()
