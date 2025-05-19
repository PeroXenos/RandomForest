import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Data Preprocessing and Analysis
def load_and_preprocess_data():
    # Load breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values detected, filling with median")
        df.fillna(df.median(), inplace=True)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for deployment
    joblib.dump(scaler, 'scaler.joblib')
    
    return X_scaled, y, data.feature_names

# 2. Model Training and Evaluation
def train_and_evaluate_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {

        'RandomForest': RandomForestClassifier(random_state=42)
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std()
        }
    
    # Select best model (RandomForest based on typically good performance)
    best_model = RandomForestClassifier(random_state=42)
    best_model.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(best_model, 'best_model.joblib')
    
    return results, X_test, y_test

# 3. FastAPI Web Service
app = FastAPI(title="Binary Classification API")

# Define input data model
class PredictionInput(BaseModel):
    features: list[float]

# Load model and scaler only after ensuring they exist
def load_resources():
    if not os.path.exists('scaler.joblib') or not os.path.exists('best_model.joblib'):
        logger.info("Model or scaler not found, training model...")
        X, y, feature_names = load_and_preprocess_data()
        results, X_test, y_test = train_and_evaluate_models(X, y)
        joblib.dump(feature_names, 'feature_names.joblib')
        # Print model performance
        print("\nModel Performance Metrics:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
    return joblib.load('scaler.joblib'), joblib.load('best_model.joblib')

scaler, model = load_resources()

@app.get("/feature_names")
async def get_feature_names():
    try:
        feature_names = joblib.load('feature_names.joblib')
        return list(feature_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Validate input dimensions
        if features.shape[1] != scaler.n_features_in_:
            raise HTTPException(status_code=400, 
                              detail=f"Expected {scaler.n_features_in_} features, got {features.shape[1]}")
        
        # Preprocess input
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "class": "Malignant" if prediction == 1 else "Benign"
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution
if __name__ == "__main__":
    
    def add_feature_names_endpoint(app: FastAPI):
        @app.get("/feature_names")
        async def get_feature_names():
            try:
                feature_names = joblib.load('feature_names.joblib')
                return list(feature_names)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
