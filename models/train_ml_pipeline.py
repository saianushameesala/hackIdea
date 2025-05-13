"""
Train and save ML models for the explainability app
"""
import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def save_model_and_data(model, X, y, model_path, data_path):
    """Save model and dataset"""
    # Save model
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)
    
    # Create and save dataset
    data = X.copy()
    data['y'] = y
    data.to_csv(data_path, index=False)
    
    print(f"Saved model to {model_path}")
    print(f"Saved data to {data_path}")

# 1. Breast Cancer Dataset (Classification)
print("Training breast cancer models...")

# Load data
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

# Train logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

logistic_pipeline.fit(X_train, y_train)
accuracy = logistic_pipeline.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# Save logistic model
save_model_and_data(
    logistic_pipeline, 
    X_cancer, 
    y_cancer, 
    "models/logistic_model.pkl", 
    "data/breast_cancer.csv"
)

# Train random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Save RF model
save_model_and_data(
    rf_model, 
    X_cancer, 
    y_cancer, 
    "models/rf_model.pkl", 
    "data/breast_cancer_rf.csv"
)

# 2. Diabetes Dataset (Regression)
print("\nTraining diabetes models...")

# Load data
diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diabetes = diabetes.target

# Train linear regression model
X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

linear_pipeline.fit(X_train, y_train)
r2 = linear_pipeline.score(X_test, y_test)
print(f"Linear Regression R² Score: {r2:.4f}")

# Save linear model
save_model_and_data(
    linear_pipeline, 
    X_diabetes, 
    y_diabetes, 
    "models/linear_reg_model.pkl", 
    "data/diabetes.csv"
)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
r2 = xgb_model.score(X_test, y_test)
print(f"XGBoost R² Score: {r2:.4f}")

# Save XGBoost model
save_model_and_data(
    xgb_model, 
    X_diabetes, 
    y_diabetes, 
    "models/xgb_model.pkl", 
    "data/diabetes_xgb.csv"
)

print("\nAll models and datasets have been created!")
