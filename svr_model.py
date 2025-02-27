import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
import os
import pickle
import matplotlib.pyplot as plt
from utils import load_data, normalize_features, normalize_target, evaluate_model, save_model

def train_svr_model(X, y, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', random_state=None):
    """
    Train a Support Vector Regression model for corrosion prediction
    """
    # Initialize the model
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma
    )
    
    # Train the model
    svr_model.fit(X, y)
    
    return svr_model

def hyperparameter_tuning(X, y, X_val, y_val):
    """
    Perform hyperparameter tuning for SVR model
    """
    # Define hyperparameter grid
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid={'svr__' + key: value for key, value in param_grid.items()},
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Get best model
    best_params = {k.replace('svr__', ''): v for k, v in grid_search.best_params_.items()}
    best_score = grid_search.best_score_
    
    # Train best model on full training data
    best_model = train_svr_model(
        X, y,
        kernel=best_params['kernel'],
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma']
    )
    
    return best_model, best_params, grid_search.cv_results_

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('IACRF/results'):
        os.makedirs('IACRF/results')
    
    # Load data
    try:
        X, y, X_val, y_val = load_data()
    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths.")
        exit(1)
    
    # Scale features (SVR works better with scaled features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_val_scaled = scaler.transform(X_val)
    
    print("Training Support Vector Regression model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model with default parameters
    svr_model = train_svr_model(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(svr_model, X_test, y_test, X_val_scaled, y_val)
    
    print("\nSupport Vector Regression Model Results:")
    print(f"Test R² Score: {results['test_r2']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    
    if 'val_r2' in results:
        print(f"Validation R² Score: {results['val_r2']:.4f}")
        print(f"Validation MAE: {results['val_mae']:.4f}")
        print(f"Validation RMSE: {results['val_rmse']:.4f}")
    
    # Save model
    model_path = save_model(svr_model, 'svr', results['test_r2'], 
                           results.get('val_r2', None))
    print(f"Model saved to: {model_path}")
    
    # Hyperparameter tuning (limited to avoid excessive computation)
    print("\nPerforming hyperparameter tuning (this may take a while)...")
    
    # Use a smaller subset of data for hyperparameter tuning if the dataset is large
    if X_train.shape[0] > 1000:
        X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)
    else:
        X_tune, y_tune = X_train, y_train
    
    # Simplified parameter grid for demonstration
    simplified_param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2],
        'gamma': ['scale', 'auto']
    }
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('svr', SVR())
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid={'svr__' + key: value for key, value in simplified_param_grid.items()},
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_tune, y_tune)
    
    # Get best parameters
    best_params = {k.replace('svr__', ''): v for k, v in grid_search.best_params_.items()}
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train best model on full training data
    best_model = train_svr_model(
        X_train, y_train,
        kernel=best_params['kernel'],
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma']
    )
    
    # Evaluate best model
    best_results = evaluate_model(best_model, X_test, y_test, X_val_scaled, y_val)
    
    print("\nBest SVR Model Results:")
    print(f"Test R² Score: {best_results['test_r2']:.4f}")
    print(f"Test MAE: {best_results['test_mae']:.4f}")
    print(f"Test RMSE: {best_results['test_rmse']:.4f}")
    
    if 'val_r2' in best_results:
        print(f"Validation R² Score: {best_results['val_r2']:.4f}")
        print(f"Validation MAE: {best_results['val_mae']:.4f}")
        print(f"Validation RMSE: {best_results['val_rmse']:.4f}")
    
    # Save best model
    best_model_path = save_model(best_model, 'svr_tuned', best_results['test_r2'], 
                                best_results.get('val_r2', None))
    print(f"Best model saved to: {best_model_path}") 