import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import time
import pickle
from utils import load_data, normalize_features, normalize_target, evaluate_model, save_model

def train_xgboost_model(X, y, n_estimators=100, max_depth=3, learning_rate=0.1, 
                        subsample=1.0, colsample_bytree=1.0, random_state=None):
    """
    Train an XGBoost model for corrosion prediction
    """
    # Initialize the model
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train the model
    xgb_model.fit(X, y)
    
    return xgb_model

def hyperparameter_tuning(X, y, X_val, y_val):
    """
    Perform hyperparameter tuning for XGBoost model
    """
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Get best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Train best model on full training data
    best_model = train_xgboost_model(
        X, y,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree']
    )
    
    return best_model, best_params, grid_search.cv_results_

def plot_feature_importance(model, feature_names=None):
    """
    Plot feature importance from the trained model
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort feature importance
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - XGBoost")
    plt.bar(range(len(importance)), importance[indices], align="center")
    
    if feature_names is not None:
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    else:
        plt.xticks(range(len(importance)), indices)
    
    plt.tight_layout()
    plt.savefig('IACRF/results/xgb_feature_importance.png')
    plt.close()

def plot_learning_curve(model, X, y, cv=5):
    """
    Plot learning curve to evaluate model performance with different training set sizes
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='r2', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve - XGBoost")
    plt.xlabel("Training examples")
    plt.ylabel("R² Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.savefig('IACRF/results/xgb_learning_curve.png')
    plt.close()

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
    
    # Use original data without normalization
    X_norm = X
    y_norm = y
    X_val_norm = X_val
    y_val_norm = y_val
    
    print("Training XGBoost model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    
    # Train model with default parameters
    xgb_model = train_xgboost_model(X_train, y_train, random_state=42)
    
    # Evaluate model
    results = evaluate_model(xgb_model, X_test, y_test, X_val_norm, y_val_norm)
    
    print("\nXGBoost Model Results:")
    print(f"Test R² Score: {results['test_r2']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    
    if 'val_r2' in results:
        print(f"Validation R² Score: {results['val_r2']:.4f}")
        print(f"Validation MAE: {results['val_mae']:.4f}")
        print(f"Validation RMSE: {results['val_rmse']:.4f}")
    
    # Save model
    model_path = save_model(xgb_model, 'xgboost', results['test_r2'], 
                           results.get('val_r2', None))
    print(f"Model saved to: {model_path}")
    
    # Plot feature importance
    try:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        plot_feature_importance(xgb_model, feature_names)
        print("Feature importance plot saved to: IACRF/results/xgb_feature_importance.png")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
    
    # Plot learning curve
    try:
        plot_learning_curve(xgb_model, X_train, y_train)
        print("Learning curve plot saved to: IACRF/results/xgb_learning_curve.png")
    except Exception as e:
        print(f"Error plotting learning curve: {e}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning (this may take a while)...")
    
    # Use a smaller subset of data for hyperparameter tuning if the dataset is large
    if X_train.shape[0] > 1000:
        X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)
    else:
        X_tune, y_tune = X_train, y_train
    
    # Simplified parameter grid for demonstration
    simplified_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=simplified_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_tune, y_tune)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train best model on full training data
    best_model = train_xgboost_model(
        X_train, y_train,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )
    
    # Evaluate best model
    best_results = evaluate_model(best_model, X_test, y_test, X_val_norm, y_val_norm)
    
    print("\nBest XGBoost Model Results:")
    print(f"Test R² Score: {best_results['test_r2']:.4f}")
    print(f"Test MAE: {best_results['test_mae']:.4f}")
    print(f"Test RMSE: {best_results['test_rmse']:.4f}")
    
    if 'val_r2' in best_results:
        print(f"Validation R² Score: {best_results['val_r2']:.4f}")
        print(f"Validation MAE: {best_results['val_mae']:.4f}")
        print(f"Validation RMSE: {best_results['val_rmse']:.4f}")
    
    # Save best model
    best_model_path = save_model(best_model, 'xgboost_tuned', best_results['test_r2'], 
                                best_results.get('val_r2', None))
    print(f"Best model saved to: {best_model_path}")
    
    # Plot feature importance for best model
    try:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        plot_feature_importance(best_model, feature_names)
        print("Feature importance plot for best model saved to: IACRF/results/xgb_feature_importance.png")
    except Exception as e:
        print(f"Error plotting feature importance for best model: {e}") 