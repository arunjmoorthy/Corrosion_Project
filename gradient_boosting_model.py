import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import time
import os
import pickle
import matplotlib.pyplot as plt
from utils import load_data, normalize_features, normalize_target, evaluate_model, save_model

def train_gradient_boosting_model(X, y, X_val=None, y_val=None, n_estimators=100, learning_rate=0.1, 
                                 max_depth=3, min_samples_split=2, random_state=None):
    """
    Train a Gradient Boosting model for corrosion prediction
    """
    # Initialize the model
    gb_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    
    # Train the model
    gb_model.fit(X, y)
    
    return gb_model

def hyperparameter_tuning(X, y, X_val, y_val):
    """
    Perform hyperparameter tuning for Gradient Boosting model
    """
    # Define hyperparameter grid
    n_estimators_list = [50, 100, 200]
    learning_rates = [0.05, 0.1, 0.2]
    max_depths = [3, 5, 7]
    
    best_score = -np.inf
    best_params = {}
    best_model = None
    
    results = []
    
    # Grid search
    for n_est in n_estimators_list:
        for lr in learning_rates:
            for depth in max_depths:
                model = train_gradient_boosting_model(
                    X, y, 
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth
                )
                
                # Evaluate on validation set
                val_score = r2_score(y_val, model.predict(X_val))
                test_score = r2_score(y, model.predict(X))
                
                results.append({
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'max_depth': depth,
                    'train_r2': test_score,
                    'val_r2': val_score
                })
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'max_depth': depth
                    }
                    best_model = model
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    return best_model, best_params, results_df

def plot_feature_importance(model, feature_names=None):
    """
    Plot feature importance from the trained model
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - Gradient Boosting")
    plt.bar(range(len(importances)), importances[indices], align="center")
    
    if feature_names is not None:
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    else:
        plt.xticks(range(len(importances)), indices)
    
    plt.tight_layout()
    plt.savefig('IACRF/results/gb_feature_importance.png')
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
    
    # Use original data without normalization for now
    X_norm = X
    y_norm = y
    X_val_norm = X_val
    y_val_norm = y_val
    
    print("Training Gradient Boosting model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    
    # Train model with default parameters
    gb_model = train_gradient_boosting_model(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(gb_model, X_test, y_test, X_val_norm, y_val_norm)
    
    print("\nGradient Boosting Model Results:")
    print(f"Test R² Score: {results['test_r2']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    
    if 'val_r2' in results:
        print(f"Validation R² Score: {results['val_r2']:.4f}")
        print(f"Validation MAE: {results['val_mae']:.4f}")
        print(f"Validation RMSE: {results['val_rmse']:.4f}")
    
    # Save model
    model_path = save_model(gb_model, 'gradient_boosting', results['test_r2'], 
                           results.get('val_r2', None))
    print(f"Model saved to: {model_path}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    best_model, best_params, tuning_results = hyperparameter_tuning(X_train, y_train, X_test, y_test)
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Evaluate best model
    best_results = evaluate_model(best_model, X_test, y_test, X_val_norm, y_val_norm)
    
    print("\nBest Gradient Boosting Model Results:")
    print(f"Test R² Score: {best_results['test_r2']:.4f}")
    print(f"Test MAE: {best_results['test_mae']:.4f}")
    print(f"Test RMSE: {best_results['test_rmse']:.4f}")
    
    if 'val_r2' in best_results:
        print(f"Validation R² Score: {best_results['val_r2']:.4f}")
        print(f"Validation MAE: {best_results['val_mae']:.4f}")
        print(f"Validation RMSE: {best_results['val_rmse']:.4f}")
    
    # Save best model
    best_model_path = save_model(best_model, 'gradient_boosting_tuned', best_results['test_r2'], 
                                best_results.get('val_r2', None))
    print(f"Best model saved to: {best_model_path}")
    
    # Plot feature importance
    try:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        plot_feature_importance(best_model, feature_names)
        print("Feature importance plot saved to: IACRF/results/gb_feature_importance.png")
    except Exception as e:
        print(f"Error plotting feature importance: {e}") 