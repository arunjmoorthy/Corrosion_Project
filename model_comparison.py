import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import time
import pickle
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# Import utility functions
from utils import load_data, normalize_features, normalize_target, evaluate_model, save_model

# Import model training functions
from gradient_boosting_model import train_gradient_boosting_model
from svr_model import train_svr_model
# Removing XGBoost import to avoid errors
# from xgboost_model import train_xgboost_model

# Try to import neural network model (may require TensorFlow)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    from neural_network_model import create_neural_network, train_neural_network, KerasWrapper
    tensorflow_available = True
    print("TensorFlow is available. Neural Network model will be included.")
except ImportError as e:
    tensorflow_available = False
    print(f"TensorFlow not available. Neural Network model will be skipped. Error: {e}")

def train_random_forest_model(X, y, n_estimators=100, max_depth=None, 
                             min_samples_split=2, random_state=None):
    """
    Train a Random Forest model for corrosion prediction
    """
    # Initialize the model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X, y)
    
    return rf_model

def compare_models(X_train, y_train, X_test, y_test):
    """
    Train and compare different regression models
    """
    # Dictionary to store models and their results
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest_model(X_train, y_train, random_state=42)
    models['Random Forest'] = rf_model
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test)
    
    # 2. Gradient Boosting
    print("\nTraining Gradient Boosting model...")
    gb_model = train_gradient_boosting_model(X_train, y_train, random_state=42)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test)
    
    # 3. Support Vector Regression
    print("\nTraining SVR model...")
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svr_model = train_svr_model(X_train_scaled, y_train)
    models['SVR'] = svr_model
    results['SVR'] = evaluate_model(svr_model, X_test_scaled, y_test)
    
    # 4. Neural Network (if TensorFlow is available)
    if tensorflow_available:
        print("\nTraining Neural Network model...")
        try:
            # Scale features for Neural Network
            if 'X_train_scaled' not in locals():
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            # Create and train neural network
            nn_model, history = train_neural_network(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                hidden_layers=[64, 32],
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                patience=20
            )
            
            # Wrap the model for scikit-learn compatibility
            wrapped_nn_model = KerasWrapper(nn_model)
            
            models['Neural Network'] = wrapped_nn_model
            results['Neural Network'] = evaluate_model(wrapped_nn_model, X_test_scaled, y_test)
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('MAE Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('IACRF/results/nn_training_history.png')
            plt.close()
            
            print("Neural Network training history saved to: IACRF/results/nn_training_history.png")
            
            # Save the neural network model
            nn_model.save('IACRF/models/neural_network/model.keras')
            print("Neural Network model saved to: IACRF/models/neural_network/model.keras")
        except Exception as e:
            print(f"Error training Neural Network model: {e}")
            print("Neural Network model will be skipped.")
    
    return models, results

def plot_model_comparison(results):
    """
    Plot comparison of model performance with improved visualization
    """
    # Extract metrics for comparison
    model_names = list(results.keys())
    test_r2_scores = [results[model]['test_r2'] for model in model_names]
    test_mae_scores = [results[model]['test_mae'] for model in model_names]
    test_rmse_scores = [results[model]['test_rmse'] for model in model_names]
    
    # Create figure with improved styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Plot R² scores
    bar1 = axes[0].bar(model_names, test_r2_scores, color=colors[:len(model_names)], alpha=0.8)
    axes[0].set_title('R² Score (higher is better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_ylim(0, max(1.0, max(test_r2_scores) * 1.1))
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bar1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot MAE scores
    bar2 = axes[1].bar(model_names, test_mae_scores, color=colors[:len(model_names)], alpha=0.8)
    axes[1].set_title('Mean Absolute Error (lower is better)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bar2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot RMSE scores
    bar3 = axes[2].bar(model_names, test_rmse_scores, color=colors[:len(model_names)], alpha=0.8)
    axes[2].set_title('Root Mean Squared Error (lower is better)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('RMSE', fontsize=12)
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bar3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('IACRF/results/model_comparison.png', dpi=300)
    plt.close()

def cross_validate_models(X, y, cv=5):
    """
    Perform cross-validation for all models
    """
    # Dictionary to store cross-validation results
    cv_results = {}
    
    # 1. Random Forest
    print("\nCross-validating Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['Random Forest'] = rf_scores
    
    # 2. Gradient Boosting
    print("Cross-validating Gradient Boosting model...")
    from sklearn.ensemble import GradientBoostingRegressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_scores = cross_val_score(gb_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['Gradient Boosting'] = gb_scores
    
    # 3. Support Vector Regression
    print("Cross-validating SVR model...")
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Create pipeline with scaling for SVR
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    svr_scores = cross_val_score(svr_pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['SVR'] = svr_scores
    
    # Neural Network is not included in cross-validation due to complexity
    # and different training approach
    
    return cv_results

def plot_cross_validation_results(cv_results):
    """
    Plot cross-validation results with improved visualization
    """
    model_names = list(cv_results.keys())
    mean_scores = [cv_results[model].mean() for model in model_names]
    std_scores = [cv_results[model].std() for model in model_names]
    
    # Create figure with improved styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    # Color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Create bars with error bars
    bars = plt.bar(model_names, mean_scores, yerr=std_scores, capsize=10, 
                  color=colors[:len(model_names)], alpha=0.8, ecolor='black', width=0.6)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = mean_scores[i]
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}±{std_scores[i]:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Customize plot
    plt.title('Cross-Validation R² Scores (5-fold)', fontsize=16, fontweight='bold')
    plt.ylabel('Mean R² Score', fontsize=14)
    plt.ylim(min(0, min(mean_scores) - max(std_scores) - 0.1), 
             max(1.0, max(mean_scores) + max(std_scores) + 0.1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add a grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('IACRF/results/cross_validation_comparison.png', dpi=300)
    plt.close()

def create_results_table(results, cv_results=None):
    """
    Create a DataFrame with model results
    """
    model_names = list(results.keys())
    
    # Create DataFrame for test results
    test_data = {
        'Model': model_names,
        'Test R²': [results[model]['test_r2'] for model in model_names],
        'Test MAE': [results[model]['test_mae'] for model in model_names],
        'Test RMSE': [results[model]['test_rmse'] for model in model_names]
    }
    
    # Add cross-validation results if available
    if cv_results is not None:
        test_data.update({
            'CV R² (mean)': [cv_results[model].mean() if model in cv_results else np.nan for model in model_names],
            'CV R² (std)': [cv_results[model].std() if model in cv_results else np.nan for model in model_names]
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(test_data)
    
    # Sort by Test R² score (descending)
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    return results_df

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set
    """
    # Predict on test set
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results = {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }
    
    return results

def save_model(model, model_name, score):
    """
    Save trained model to disk
    """
    outdir = f'IACRF/models/{model_name}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    tm = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    score_100 = round(score, 2) * 100
    
    filename = f'{score_100}-{tm}.pickle'
    
    with open(f'{outdir}/{filename}', 'wb') as f:
        pickle.dump(model, f)
    
    return f'{outdir}/{filename}'

def load_data_from_csv():
    """
    Load data from normal-wf.csv and split it into train and test sets
    """
    try:
        # Load data from CSV file
        file_path = 'IACRF/data/raw_data/normal-wf.csv'
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Extract features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Split data into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Data loaded successfully. Total samples: {len(df)}")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Create figure
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        
        # Plot feature importances
        plt.bar(range(len(indices)), importances[indices], color='#3498db', alpha=0.8)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, len(indices)])
        plt.tight_layout()
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        plt.ylabel('Importance', fontsize=14)
        plt.savefig('IACRF/results/feature_importance.png', dpi=300)
        plt.close()
        
        print("Feature importance plot saved to: IACRF/results/feature_importance.png")
        
        # Return sorted feature importances
        return [(feature_names[i], importances[i]) for i in indices]
    else:
        print("Model does not have feature_importances_ attribute")
        return None

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """
    Plot actual vs predicted values
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred, alpha=0.7, color='#3498db')
    
    # Plot perfect prediction line
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Actual vs Predicted Values - {model_name}', fontsize=16, fontweight='bold')
    
    # Add R² score
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'IACRF/results/actual_vs_predicted_{model_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()
    
    print(f"Actual vs predicted plot saved for {model_name}")

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('IACRF/results'):
        os.makedirs('IACRF/results')
    
    # Create models directory if it doesn't exist
    if not os.path.exists('IACRF/models'):
        os.makedirs('IACRF/models')
    
    # Create neural network model directories if they don't exist
    if tensorflow_available:
        if not os.path.exists('IACRF/models/neural_network'):
            os.makedirs('IACRF/models/neural_network')
        if not os.path.exists('IACRF/models/neural_network_tuned'):
            os.makedirs('IACRF/models/neural_network_tuned')
        if not os.path.exists('IACRF/models/neural_network_checkpoints'):
            os.makedirs('IACRF/models/neural_network_checkpoints')
    
    # Load data from CSV and split into train and test sets
    X_train, X_test, y_train, y_test = load_data_from_csv()
    
    # Get feature names from CSV file
    feature_names = pd.read_csv('IACRF/data/raw_data/normal-wf.csv').columns[:-1].tolist()
    
    print("Comparing different regression models for corrosion prediction...")
    
    # Train and compare models
    models, results = compare_models(X_train, y_train, X_test, y_test)
    
    # Plot model comparison
    plot_model_comparison(results)
    print("\nModel comparison plot saved to: IACRF/results/model_comparison.png")
    
    # Perform cross-validation
    print("\nPerforming cross-validation for all models...")
    cv_results = cross_validate_models(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=5)
    
    # Plot cross-validation results
    plot_cross_validation_results(cv_results)
    print("Cross-validation comparison plot saved to: IACRF/results/cross_validation_comparison.png")
    
    # Create results table
    results_table = create_results_table(results, cv_results)
    
    # Save results table to CSV
    results_table.to_csv('IACRF/results/model_comparison_results.csv', index=False)
    print("\nResults table saved to: IACRF/results/model_comparison_results.csv")
    
    # Print results table
    print("\nModel Comparison Results:")
    print(results_table.to_string(index=False))
    
    # Find best model
    best_model_name = results_table.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
    
    # Plot feature importance for the best model (if applicable)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = plot_feature_importance(best_model, feature_names)
        if feature_importance:
            print("\nFeature Importance:")
            for feature, importance in feature_importance:
                print(f"{feature}: {importance:.4f}")
    
    # Plot actual vs predicted values for all models
    print("\nGenerating actual vs predicted plots...")
    for model_name, model in models.items():
        if model_name in ['SVR', 'Neural Network']:
            # Scale features for SVR and Neural Network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        plot_actual_vs_predicted(y_test, y_pred, model_name)
    
    # Save best model
    best_model_path = save_model(best_model, f'best_model_{best_model_name}', 
                                results[best_model_name]['test_r2'])
    print(f"Best model saved to: {best_model_path}") 