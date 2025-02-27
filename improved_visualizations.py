import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import neural network components
try:
    import tensorflow as tf
    from neural_network_model import create_neural_network, train_neural_network, KerasWrapper
    tensorflow_available = True
    print("TensorFlow is available. Neural Network model will be included.")
except ImportError as e:
    tensorflow_available = False
    print(f"TensorFlow not available. Neural Network model will be skipped. Error: {e}")

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_original_data():
    """
    Load the original data from CSV file
    """
    file_path = 'IACRF/data/raw_data/normal-wf.csv'
    print(f"Loading original data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Extract features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Original data loaded successfully. Total samples: {len(df)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, df.columns[:-1].tolist()

def load_synthetic_data():
    """
    Load the synthetic data from CSV file
    """
    file_path = 'IACRF/data/synthetic_data/synthetic_corrosion_data.csv'
    print(f"Loading synthetic data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Extract features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Synthetic data loaded successfully. Total samples: {len(df)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

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

def train_gradient_boosting_model(X, y, n_estimators=100, learning_rate=0.1, 
                                 max_depth=3, random_state=None):
    """
    Train a Gradient Boosting model for corrosion prediction
    """
    # Initialize the model
    gb_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    
    # Train the model
    gb_model.fit(X, y)
    
    return gb_model

def train_svr_model(X, y, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
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

def train_and_evaluate_models(X_train, y_train, X_test, y_test, dataset_name="Original"):
    """
    Train and evaluate all models on the given dataset
    """
    print(f"\nTraining and evaluating models on {dataset_name} data...")
    
    # Dictionary to store models and their results
    models = {}
    results = {}
    
    # 1. Random Forest
    print(f"Training Random Forest model on {dataset_name} data...")
    rf_model = train_random_forest_model(X_train, y_train, random_state=42)
    models['Random Forest'] = rf_model
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test)
    
    # 2. Gradient Boosting
    print(f"Training Gradient Boosting model on {dataset_name} data...")
    gb_model = train_gradient_boosting_model(X_train, y_train, random_state=42)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test)
    
    # 3. Support Vector Regression
    print(f"Training SVR model on {dataset_name} data...")
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svr_model = train_svr_model(X_train_scaled, y_train)
    models['SVR'] = svr_model
    results['SVR'] = evaluate_model(svr_model, X_test_scaled, y_test)
    
    # 4. Neural Network (if TensorFlow is available)
    if tensorflow_available:
        print(f"Training Neural Network model on {dataset_name} data...")
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
            
            # Save the neural network model
            if not os.path.exists(f'IACRF/models/neural_network_{dataset_name.lower()}'):
                os.makedirs(f'IACRF/models/neural_network_{dataset_name.lower()}')
            nn_model.save(f'IACRF/models/neural_network_{dataset_name.lower()}/model.keras')
            print(f"Neural Network model saved to: IACRF/models/neural_network_{dataset_name.lower()}/model.keras")
        except Exception as e:
            print(f"Error training Neural Network model: {e}")
            print("Neural Network model will be skipped.")
    
    return models, results

def create_results_table(results, dataset_name="Original"):
    """
    Create a DataFrame with model results
    """
    model_names = list(results.keys())
    
    # Create DataFrame for test results
    test_data = {
        'Model': model_names,
        'R²': [results[model]['test_r2'] for model in model_names],
        'MAE': [results[model]['test_mae'] for model in model_names],
        'RMSE': [results[model]['test_rmse'] for model in model_names]
    }
    
    # Create DataFrame
    results_df = pd.DataFrame(test_data)
    
    # Sort by R² score (descending)
    results_df = results_df.sort_values('R²', ascending=False)
    
    # Save results to CSV
    results_df.to_csv(f'IACRF/results/{dataset_name.lower()}_data_results.csv', index=False)
    print(f"Results table saved to: IACRF/results/{dataset_name.lower()}_data_results.csv")
    
    return results_df

def plot_model_comparison(results, dataset_name="Original"):
    """
    Create an improved model comparison plot
    """
    # Extract data
    model_names = list(results.keys())
    r2_scores = [results[model]['test_r2'] for model in model_names]
    mae_scores = [results[model]['test_mae'] for model in model_names]
    rmse_scores = [results[model]['test_rmse'] for model in model_names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Plot R² scores
    sns.barplot(x=model_names, y=r2_scores, ax=axes[0], palette="Blues_d")
    axes[0].set_title(f'R² Score (higher is better)\n{dataset_name} Data', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_ylim(0, max(1.0, max(r2_scores) * 1.1))
    
    # Add value labels
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot MAE scores
    sns.barplot(x=model_names, y=mae_scores, ax=axes[1], palette="Reds_d")
    axes[1].set_title(f'Mean Absolute Error (lower is better)\n{dataset_name} Data', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(mae_scores):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot RMSE scores
    sns.barplot(x=model_names, y=rmse_scores, ax=axes[2], palette="Greens_d")
    axes[2].set_title(f'Root Mean Squared Error (lower is better)\n{dataset_name} Data', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('RMSE', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(rmse_scores):
        axes[2].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'IACRF/results/{dataset_name.lower()}_data_model_comparison.png', dpi=300)
    plt.close()
    
    print(f"Model comparison plot saved to: IACRF/results/{dataset_name.lower()}_data_model_comparison.png")

def plot_combined_comparison(original_results, synthetic_results):
    """
    Create an improved combined comparison plot for original vs synthetic data
    """
    # Get common models between both results
    common_models = [model for model in original_results.keys() if model in synthetic_results]
    
    # Create figure with subplots for R², MAE, and RMSE
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Set width for bars
    width = 0.35
    x = np.arange(len(common_models))
    
    # Plot R² scores
    original_r2 = [original_results[model]['test_r2'] for model in common_models]
    synthetic_r2 = [synthetic_results[model]['test_r2'] for model in common_models]
    
    axes[0].bar(x - width/2, original_r2, width, label='Original Data', color='#3498db')
    axes[0].bar(x + width/2, synthetic_r2, width, label='Synthetic Data', color='#e74c3c')
    axes[0].set_title('R² Score Comparison (higher is better)', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(common_models, fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(original_r2):
        axes[0].text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i, v in enumerate(synthetic_r2):
        axes[0].text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot MAE scores
    original_mae = [original_results[model]['test_mae'] for model in common_models]
    synthetic_mae = [synthetic_results[model]['test_mae'] for model in common_models]
    
    axes[1].bar(x - width/2, original_mae, width, label='Original Data', color='#3498db')
    axes[1].bar(x + width/2, synthetic_mae, width, label='Synthetic Data', color='#e74c3c')
    axes[1].set_title('Mean Absolute Error Comparison (lower is better)', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(common_models, fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(original_mae):
        axes[1].text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i, v in enumerate(synthetic_mae):
        axes[1].text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot RMSE scores
    original_rmse = [original_results[model]['test_rmse'] for model in common_models]
    synthetic_rmse = [synthetic_results[model]['test_rmse'] for model in common_models]
    
    axes[2].bar(x - width/2, original_rmse, width, label='Original Data', color='#3498db')
    axes[2].bar(x + width/2, synthetic_rmse, width, label='Synthetic Data', color='#e74c3c')
    axes[2].set_title('Root Mean Squared Error Comparison (lower is better)', fontsize=16, fontweight='bold')
    axes[2].set_ylabel('RMSE', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(common_models, fontsize=12)
    axes[2].legend(fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(original_rmse):
        axes[2].text(i - width/2, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i, v in enumerate(synthetic_rmse):
        axes[2].text(i + width/2, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('IACRF/results/original_vs_synthetic_comparison.png', dpi=300)
    plt.close()
    
    print("Original vs Synthetic data comparison plot saved to: IACRF/results/original_vs_synthetic_comparison.png")

def create_combined_results_table(original_results, synthetic_results):
    """
    Create a combined results table for original and synthetic data
    """
    # Get common models between both results
    common_models = [model for model in original_results.keys() if model in synthetic_results]
    
    # Create DataFrame
    combined_data = {
        'Model': common_models,
        'Original R²': [original_results[model]['test_r2'] for model in common_models],
        'Synthetic R²': [synthetic_results[model]['test_r2'] for model in common_models],
        'Original MAE': [original_results[model]['test_mae'] for model in common_models],
        'Synthetic MAE': [synthetic_results[model]['test_mae'] for model in common_models],
        'Original RMSE': [original_results[model]['test_rmse'] for model in common_models],
        'Synthetic RMSE': [synthetic_results[model]['test_rmse'] for model in common_models]
    }
    
    combined_df = pd.DataFrame(combined_data)
    
    # Sort by Original R² score (descending)
    combined_df = combined_df.sort_values('Original R²', ascending=False)
    
    # Save results to CSV
    combined_df.to_csv('IACRF/results/combined_model_comparison.csv', index=False)
    print("Combined results table saved to: IACRF/results/combined_model_comparison.csv")
    
    return combined_df

def plot_actual_vs_predicted(models, X_test, y_test, dataset_name="Original"):
    """
    Create actual vs predicted plots for all models
    """
    # Create directory for actual vs predicted plots
    if not os.path.exists('IACRF/results/actual_vs_predicted'):
        os.makedirs('IACRF/results/actual_vs_predicted')
    
    # Create plots for each model
    for model_name, model in models.items():
        # Scale features for SVR and Neural Network if needed
        if model_name in ['SVR', 'Neural Network']:
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate R² score
        r2 = r2_score(y_test, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot actual vs predicted values
        plt.scatter(y_test, y_pred, alpha=0.7, s=80, edgecolor='k', linewidth=0.5)
        
        # Plot perfect prediction line
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f'Actual vs Predicted Values - {model_name}\n{dataset_name} Data', fontsize=16, fontweight='bold')
        
        # Add R² score
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'IACRF/results/actual_vs_predicted/{dataset_name.lower()}_{model_name.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
        
        print(f"Actual vs predicted plot saved for {model_name} on {dataset_name} data")

def plot_feature_importance(model, feature_names, dataset_name="Original"):
    """
    Create feature importance plot for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot feature importances
        plt.barh(range(len(indices)), importances[indices], color='#3498db', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=14)
        plt.title(f'Feature Importance - {dataset_name} Data', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'IACRF/results/{dataset_name.lower()}_feature_importance.png', dpi=300)
        plt.close()
        
        print(f"Feature importance plot saved for {dataset_name} data")
        
        # Return sorted feature importances
        return [(feature_names[i], importances[i]) for i in indices]
    else:
        print("Model does not have feature_importances_ attribute")
        return None

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('IACRF/results'):
        os.makedirs('IACRF/results')
    
    # Load original data
    X_train_orig, X_test_orig, y_train_orig, y_test_orig, feature_names = load_original_data()
    
    # Train and evaluate models on original data
    orig_models, orig_results = train_and_evaluate_models(X_train_orig, y_train_orig, X_test_orig, y_test_orig, "Original")
    
    # Create results table for original data
    orig_results_df = create_results_table(orig_results, "Original")
    
    # Plot model comparison for original data
    plot_model_comparison(orig_results, "Original")
    
    # Plot actual vs predicted for original data
    plot_actual_vs_predicted(orig_models, X_test_orig, y_test_orig, "Original")
    
    # Plot feature importance for best model on original data
    best_model_name = orig_results_df.iloc[0]['Model']
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = plot_feature_importance(orig_models[best_model_name], feature_names, "Original")
        if feature_importance:
            print("\nFeature Importance (Original Data):")
            for feature, importance in feature_importance:
                print(f"{feature}: {importance:.4f}")
    
    # Load synthetic data
    try:
        X_train_synth, X_test_synth, y_train_synth, y_test_synth = load_synthetic_data()
        
        # Train and evaluate models on synthetic data
        synth_models, synth_results = train_and_evaluate_models(X_train_synth, y_train_synth, X_test_synth, y_test_synth, "Synthetic")
        
        # Create results table for synthetic data
        synth_results_df = create_results_table(synth_results, "Synthetic")
        
        # Plot model comparison for synthetic data
        plot_model_comparison(synth_results, "Synthetic")
        
        # Plot actual vs predicted for synthetic data
        plot_actual_vs_predicted(synth_models, X_test_synth, y_test_synth, "Synthetic")
        
        # Plot feature importance for best model on synthetic data
        best_model_name = synth_results_df.iloc[0]['Model']
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            feature_importance = plot_feature_importance(synth_models[best_model_name], feature_names, "Synthetic")
            if feature_importance:
                print("\nFeature Importance (Synthetic Data):")
                for feature, importance in feature_importance:
                    print(f"{feature}: {importance:.4f}")
        
        # Create combined comparison plot and table
        plot_combined_comparison(orig_results, synth_results)
        combined_df = create_combined_results_table(orig_results, synth_results)
        
        print("\nCombined Model Comparison Results:")
        print(combined_df.to_string(index=False))
    except Exception as e:
        print(f"Error processing synthetic data: {e}")
        print("Skipping synthetic data comparison.") 