import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import save_model

def load_original_data():
    """
    Load the original data from CSV files
    """
    try:
        # Load training data
        train_path = 'IACRF/data/raw_data/normal-wf.csv'
        print(f"Loading original training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        
        # Handle missing values in training data
        train_df = train_df.fillna(train_df.mean())
        
        # Extract features and target
        X = train_df.iloc[:, :-1].values
        y = train_df.iloc[:, -1].values
        
        # Load validation data
        val_path = 'IACRF/data/raw_data/SouthEastpredict02.csv'
        print(f"Loading original validation data from: {val_path}")
        val_df = pd.read_csv(val_path)
        
        # Handle missing values in validation data
        # For validation data, we'll fill missing values with the mean from training data
        for col in val_df.columns:
            if col in train_df.columns:
                val_df[col] = val_df[col].fillna(train_df[col].mean())
        
        # Fill any remaining NaNs with column means
        val_df = val_df.fillna(val_df.mean())
        
        # Extract features and target
        X_val = val_df.iloc[:, :-1].values
        y_val = val_df.iloc[:, -1].values
        
        # If validation target is all NaN, set to zeros (for demonstration)
        if np.all(np.isnan(y_val)):
            print("Warning: Validation target values are all NaN. Setting to zeros for demonstration.")
            y_val = np.zeros_like(y_val)
        
        return train_df, X, y, val_df, X_val, y_val
    
    except Exception as e:
        print(f"Error loading original data: {e}")
        raise

def generate_synthetic_data(original_df, n_samples=300):
    """
    Generate synthetic data based on the statistical properties of the original data
    """
    print(f"Generating {n_samples} synthetic data samples...")
    
    # Create a copy of the original dataframe to understand its structure
    synthetic_df = pd.DataFrame(columns=original_df.columns)
    
    # Calculate mean and standard deviation for each feature
    means = original_df.mean()
    stds = original_df.std()
    
    # Calculate correlations between features and target
    correlations = original_df.corr()['Corrosion_Rate'].drop('Corrosion_Rate')
    
    # Generate synthetic data for each feature
    synthetic_data = {}
    
    # First generate random values for each feature
    for col in original_df.columns:
        if col != 'Corrosion_Rate':
            # Use the mean and std of the original data, but add some noise
            synthetic_data[col] = np.random.normal(
                loc=means[col], 
                scale=stds[col] * 1.2,  # Slightly increase variability
                size=n_samples
            )
    
    # Create a DataFrame with the synthetic features
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Generate synthetic target values based on feature correlations
    # This is a simplified approach to maintain some relationship between features and target
    synthetic_target = np.zeros(n_samples)
    
    for col in correlations.index:
        # Weight the contribution of each feature by its correlation with the target
        synthetic_target += correlations[col] * (synthetic_df[col] - means[col]) / stds[col]
    
    # Scale the target to match the original distribution
    synthetic_target = synthetic_target * stds['Corrosion_Rate'] + means['Corrosion_Rate']
    
    # Add some random noise to the target
    synthetic_target += np.random.normal(0, stds['Corrosion_Rate'] * 0.3, n_samples)
    
    # Ensure no negative corrosion rates
    synthetic_target = np.maximum(synthetic_target, 0)
    
    # Add the target to the synthetic dataframe
    synthetic_df['Corrosion_Rate'] = synthetic_target
    
    # Extract features and target
    X_synthetic = synthetic_df.iloc[:, :-1].values
    y_synthetic = synthetic_df.iloc[:, -1].values
    
    return synthetic_df, X_synthetic, y_synthetic

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

def train_svr_model(X, y, C=1.0, epsilon=0.1, kernel='rbf'):
    """
    Train a Support Vector Regression model for corrosion prediction
    """
    # Initialize the model
    svr_model = SVR(
        C=C,
        epsilon=epsilon,
        kernel=kernel
    )
    
    # Train the model
    svr_model.fit(X, y)
    
    return svr_model

def evaluate_model(model, X_test, y_test, X_val=None, y_val=None):
    """
    Evaluate a model on test and validation data, handling NaN values
    """
    # Predict on test set
    y_pred_test = model.predict(X_test)
    
    # Filter out NaN values for evaluation
    mask_test = ~np.isnan(y_test)
    if np.sum(mask_test) == 0:
        print("Warning: All test target values are NaN. Cannot evaluate on test set.")
        test_r2 = np.nan
        test_mae = np.nan
        test_rmse = np.nan
    else:
        test_r2 = r2_score(y_test[mask_test], y_pred_test[mask_test])
        test_mae = mean_absolute_error(y_test[mask_test], y_pred_test[mask_test])
        test_rmse = np.sqrt(mean_squared_error(y_test[mask_test], y_pred_test[mask_test]))
    
    results = {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }
    
    # If validation data is provided, evaluate on it too
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        
        # Filter out NaN values for evaluation
        mask_val = ~np.isnan(y_val)
        if np.sum(mask_val) == 0:
            print("Warning: All validation target values are NaN. Cannot evaluate on validation set.")
            val_r2 = np.nan
            val_mae = np.nan
            val_rmse = np.nan
        else:
            val_r2 = r2_score(y_val[mask_val], y_pred_val[mask_val])
            val_mae = mean_absolute_error(y_val[mask_val], y_pred_val[mask_val])
            val_rmse = np.sqrt(mean_squared_error(y_val[mask_val], y_pred_val[mask_val]))
        
        results.update({
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })
    
    return results

def compare_models(X_train, y_train, X_test, y_test, X_val=None, y_val=None, dataset_name=""):
    """
    Train and compare different regression models
    """
    # Dictionary to store models and their results
    models = {}
    results = {}
    
    # 1. Random Forest
    print(f"\nTraining Random Forest model on {dataset_name}...")
    rf_model = train_random_forest_model(X_train, y_train, random_state=42)
    models['Random Forest'] = rf_model
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, X_val, y_val)
    
    # 2. Gradient Boosting
    print(f"\nTraining Gradient Boosting model on {dataset_name}...")
    gb_model = train_gradient_boosting_model(X_train, y_train, random_state=42)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test, X_val, y_val)
    
    # 3. Support Vector Regression
    print(f"\nTraining SVR model on {dataset_name}...")
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    
    svr_model = train_svr_model(X_train_scaled, y_train)
    models['SVR'] = svr_model
    results['SVR'] = evaluate_model(svr_model, X_test_scaled, y_test, X_val_scaled, y_val)
    
    return models, results

def cross_validate_models(X, y, cv=5, dataset_name=""):
    """
    Perform cross-validation for all models
    """
    # Dictionary to store cross-validation results
    cv_results = {}
    
    # 1. Random Forest
    print(f"\nCross-validating Random Forest model on {dataset_name}...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['Random Forest'] = rf_scores
    
    # 2. Gradient Boosting
    print(f"Cross-validating Gradient Boosting model on {dataset_name}...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_scores = cross_val_score(gb_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['Gradient Boosting'] = gb_scores
    
    # 3. Support Vector Regression
    print(f"Cross-validating SVR model on {dataset_name}...")
    # Create pipeline with scaling for SVR
    from sklearn.pipeline import Pipeline
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    svr_scores = cross_val_score(svr_pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)
    cv_results['SVR'] = svr_scores
    
    return cv_results

def plot_model_comparison(results, dataset_name=""):
    """
    Plot model comparison results
    """
    model_names = list(results.keys())
    test_scores = [results[model]['test_r2'] for model in model_names]
    val_scores = [results[model].get('val_r2', 0) for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, test_scores, width, label='Test R²', color='skyblue')
    rects2 = ax.bar(x + width/2, val_scores, width, label='Validation R²', color='lightgreen')
    
    ax.set_title(f'Model Comparison - {dataset_name}')
    ax.set_ylabel('R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add score values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f'IACRF/results/model_comparison_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_cross_validation_results(cv_results, dataset_name=""):
    """
    Plot cross-validation results
    """
    model_names = list(cv_results.keys())
    mean_scores = [cv_results[model].mean() for model in model_names]
    std_scores = [cv_results[model].std() for model in model_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mean_scores, yerr=std_scores, capsize=10, color='skyblue', alpha=0.8)
    plt.title(f'Cross-Validation R² Scores (5-fold) - {dataset_name}')
    plt.ylabel('Mean R² Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add mean score values on top of bars
    for i, score in enumerate(mean_scores):
        plt.text(i, score + 0.02, f"{score:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'IACRF/results/cross_validation_comparison_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close()

def create_results_table(results, cv_results=None, dataset_name=""):
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
    
    # Add validation results if available
    if 'val_r2' in results[model_names[0]]:
        test_data.update({
            'Validation R²': [results[model]['val_r2'] for model in model_names],
            'Validation MAE': [results[model]['val_mae'] for model in model_names],
            'Validation RMSE': [results[model]['val_rmse'] for model in model_names]
        })
    
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

def plot_combined_comparison(original_results, synthetic_results):
    """
    Plot a comparison of model performance between original and synthetic data
    """
    model_names = list(original_results.keys())
    
    original_test_scores = [original_results[model]['test_r2'] for model in model_names]
    synthetic_test_scores = [synthetic_results[model]['test_r2'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, original_test_scores, width, label='Original Data', color='skyblue')
    rects2 = ax.bar(x + width/2, synthetic_test_scores, width, label='Synthetic Data', color='salmon')
    
    ax.set_title('Model Performance Comparison: Original vs. Synthetic Data')
    ax.set_ylabel('Test R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add score values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('IACRF/results/original_vs_synthetic_comparison.png')
    plt.close()

def save_synthetic_data(synthetic_df):
    """
    Save the synthetic data to a CSV file
    """
    # Create directory if it doesn't exist
    if not os.path.exists('IACRF/data/synthetic_data'):
        os.makedirs('IACRF/data/synthetic_data')
    
    # Save to CSV
    file_path = 'IACRF/data/synthetic_data/synthetic_corrosion_data.csv'
    synthetic_df.to_csv(file_path, index=False)
    print(f"Synthetic data saved to: {file_path}")
    
    return file_path

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    if not os.path.exists('IACRF/results'):
        os.makedirs('IACRF/results')
    
    # Load original data
    try:
        original_train_df, X_original, y_original, original_val_df, X_val_original, y_val_original = load_original_data()
        print("Original data loaded successfully.")
        print(f"Original training data shape: {X_original.shape}")
        print(f"Original validation data shape: {X_val_original.shape}")
    except Exception as e:
        print(f"Error loading original data: {e}")
        exit(1)
    
    # Generate synthetic data
    try:
        synthetic_df, X_synthetic, y_synthetic = generate_synthetic_data(original_train_df, n_samples=300)
        print(f"Synthetic data generated successfully. Shape: {X_synthetic.shape}")
        
        # Save synthetic data
        synthetic_data_path = save_synthetic_data(synthetic_df)
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        exit(1)
    
    # Split original data into train and test sets
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42
    )
    
    # Split synthetic data into train and test sets
    X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42
    )
    
    print("\n" + "="*50)
    print("TRAINING AND EVALUATING MODELS ON ORIGINAL DATA")
    print("="*50)
    
    # Train and compare models on original data
    original_models, original_results = compare_models(
        X_train_original, y_train_original, 
        X_test_original, y_test_original, 
        X_val_original, y_val_original,
        dataset_name="Original Data"
    )
    
    # Perform cross-validation on original data
    print("\nPerforming cross-validation for all models on original data...")
    original_cv_results = cross_validate_models(X_original, y_original, cv=5, dataset_name="Original Data")
    
    # Plot model comparison for original data
    plot_model_comparison(original_results, dataset_name="Original Data")
    print("\nModel comparison plot for original data saved to: IACRF/results/model_comparison_original_data.png")
    
    # Plot cross-validation results for original data
    plot_cross_validation_results(original_cv_results, dataset_name="Original Data")
    print("Cross-validation comparison plot for original data saved to: IACRF/results/cross_validation_comparison_original_data.png")
    
    # Create results table for original data
    original_results_table = create_results_table(original_results, original_cv_results, dataset_name="Original Data")
    
    # Save results table to CSV
    original_results_table.to_csv('IACRF/results/model_comparison_results_original_data.csv', index=False)
    print("\nResults table for original data saved to: IACRF/results/model_comparison_results_original_data.csv")
    
    # Print results table for original data
    print("\nModel Comparison Results (Original Data):")
    print(original_results_table.to_string(index=False))
    
    # Find best model for original data
    best_model_name_original = original_results_table.iloc[0]['Model']
    best_model_original = original_models[best_model_name_original]
    
    print(f"\nBest model on original data: {best_model_name_original}")
    print(f"Test R² Score: {original_results[best_model_name_original]['test_r2']:.4f}")
    
    if 'val_r2' in original_results[best_model_name_original]:
        print(f"Validation R² Score: {original_results[best_model_name_original]['val_r2']:.4f}")
    
    # Save best model for original data
    best_model_path_original = save_model(
        best_model_original, 
        f'best_model_original_{best_model_name_original}', 
        original_results[best_model_name_original]['test_r2'],
        original_results[best_model_name_original].get('val_r2', None)
    )
    print(f"Best model on original data saved to: {best_model_path_original}")
    
    print("\n" + "="*50)
    print("TRAINING AND EVALUATING MODELS ON SYNTHETIC DATA")
    print("="*50)
    
    # Train and compare models on synthetic data
    synthetic_models, synthetic_results = compare_models(
        X_train_synthetic, y_train_synthetic, 
        X_test_synthetic, y_test_synthetic, 
        X_val_original, y_val_original,  # Use original validation data for consistent comparison
        dataset_name="Synthetic Data"
    )
    
    # Perform cross-validation on synthetic data
    print("\nPerforming cross-validation for all models on synthetic data...")
    synthetic_cv_results = cross_validate_models(X_synthetic, y_synthetic, cv=5, dataset_name="Synthetic Data")
    
    # Plot model comparison for synthetic data
    plot_model_comparison(synthetic_results, dataset_name="Synthetic Data")
    print("\nModel comparison plot for synthetic data saved to: IACRF/results/model_comparison_synthetic_data.png")
    
    # Plot cross-validation results for synthetic data
    plot_cross_validation_results(synthetic_cv_results, dataset_name="Synthetic Data")
    print("Cross-validation comparison plot for synthetic data saved to: IACRF/results/cross_validation_comparison_synthetic_data.png")
    
    # Create results table for synthetic data
    synthetic_results_table = create_results_table(synthetic_results, synthetic_cv_results, dataset_name="Synthetic Data")
    
    # Save results table to CSV
    synthetic_results_table.to_csv('IACRF/results/model_comparison_results_synthetic_data.csv', index=False)
    print("\nResults table for synthetic data saved to: IACRF/results/model_comparison_results_synthetic_data.csv")
    
    # Print results table for synthetic data
    print("\nModel Comparison Results (Synthetic Data):")
    print(synthetic_results_table.to_string(index=False))
    
    # Find best model for synthetic data
    best_model_name_synthetic = synthetic_results_table.iloc[0]['Model']
    best_model_synthetic = synthetic_models[best_model_name_synthetic]
    
    print(f"\nBest model on synthetic data: {best_model_name_synthetic}")
    print(f"Test R² Score: {synthetic_results[best_model_name_synthetic]['test_r2']:.4f}")
    
    if 'val_r2' in synthetic_results[best_model_name_synthetic]:
        print(f"Validation R² Score: {synthetic_results[best_model_name_synthetic]['val_r2']:.4f}")
    
    # Save best model for synthetic data
    best_model_path_synthetic = save_model(
        best_model_synthetic, 
        f'best_model_synthetic_{best_model_name_synthetic}', 
        synthetic_results[best_model_name_synthetic]['test_r2'],
        synthetic_results[best_model_name_synthetic].get('val_r2', None)
    )
    print(f"Best model on synthetic data saved to: {best_model_path_synthetic}")
    
    # Plot combined comparison
    plot_combined_comparison(original_results, synthetic_results)
    print("\nCombined comparison plot saved to: IACRF/results/original_vs_synthetic_comparison.png")
    
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    print(f"Best model on original data: {best_model_name_original} (Test R²: {original_results[best_model_name_original]['test_r2']:.4f})")
    print(f"Best model on synthetic data: {best_model_name_synthetic} (Test R²: {synthetic_results[best_model_name_synthetic]['test_r2']:.4f})")
    print("\nCheck the results directory for detailed plots and comparison tables.") 