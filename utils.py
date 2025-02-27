import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import time
import pickle

def load_data(train_file='normal-wf.csv', test_file='SouthEastpredict02.csv'):
    """
    Load training and testing data from CSV files
    """
    # Load training data
    print(f'IACRF/data/raw_data/{train_file}')
    df = pd.read_csv(f'IACRF/data/raw_data/{train_file}')
    a = np.array(df)
    Y = a[:, -1]
    X = a[:, :-1]
    
    # Load testing data
    print(f'IACRF/data/raw_data/{test_file}')
    ts = pd.read_csv(f'IACRF/data/raw_data/{test_file}')
    tsa = np.array(ts)
    ts_x = tsa[:, :-1]
    ts_y = tsa[:, -1]
    
    return X, Y, ts_x, ts_y

def normalize_features(X):
    """
    Normalize features to [0, 1] range
    """
    X_norm = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        col = X[:, i]
        col_min = np.min(col)
        col_max = np.max(col)
        if col_max > col_min:
            X_norm[:, i] = (col - col_min) / (col_max - col_min)
        else:
            X_norm[:, i] = 0.0
    return X_norm

def normalize_target(y):
    """
    Normalize target variable to [0, 1] range
    """
    y_min = np.min(y)
    y_max = np.max(y)
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y)
    return y_norm

def evaluate_model(model, X_test, y_test, X_val=None, y_val=None):
    """
    Evaluate model performance on test and validation sets
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
    
    # If validation data is provided, evaluate on it too
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        results.update({
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })
    
    return results

def save_model(model, model_name, score, val_score=None):
    """
    Save trained model to disk
    """
    outdir = f'IACRF/models/{model_name}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    tm = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    score_100 = round(score, 2) * 100
    
    if val_score is not None:
        val_score_100 = round(val_score, 2) * 100
        filename = f'{score_100}-{val_score_100}-{tm}.pickle'
    else:
        filename = f'{score_100}-{tm}.pickle'
    
    with open(f'{outdir}/{filename}', 'wb') as f:
        pickle.dump(model, f)
    
    return f'{outdir}/{filename}' 