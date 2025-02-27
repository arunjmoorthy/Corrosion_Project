import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import time
from utils import load_data, normalize_features, normalize_target, evaluate_model, save_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create a wrapper class to make the Keras model compatible with scikit-learn
class KerasWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

def create_neural_network(input_dim, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    """
    Create a neural network model for corrosion prediction
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_layers : list
        List of integers representing the number of neurons in each hidden layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for the optimizer
        
    Returns:
    --------
    model : keras.models.Sequential
        Compiled neural network model
    """
    # Create model
    model = Sequential()
    
    # Add input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    # Add hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Add output layer
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def train_neural_network(X_train, y_train, X_val, y_val,
                        hidden_layers=[64, 32],
                        dropout_rate=0.2,
                        learning_rate=0.001,
                        batch_size=32,
                        epochs=100,
                        patience=20):
    """
    Train a neural network model for corrosion prediction
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
    hidden_layers : list
        List of integers representing the number of neurons in each hidden layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for the optimizer
    batch_size : int
        Batch size for training
    epochs : int
        Maximum number of epochs for training
    patience : int
        Number of epochs with no improvement after which training will be stopped
        
    Returns:
    --------
    model : keras.models.Sequential
        Trained neural network model
    history : keras.callbacks.History
        Training history
    """
    # Create model
    model = create_neural_network(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Create directory for model checkpoints
    checkpoint_dir = 'IACRF/models/neural_network_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint to save the best model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
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

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning for neural network model
    """
    # Define hyperparameter grid
    hidden_layers_options = [
        [64, 32],
        [128, 64, 32],
        [64, 64, 32, 16]
    ]
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    
    best_val_loss = float('inf')
    best_params = {}
    best_model = None
    
    results = []
    
    # Grid search
    for hidden_layers in hidden_layers_options:
        for dropout_rate in dropout_rates:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    print(f"\nTraining with parameters: hidden_layers={hidden_layers}, "
                          f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                          f"batch_size={batch_size}")
                    
                    # Create and train model
                    model = create_neural_network(
                        input_dim=X_train.shape[1],
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # Early stopping
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    # Train model with fewer epochs for tuning
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=50,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate model
                    val_loss = min(history.history['val_loss'])
                    train_loss = min(history.history['loss'])
                    
                    results.append({
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'epochs_trained': len(history.history['loss'])
                    })
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        }
                        best_model = model
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    return best_model, best_params, results_df

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('IACRF/results'):
        os.makedirs('IACRF/results')
    
    # Create models directory if it doesn't exist
    if not os.path.exists('IACRF/models/neural_network'):
        os.makedirs('IACRF/models/neural_network')
    
    # Load data
    try:
        X, y, X_val, y_val = load_data()
    except FileNotFoundError:
        print("Error: Data files not found. Please check the file paths.")
        exit(1)
    
    # Scale features (neural networks work better with scaled features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_val_scaled = scaler.transform(X_val)
    
    print("Training Neural Network model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model with default parameters
    model, history = train_neural_network(
        X_train, y_train,
        X_test, y_test,  # Using test set as validation during training
        hidden_layers=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=20
    )
    
    # Plot training history
    plot_training_history(history)
    print("Training history plot saved to: IACRF/results/nn_training_history.png")
    
    # Wrap the model
    wrapped_model = KerasWrapper(model)
    
    # Evaluate model
    results = evaluate_model(wrapped_model, X_test, y_test, X_val_scaled, y_val)
    
    print("\nNeural Network Model Results:")
    print(f"Test R² Score: {results['test_r2']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    
    if 'val_r2' in results:
        print(f"Validation R² Score: {results['val_r2']:.4f}")
        print(f"Validation MAE: {results['val_mae']:.4f}")
        print(f"Validation RMSE: {results['val_rmse']:.4f}")
    
    # Save model
    model.save('IACRF/models/neural_network/model.keras')
    print("Model saved to: IACRF/models/neural_network/model.keras")
    
    # Hyperparameter tuning (limited to avoid excessive computation)
    print("\nPerforming hyperparameter tuning (this may take a while)...")
    
    # Use a smaller subset of data for hyperparameter tuning if the dataset is large
    if X_train.shape[0] > 1000:
        X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42)
        X_tune_val, _, y_tune_val, _ = train_test_split(X_test, y_test, train_size=200, random_state=42)
    else:
        X_tune, y_tune = X_train, y_train
        X_tune_val, y_tune_val = X_test, y_test
    
    # Simplified hyperparameter tuning for demonstration
    simplified_hidden_layers = [[32, 16], [64, 32]]
    simplified_dropout_rates = [0.1, 0.2]
    simplified_learning_rates = [0.01, 0.001]
    simplified_batch_sizes = [32]
    
    best_val_loss = float('inf')
    best_params = {}
    best_model = None
    
    results = []
    
    # Grid search
    for hidden_layers in simplified_hidden_layers:
        for dropout_rate in simplified_dropout_rates:
            for learning_rate in simplified_learning_rates:
                for batch_size in simplified_batch_sizes:
                    print(f"\nTraining with parameters: hidden_layers={hidden_layers}, "
                          f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
                          f"batch_size={batch_size}")
                    
                    # Create and train model
                    model = create_neural_network(
                        input_dim=X_tune.shape[1],
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # Early stopping
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    # Train model with fewer epochs for tuning
                    history = model.fit(
                        X_tune, y_tune,
                        validation_data=(X_tune_val, y_tune_val),
                        batch_size=batch_size,
                        epochs=50,
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    
                    # Evaluate model
                    val_loss = min(history.history['val_loss'])
                    train_loss = min(history.history['loss'])
                    
                    results.append({
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'epochs_trained': len(history.history['loss'])
                    })
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        }
                        best_model = model
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train final model with best parameters
    final_model, final_history = train_neural_network(
        X_train, y_train,
        X_test, y_test,
        hidden_layers=best_params['hidden_layers'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=150,
        patience=30
    )
    
    # Plot final training history
    plot_training_history(final_history)
    print("Final training history plot saved to: IACRF/results/nn_training_history.png")
    
    # Wrap the final model
    final_wrapped_model = KerasWrapper(final_model)
    
    # Evaluate final model
    final_results = evaluate_model(final_wrapped_model, X_test, y_test, X_val_scaled, y_val)
    
    print("\nFinal Neural Network Model Results:")
    print(f"Test R² Score: {final_results['test_r2']:.4f}")
    print(f"Test MAE: {final_results['test_mae']:.4f}")
    print(f"Test RMSE: {final_results['test_rmse']:.4f}")
    
    if 'val_r2' in final_results:
        print(f"Validation R² Score: {final_results['val_r2']:.4f}")
        print(f"Validation MAE: {final_results['val_mae']:.4f}")
        print(f"Validation RMSE: {final_results['val_rmse']:.4f}")
    
    # Save final model
    final_model.save('IACRF/models/neural_network_tuned/model.keras')
    print("Final model saved to: IACRF/models/neural_network_tuned/model.keras") 