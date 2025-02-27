import pandas as pd
import numpy as np

def convert_training_data_to_csv():
    """
    Convert the training and testing data text file to CSV format
    """
    # Read the text file, skipping the header comments
    with open('IACRF/data/Training and testing data.txt', 'r') as f:
        lines = f.readlines()
    
    # Find the line that indicates the start of data
    data_start_line = 0
    for i, line in enumerate(lines):
        if '---Under is the data---' in line:
            data_start_line = i + 1
            break
    
    # Extract data lines (skip empty lines)
    data_lines = [line.strip() for line in lines[data_start_line:] if line.strip()]
    
    # Parse data lines
    data = []
    for line in data_lines:
        # Split by tabs and convert to float
        values = [float(val) if val.strip() and val.strip() != '/' else np.nan for val in line.split('\t')]
        data.append(values)
    
    # Create DataFrame
    column_names = ['Al', 'Zn', 'Mg', 'Cu', 'Si', 'Fe', 'Mn', 'Cr', 'Ti', 'Other', 
                    'Time', 'Temperature', 'Rainfall', 'pH', 'Cl', 'Corrosion_Rate']
    df = pd.DataFrame(data, columns=column_names)
    
    # Save to CSV
    df.to_csv('IACRF/data/raw_data/normal-wf.csv', index=False)
    print(f"Training data saved to IACRF/data/raw_data/normal-wf.csv")
    
    return df

def convert_validation_data_to_csv():
    """
    Convert the validation data text file to CSV format
    """
    # Read the text file, skipping the header comments
    with open('IACRF/data/Validation data from SEA.txt', 'r') as f:
        lines = f.readlines()
    
    # Find the line that indicates the start of data
    data_start_line = 0
    for i, line in enumerate(lines):
        if '---Under is the data---' in line:
            data_start_line = i + 1
            break
    
    # Extract data lines (skip empty lines)
    data_lines = [line.strip() for line in lines[data_start_line:] if line.strip()]
    
    # Parse data lines
    data = []
    for line in data_lines:
        # Split by tabs
        values = line.split('\t')
        
        # Convert to float, handling empty strings and '/'
        processed_values = []
        for val in values:
            if not val.strip() or val.strip() == '/':
                processed_values.append(np.nan)
            else:
                try:
                    processed_values.append(float(val.strip()))
                except ValueError:
                    processed_values.append(np.nan)
        
        # Add a placeholder for corrosion rate if it's missing
        if len(processed_values) == 15:  # No corrosion rate value
            processed_values.append(np.nan)
        
        data.append(processed_values)
    
    # Create DataFrame
    column_names = ['Al', 'Zn', 'Mg', 'Cu', 'Si', 'Fe', 'Mn', 'Cr', 'Ti', 'Other', 
                    'Time', 'Temperature', 'Rainfall', 'pH', 'Cl', 'Corrosion_Rate']
    df = pd.DataFrame(data, columns=column_names)
    
    # Save to CSV
    df.to_csv('IACRF/data/raw_data/SouthEastpredict02.csv', index=False)
    print(f"Validation data saved to IACRF/data/raw_data/SouthEastpredict02.csv")
    
    return df

if __name__ == "__main__":
    print("Converting text data files to CSV format...")
    train_df = convert_training_data_to_csv()
    val_df = convert_validation_data_to_csv()
    
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    
    # Print sample of training data
    print("\nSample of training data:")
    print(train_df.head())
    
    # Print sample of validation data
    print("\nSample of validation data:")
    print(val_df.head()) 