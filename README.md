# Corrosion Rate Prediction Models

This repository contains various machine learning models for predicting corrosion rates based on material composition and environmental factors. The original implementation used Random Forest, and this extension explores additional models to compare performance.

## Models Implemented

1. **Random Forest** (original implementation)
2. **Gradient Boosting**
3. **Support Vector Regression (SVR)**
4. **Neural Network** (using TensorFlow)

## Project Structure

- `main.py`: Original Random Forest implementation
- `utils.py`: Common utility functions for data loading, preprocessing, and evaluation
- `gradient_boosting_model.py`: Gradient Boosting implementation
- `svr_model.py`: Support Vector Regression implementation
- `xgboost_model.py`: XGBoost implementation
- `neural_network_model.py`: Neural Network implementation using TensorFlow
- `model_comparison.py`: Script to compare all models and visualize results
- `requirements.txt`: Dependencies required to run the code
- `models/`: Directory to store trained models
- `results/`: Directory to store evaluation results and visualizations

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Individual Models

To train and evaluate a specific model, run the corresponding script:

```bash
# Gradient Boosting
python gradient_boosting_model.py

# Support Vector Regression
python svr_model.py

# XGBoost
python xgboost_model.py

# Neural Network
python neural_network_model.py
```

### Comparing All Models

To train and compare all models:

```bash
python model_comparison.py
```

This will:
1. Train all implemented models
2. Evaluate them on test and validation data
3. Perform cross-validation
4. Generate comparison plots and tables
5. Save the best performing model

## Data

The models are trained on corrosion data with the following features:
- Material composition (Al, Zn, Mg, Cu, Si, Fe, Mn, Cr, Ti, etc.)
- Environmental factors (Time, Temperature, Rainfall, pH, Cl)
- Target variable: Corrosion rate

## Results

The comparison results are saved in:
- `results/model_comparison.png`: Bar charts comparing model performance metrics
- `results/cross_validation_comparison.png`: Cross-validation results
- `results/model_comparison_results.csv`: Detailed results table

## References

This work extends the original paper:
- Authors: Yucheng Ji, Ni Li, Zhanming Cheng, Xiaoqian Fu, Min Ao, Menglin Li, Xiaoguang Sun, Thee Chowwanonthapunya, Dawei Zhang, Kui Xiao, Jingli Ren, Poulumi Dey, Xiaoguang Li, Chaofang Dong
- Corresponding author: Prof. Chaofang Dong (cfdong@ustb.edu.cn)
