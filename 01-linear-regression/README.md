# Linear Regression Projects

This folder contains two implementations of Linear Regression for price prediction based on two features.

## 1. Linear_regression.py

- Manual implementation of Linear Regression using Gradient Descent.
- Loads data from `project2-data2.csv`.
- Normalizes the features manually.
- Trains the model with a learning rate of 0.001 and 500 epochs.
- Predicts the price for input `[500, 5]`.

## 2. Linear_Regression(scikit).py

- Implementation of Linear Regression using scikit-learn's `SGDRegressor`.
- Loads data from `project2-data2.csv`.
- Standardizes the features using `StandardScaler`.
- Trains the model with squared error loss, learning rate 0.001, and 5000 iterations.
- Predicts the price for input `[1600, 3]`.

## How to Run

1. Make sure `project2-data2.csv` is in this folder.
2. Run either script using Python:

```bash
python Linear_regression.py
python Linear_Regression(scikit).py
