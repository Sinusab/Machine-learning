# 🧠 Bias-Variance Tradeoff & Polynomial Regression Analysis

This project explores **bias-variance tradeoff** and **regularized polynomial regression** using:
- Gradient descent optimization
- Learning and validation curves
- Polynomial feature engineering
- Regularization (L2)

---

## 📁 File: `BiasVarianceAnalysis.py`

This script trains and analyzes regression models on real-world data, using a variety of techniques to visualize bias, variance, and model performance.

---

## 🧪 What It Does

### ✅ Part 1: Linear Regression with Regularization
- Fits a simple linear model to a dataset using gradient descent.
- Uses L2 regularization to avoid overfitting.
- Plots the regression line over data.

### ✅ Part 2: Learning Curve
- Plots training and validation errors as a function of training set size.
- Helps detect **high bias** or **high variance** behavior.

### ✅ Part 3: Polynomial Regression
- Fits polynomial models of increasing degree.
- Normalizes features.
- Evaluates performance on validation set to find the **optimal degree**.

### ✅ Part 4: Validation Curve (for λ)
- Computes training and validation errors for different values of λ (regularization parameter).
- Helps identify the best regularization strength.

---

## 📌 Required Files

- `project6data1.mat`  
  A `.mat` file containing:
  - `'X'`, `'y'` – training data
  - `'Xval'`, `'yval'` – cross-validation data
  - `'Xtest'`, `'ytest'` – test data

---

## 📦 Dependencies

Install the following Python libraries:

```bash
pip install numpy matplotlib scipy
