# Logistic Regression Projects

This folder contains multiple implementations of Logistic Regression for binary and multiclass classification tasks.

---

## 1. Logistic_regression.py

- **Description:**  
  A manual implementation of binary logistic regression using gradient descent.  
  The model predicts if a student passes or fails based on exam scores.

- **Data:**  
  `project3data1.csv` containing exam scores and pass/fail labels.

- **Features:**  
  Two exam scores as input features, and a binary label (passed/not passed).

- **Highlights:**  
  - Data visualization with matplotlib scatter plots.  
  - Sigmoid function implementation for classification.  
  - Cost function and gradient descent implemented from scratch.  
  - Trains for 1,000,000 iterations with learning rate 0.004.  
  - Decision boundary plotted over the data.  
  - Identifies and prints misclassified samples with their probabilities.

---

## 2. Logistic_regression_multiclass_classification.py

- **Description:**  
  Manual implementation of one-vs-all logistic regression for multiclass digit classification (0 to 9).

- **Data:**  
  `project4data1.mat` containing handwritten digit data.

- **Highlights:**  
  - Uses regularized cost function and gradient descent.  
  - Implements one-vs-all strategy to train separate classifiers for each digit.  
  - Visualizes sample digits.  
  - Demonstrates vectorized vs loop prediction performance comparison.

---

## 3. Logistic_regression_multiclass_classification_With_Sklearn.py

- **Description:**  
  Multiclass logistic regression using scikit-learn’s `LogisticRegression`.

- **Data:**  
  Same `project4data1.mat` dataset loaded and split into train/test sets.

- **Highlights:**  
  - Uses sklearn for model training, prediction, and evaluation.  
  - Outputs accuracy and classification report.  
  - Suitable for practical applications requiring quick training.

---

## 4. Logistic_Regression_with_regularization.py

- **Description:**  
  Binary logistic regression with polynomial feature mapping and regularization to prevent overfitting.

- **Data:**  
  `project3data2.csv` with test scores and binary labels.

- **Highlights:**  
  - Maps features to polynomial terms up to degree 6.  
  - Implements regularized cost function and gradient descent.  
  - Visualizes decision boundary with contour plots.  
  - Trains with 300,000 iterations, learning rate 0.001, and regularization parameter 0.01.

---

## How to Run

1. Ensure the required data files (`project3data1.csv`, `project3data2.csv`, `project4data1.mat`) are in the folder.  
2. Run the scripts using Python:  
   ```bash
   python Logistic_regression.py
   python Logistic_regression_multiclass_classification.py
   python Logistic_regression_multiclass_classification_With_Sklearn.py
   python Logistic_Regression_with_regularization.py
