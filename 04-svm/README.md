# Support Vector Machines (SVM)

This folder contains various implementations of Support Vector Machines (SVMs) applied to both synthetic datasets and real-world spam classification.

---

## 📁 Files Overview

### 1. `SVM.py`
Demonstrates SVM classification using both linear and non-linear kernels on synthetic datasets from `.mat` files.

#### 🔹 Part 1 – Linear SVM (`project7data1.mat`)
- Binary classification using a linear kernel.
- Visualizes the decision boundary.
- Plots positive vs negative samples.
- Uses `SVC(kernel='linear')`.

#### 🔹 Part 2 – RBF (Gaussian) SVM (`project7data2.mat`)
- Non-linear classification using RBF kernel.
- Scales features using `StandardScaler`.
- Visualizes decision surface using contour plots.
- Uses `SVC(kernel='rbf')` with manual gamma and C.
- Also uses `GridSearchCV` for hyperparameter tuning.

#### 🔹 Part 3 – RBF SVM with Validation (`project7data3.mat`)
- Splits data into training and validation sets.
- Manually tunes hyperparameters (C and γ) and prints accuracy.
- Compares manual tuning with `GridSearchCV`.
- Visualizes best decision boundary.

---

### 2. `SVM_spam_classification.py`
Trains a **spam email classifier** using a linear SVM and bag-of-words features.

#### 📌 Workflow:

1. **Vocabulary Loading**
   - Loads a vocabulary text file and maps words to indices.

2. **Email Preprocessing**
   - Lowercasing, removing URLs, emails, numbers, and special characters.

3. **Stemming**
   - Applies stemming using NLTK's `PorterStemmer`.

4. **Feature Extraction**
   - Converts preprocessed email into a 1899-dimensional binary feature vector.

5. **Model Training**
   - Loads `spamTrain.mat` and `spamTest.mat`.
   - Trains a linear SVM on the training data.
   - Evaluates accuracy on both training and test sets.

6. **Feature Weight Analysis**
   - Extracts SVM weights and shows the top 10 most influential words in spam classification.

---

## 📦 Dataset Files Required

- `project7data1.mat`
- `project7data2.mat`
- `project7data3.mat`
- `spamTrain.mat`
- `spamTest.mat`
- `vocab.txt`

---

## 🧪 How to Run

Run SVM visualizations:
```bash
python SVM.py
