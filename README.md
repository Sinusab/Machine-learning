
# 🧠 Machine Learning Projects

This repository contains Python implementations of classic machine learning algorithms and applications. Each script demonstrates a specific concept such as regression, classification, clustering, or dimensionality reduction, using real-world or synthetic datasets.

---

## 📁 Projects Overview

### 1. 📉 Bias-Variance in Polynomial Regression
Analyze bias and variance trade-off using polynomial regression on validation and training data.

- **Script:** `polynomial_bias_variance.py`
- **Dataset:** `project6data1.mat`

---

### 2. 📌 Support Vector Machines (SVM)
Train and visualize linear and RBF-kernel SVMs on various datasets, including a spam classifier.

- **Scripts:**
  - `SVM.py`
  - `SVM_spam_classification.py`
- **Datasets:**
  - `project7data1.mat`, `project7data2.mat`, `project7data3.mat`
  - `spamTrain.mat`, `spamTest.mat`, `vocab.txt`

---

### 3. 🎨 K-Means Clustering
Apply K-Means clustering to synthetic 2D data and RGB image compression.

- **Scripts:**
  - `kmeans_clustering.py`
  - `image_compression_kmeans.py`
- **Datasets:**
  - `project8data1.mat`
  - `image.jpg` (or any RGB image)

---

### 4. 🧬 Principal Component Analysis (PCA)
Reduce dimensionality of both 2D data and face images using PCA and visualize the reconstruction.

- **Script:** `pca_dimensionality_reduction.py`
- **Datasets:**
  - `project8data1.mat`
  - `project8faces.mat`

---

## 📦 Dataset Archive

All required datasets are included in a single compressed archive:

**➡ [`all_projects_data.rar`](datasets/all_projects_data.rar)**

### 🔧 How to use:
1. Download and extract the `.rar` file from the `datasets/` directory.
2. Place the extracted files in the root directory of this repository or alongside each Python script.
3. Make sure the filenames remain unchanged, as they are referenced directly in the code.

> ⚠️ If you move or rename the files, remember to update the file paths in the scripts.

---

## 🧰 Requirements

Install required Python libraries with:

```bash
pip install numpy matplotlib scipy scikit-learn pandas nltk


💬 Contact
For any questions or feedback, feel free to open an issue or reach out directly.

=======
# Machine Learning Algorithms

This repository contains implementations of various machine learning algorithms and analyses.

## Projects

### Bias-Variance Analysis for Polynomial Regression
A Python script that analyzes bias and variance in a polynomial regression model using learning and validation curves. The script trains a polynomial regression model on the `project6data1.mat` dataset, evaluates the effect of polynomial degree and regularization parameter (lambda), and visualizes the results.

#### Dataset
The script uses the dataset `project6data1.mat`, which is included in this repository.

#### Requirements
To run the script, you need the following Python libraries:
- `numpy`
- `matplotlib`
- `scipy`

Install them using pip:
```bash
pip install numpy matplotlib scipy
>>>>>>> 551b6e2e54a272b79f7624f9ae7fcdbd7dd9dc63
