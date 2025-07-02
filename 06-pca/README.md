# 📉 PCA Dimensionality Reduction & Compression

This project demonstrates the use of **Principal Component Analysis (PCA)** for:
1. Reducing 2D data to 1D and reconstructing it.
2. Compressing grayscale face images (32×32 pixels) using PCA and visualizing the results.

---

## 📁 Files Overview

### `pca_dimensionality_reduction.py`
A script implementing PCA using `scikit-learn` on:
- A 2D dataset (`project8data1.mat`)
- A face dataset (`project8faces.mat`)

---

## 🧪 Part 1: PCA on 2D Data

- Reduces 2D data to 1D using PCA.
- Reconstructs the original 2D data from the reduced form.
- Plots both original and reconstructed points for visual comparison.

### 📌 Required file:
- `project8data1.mat` (must contain `'X'` key with shape `(m, 2)`)

---

## 🖼️ Part 2: PCA on Face Images

- Loads face images (32x32 = 1024 features per image).
- Normalizes and compresses each face using PCA to 324 components.
- Reconstructs the original face from compressed data.
- Displays:
  - First original face image
  - Compressed image reshaped as 18×18
  - Reconstructed face (from compressed data)

### 📌 Required file:
- `project8faces.mat` (must contain `'X'` with shape `(m, 1024)`)

---

## 📦 Dependencies

Make sure the following Python libraries are installed:

```bash
pip install numpy matplotlib scipy scikit-learn
