# 🧠 K-Means Clustering Projects

This folder contains two applications of the **K-Means clustering algorithm**:

1. **Unsupervised Clustering on a 2D Dataset**
2. **Image Compression using K-Means in RGB space**

---

## 📁 Files Overview

### 1. `kmeans_clustering.py`
Implements K-Means from scratch on a 2D dataset (`project8data1.mat`) and visualizes the results iteratively.

#### 🔹 Key Features:
- Loads dataset using `scipy.io.loadmat`.
- Initializes cluster centroids manually or randomly.
- Visualizes:
  - Data points
  - Cluster centroids
  - Cluster membership per iteration
- Final cluster assignments and centroids are returned.

#### 📌 Dataset:
- `project8data1.mat` (must contain a key `'X'` with shape `(m, 2)`)

#### 🛠️ Parameters:
- `K = 3`: Number of clusters.
- `max_iter = 16`: Number of iterations.

---

### 2. `image_compression_kmeans.py`
Compresses images by clustering RGB values using K-Means, significantly reducing the number of distinct colors.

#### 🔹 Key Features:
- Loads and normalizes an image.
- Flattens pixels to 3D vectors (R, G, B).
- Applies K-Means clustering to group similar colors.
- Replaces each pixel with its cluster centroid to compress the image.
- Shows compressed versions with varying `K` and `max_iter`.

#### 📌 Test Configurations:
```python
compression_levels = [
    {"K": 4, "max_iter": 2},
    {"K": 4, "max_iter": 1},
    {"K": 2, "max_iter": 1},
    {"K": 8, "max_iter": 1},
    {"K": 16, "max_iter": 1}
]
