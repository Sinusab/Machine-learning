# Neural Networks - Multiclass Classification

This folder contains multiple implementations of neural networks for classifying handwritten digits (0–9) using the MNIST-like dataset from `project4data1.mat`.

---

## 1. Neural_Networks_multiclass_classification.py

- **Description:**  
  Simple forward propagation using pre-trained weights for prediction.

- **Highlights:**  
  - Loads and visualizes sample images.
  - Applies feedforward pass using two-layer neural network (Theta1 and Theta2).
  - Predicts the digit for a selected image using `np.argmax`.

- **Input:**  
  - `project4data1.mat` – dataset  
  - `project4-2-weights.mat` – pre-trained weights

---

## 2. Neural_Networks_Multiclass_Classification_From_Scratch(Nonvectorized).py

- **Description:**  
  Full implementation of neural network training **without vectorization** (uses loops).

- **Highlights:**  
  - Manual forward and backward propagation (backpropagation)
  - One-hot encoding for labels
  - Gradient descent with regularization
  - Epoch-wise cost printing
  - Accuracy evaluation
  - Execution time measurement

- **Architecture:**  
  - Input: 400 features (20×20 pixels)  
  - Hidden: 25 neurons  
  - Output: 10 classes

- **Training Parameters:**  
  - Learning rate: 0.9  
  - Epochs: 1000  
  - Regularization: λ = 1

---

## 3. Neural_Networks_Multiclass_Classification_From_Scratch_(Vectorized).py

- **Description:**  
  Same as the previous script, but fully **vectorized** using NumPy operations.

- **Highlights:**  
  - Much faster due to vectorized matrix computations
  - Implements backpropagation efficiently
  - One-hot label encoding, cost calculation, training, prediction
  - Measures execution time

- **Expected Accuracy:** ~95–98% on training set

---

## 4. Neural_Networks_Multiclass_Classification_From_Scratch_(Scikit).py

- **Description:**  
  Implementation using `scikit-learn`'s `MLPClassifier`.

- **Highlights:**  
  - Uses built-in multilayer perceptron (MLP) with sigmoid activation
  - Trains on 80% of the data and evaluates on 20%
  - Plots the training loss curve
  - Visualizes prediction on a test sample

- **Model Config:**  
  - Hidden layer: 25 neurons  
  - Activation: logistic  
  - Optimizer: Adam  
  - Max iterations: 1000

---

## Dataset

- `project4data1.mat`:  
  - Contains 5000 grayscale images (20×20) of handwritten digits.  
  - Shape: `X` = (5000, 400), `y` = (5000, 1)

- `project4-2-weights.mat`:  
  - Contains pre-trained weights `Theta1`, `Theta2` used in the first script.

---

## How to Run

1. Ensure the following files are in the same folder:
   - `project4data1.mat`  
   - `project4-2-weights.mat` (for pre-trained model)
2. Run any of the scripts with Python:

```bash
python Neural_Networks_multiclass_classification.py
python Neural_Networks_Multiclass_Classification_From_Scratch(Nonvectorized).py
python Neural_Networks_Multiclass_Classification_From_Scratch_(Vectorized).py
python Neural_Networks_Multiclass_Classification_From_Scratch_(Scikit).py
