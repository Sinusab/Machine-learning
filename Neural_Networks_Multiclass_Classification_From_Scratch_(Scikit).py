import numpy as np
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# ============================================================================
# Data Loading and Exploration
# ============================================================================

# Load the dataset from .mat file
data = loadmat('project4data1.mat')
X = data['X']  # Feature matrix: 5000 samples, 400 features (20x20 pixels)
y = data['y'].flatten()  # Label vector: 5000 samples, flattened to 1D

# Print dataset shapes for verification
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

# Check min and max values of X to understand data range
print(f'Min value in X: {X.min()}, Max value in X: {X.max()}')

# ============================================================================
# Visualization Function
# ============================================================================

def show_image(X, y, idx):
    """
    Display a single image from the dataset with its label.
    
    Parameters:
    - X: Feature matrix (samples x features)
    - y: Label vector
    - idx: Index of the sample to display
    """
    plt.figure(figsize=(2, 2))
    plt.imshow(X[idx].reshape(20, 20).T, cmap='gray')
    plt.title(f'Label: {y[idx]}')
    plt.show()

# Visualize a sample image (e.g., index 2600)
show_image(X, y, 2600)

# ============================================================================
# Data Splitting
# ============================================================================

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# Model Definition and Training
# ============================================================================

# Initialize the MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(25,),        # Single hidden layer with 25 neurons
    activation='logistic',           # Sigmoid activation function
    solver='adam',                   # Adaptive moment estimation optimizer
    learning_rate_init=0.001,        # Initial learning rate
    max_iter=1000,                   # Maximum number of iterations
    random_state=42                  # Seed for reproducibility
)

# Train the model on the training data with timing
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f'Training Time: {training_time:.2f} seconds')

# ============================================================================
# Model Evaluation
# ============================================================================

# Calculate accuracy on training and test sets
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

# Print evaluation results
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')

# ============================================================================
# Prediction and Visualization
# ============================================================================

# Predict and visualize a test sample
idx = 10  # Index of the test sample to predict
sample = X_test[idx].reshape(1, -1)  # Reshape for single-sample prediction
pred = model.predict(sample)[0]       # Get predicted label

# Show the test sample with its predicted and actual labels
show_image(X_test, y_test, idx)
print(f'Predicted: {pred}, Actual: {y_test[idx]}')

# ============================================================================
# Training Loss Curve
# ============================================================================

# Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
