import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# Start timing for non-vectorized code execution
start_time_nonvec = time.time()

# Function to visualize a sample image
def showImage(X, idx):
    """Display an image from the dataset."""
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X[idx], (20, 20)).T, cmap='gray')
    plt.title(str(Y[idx]), fontsize=20)
    plt.show()

# Sigmoid activation function
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Compute cost and gradients using non-vectorized operations
def nnCostGrad(X_train, Y_onehot, theta1, theta2, lambda_):
    """Compute cost and gradients using backpropagation without vectorization."""
    m = X_train.shape[1]
    sumCost_m = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    
    for i in range(m):
        # Forward propagation
        a1 = X_train[:, i]
        z2 = np.matmul(theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.concatenate([np.ones((1,)), a2], axis=0)
        z3 = np.matmul(theta2, a2)
        a3 = sigmoid(z3)
        
        # Compute cost
        sumCost_m += np.sum(Y_onehot[:, i] * np.log(a3) + (1 - Y_onehot[:, i]) * np.log(1 - a3))
        
        # Backpropagation
        delta3 = a3 - Y_onehot[:, i]
        delta2 = np.matmul(theta2.T, delta3) * (a2 * (1 - a2))
        delta2 = delta2[1:]  # Remove bias term
        
        theta1_grad += np.outer(delta2, a1)
        theta2_grad += np.outer(delta3, a2)
    
    cost = (-1 / m) * sumCost_m + (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    theta1_grad[:, 1:] += lambda_ * theta1[:, 1:]
    theta2_grad[:, 1:] += lambda_ * theta2[:, 1:]
    
    return cost, theta1_grad / m, theta2_grad / m

# Training function
def train(X_train, Y_onehot, nnCostGrad, theta1, theta2, epochs, lr, lambda_):
    """Train the neural network using gradient descent."""
    J_history = []
    for epoch in range(epochs):
        cost, theta1_grad, theta2_grad = nnCostGrad(X_train, Y_onehot, theta1, theta2, lambda_)
        theta1 -= lr * theta1_grad
        theta2 -= lr * theta2_grad
        J_history.append(cost)
        print(f'Cost at epoch {epoch}: {cost}')
    return theta1, theta2, J_history

# Prediction function
def predict(theta1, theta2, X_train):
    """Predict outputs for the input dataset."""
    z2 = np.matmul(theta1, X_train)
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((1, X_train.shape[1])), a2], axis=0)
    z3 = np.matmul(theta2, a2)
    a3 = sigmoid(z3)
    return a3

# Load dataset and prepare training data
data = loadmat('project4data1.mat')
X = data['X']
Y = data['y']
m = X.shape[0]
print(f'Shape of X: {X.shape}\nShape of Y: {Y.shape}\nSize of m: {m}')

# Show an example image
showImage(X, 2600)

# Prepare training data
X_train = np.concatenate([np.ones((1, m)), X.T], axis=0)  # Add bias term
Y_train = Y.T
Y_onehot = np.eye(10)[Y_train].T
Y_onehot = Y_onehot[:, :, 0]
print(f'X_train shape: {X_train.shape}\nY_onehot shape: {Y_onehot.shape}')

# Initialize neural network parameters
in_layer_size = 400  # Input layer neurons (20x20 pixels)
hid_layer_size = 25   # Hidden layer neurons
out_layer_size = 10   # Output layer neurons (10 classes)
eps = 0.1  # Small value for initial weight values

# Random initialization of weights
theta1 = np.random.rand(hid_layer_size, in_layer_size + 1) * (2 * eps) - eps
theta2 = np.random.rand(out_layer_size, hid_layer_size + 1) * (2 * eps) - eps
print(f'theta1 shape: {theta1.shape}\ntheta2 shape: {theta2.shape}')

# Train neural network
theta1new, theta2new, J_history = train(X_train, Y_onehot, nnCostGrad, theta1, theta2, epochs=1000, lr=0.9, lambda_=1)

# Evaluate model
outputs = predict(theta1new, theta2new, X_train)
predicted_digit = np.argmax(outputs, axis=0)
accuracy = np.mean(predicted_digit.flatten() == Y.flatten()) * 100
print(f'Training Set Accuracy: {accuracy:.2f}%')

# Test on a specific sample
idx = 2600
showImage(X, idx)
predicted_label = np.argmax(outputs[:, idx])
actual_label = Y[idx, 0]
print(f'Actual Label: {actual_label}')
print(f'Predicted Label: {predicted_label}')

# End timing and print execution time
end_time_nonvec = time.time()
nonvec_time = end_time_nonvec - start_time_nonvec
print(f'Execution time for non-vectorized implementation: {nonvec_time:.4f} seconds')
