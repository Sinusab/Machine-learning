import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# Start timing for vectorized code execution
start_time_vec = time.time()

def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Function to display an image from the dataset
def showImage(X, Y, idx):
    """Display an image along with its corresponding class."""
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X[idx], (20, 20)).T, cmap='gray')
    plt.title(str(Y[idx]), fontsize=20)
    plt.show()

def nnCostGrad(X, Y, theta1, theta2, lambda_):
    """Compute cost and gradients using backpropagation."""
    m = X.shape[1]  # Number of samples
    
    # Forward propagation
    z2 = np.matmul(theta1, X)
    a2 = sigmoid(z2)
    a2 = np.vstack([np.ones((1, m)), a2])  # Add bias
    
    z3 = np.matmul(theta2, a2)
    a3 = sigmoid(z3)
    
    # Compute cost
    cost = (-1 / m) * np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3))
    cost += (lambda_ / (2 * m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    
    # Backpropagation of errors
    delta3 = a3 - Y
    delta2 = np.matmul(theta2.T, delta3) * (a2 * (1 - a2))
    delta2 = delta2[1:, :]
    
    theta1_grad = np.matmul(delta2, X.T) / m
    theta2_grad = np.matmul(delta3, a2.T) / m
    
    # Add regularization term (except for bias)
    theta1_grad[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    theta2_grad[:, 1:] += (lambda_ / m) * theta2[:, 1:]
    
    return cost, theta1_grad, theta2_grad

def train(X, Y, theta1, theta2, epochs, lr, lambda_):
    """Train the neural network using gradient descent."""
    J_history = []
    for epoch in range(epochs):
        cost, theta1_grad, theta2_grad = nnCostGrad(X, Y, theta1, theta2, lambda_)
        theta1 -= lr * theta1_grad
        theta2 -= lr * theta2_grad
        J_history.append(cost)
        print(f'Epoch {epoch + 1}, Cost: {cost:.4f}')
    return theta1, theta2, J_history

def predict(theta1, theta2, X):
    """Predict outputs for the input dataset."""
    z2 = np.matmul(theta1, X)
    a2 = sigmoid(z2)
    a2 = np.vstack([np.ones((1, X.shape[1])), a2])
    
    z3 = np.matmul(theta2, a2)
    a3 = sigmoid(z3)
    return np.argmax(a3, axis=0)

# Load data
data = loadmat('project4data1.mat')
X = data['X'].T  # Transpose for vectorized operations
Y = data['y'].flatten()

# Prepare data
m = X.shape[1]  # Number of samples
X_train = np.vstack([np.ones((1, m)), X])  # Add bias
Y_onehot = np.eye(10)[Y].T  # Convert output to one-hot encoding

# Initialize weights
in_layer_size = 400
hid_layer_size = 25
out_layer_size = 10
eps = 0.1  # Small value for initial weight values

# Initialize weight matrices
theta1 = np.random.rand(hid_layer_size, in_layer_size + 1) * (2 * eps) - eps
theta2 = np.random.rand(out_layer_size, hid_layer_size + 1) * (2 * eps) - eps

# Train the neural network
theta1, theta2, J_history = train(X_train, Y_onehot, theta1, theta2, epochs=1000, lr=0.9, lambda_=1)

# Prediction and accuracy calculation
preds = predict(theta1, theta2, X_train)
accuracy = np.mean(preds == Y) * 100
print(f'Training Accuracy: {accuracy:.2f}%')

# End timing and print execution time
end_time_vec = time.time()
vec_time = end_time_vec - start_time_vec
print(f'Execution time for vectorized implementation: {vec_time:.4f} seconds')
