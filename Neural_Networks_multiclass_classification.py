import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load data
data = loadmat('project4data1.mat')
X = data['X']
Y = data['y']

# Display dataset shape
print(X.shape, Y.shape)

# Function to display an image
def showImage(X, idx):
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X[idx], (20, 20)).T, cmap='gray')
    plt.title(str(Y[idx]), fontsize=20)
    plt.show()

# Load pre-trained weights
weights = loadmat('project4-2-weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

# Display weights shape
print(theta1.shape, theta2.shape)

# Prepare input data
m = X.shape[0]
X_train = np.concatenate([np.ones((1, m)), X.T], axis=0)
Y_train = Y.T

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(theta1, theta2, X):
    z2 = np.matmul(theta1, X)
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((1)) , a2] , axis = 0)
    z3 = np.matmul(theta2, a2)
    a3 = sigmoid(z3)
    return a3

# Select an image index
idx = 4100
outputs = predict(theta1, theta2, X_train[:, idx])
print(outputs)

# Determine predicted digit
predicted_digit = np.argmax(outputs)
print('Predicted digit is:', predicted_digit)

# Show the image
showImage(X, idx)
