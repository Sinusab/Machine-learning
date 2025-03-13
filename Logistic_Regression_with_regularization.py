import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
data = pd.read_csv('project3data2.csv')

# Displaying the first few rows of the data
data.head()

# Defining X (features) and Y (labels)
X = data.values[:, 0:2]  # Features (Test 1 and Test 2 scores)
Y = data.values[:, 2]    # Labels (Passed or Not Passed)
m = data.shape[0]        # Number of examples

print(X.shape, Y.shape, m)

# Function to plot the data
def plotData(X, y):
    pos = (y == 1)  # Positive examples (Passed)
    neg = (y == 0)  # Negative examples (Not Passed)
    plt.plot(X[pos, 0], X[pos, 1], 'r*', ms=10, label='Passed')  # Red stars for positive class
    plt.plot(X[neg, 0], X[neg, 1], 'bo', ms=8, label='Not Passed')  # Blue circles for negative class

# Plot the data points
plotData(X, Y)
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
plt.legend()

# Function to map the features to polynomial features
def mapFeature(X, degree=6):
    X1 = X[:, 0]  # First feature (Test 1 score)
    X2 = X[:, 1]  # Second feature (Test 2 score)
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]  # Add a column of ones (bias term)
    else:
        out = [np.ones(1)]
    
    # Generate polynomial features
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    
    if X1.ndim > 0:
        return np.stack(out, axis=1)  # Stack the features into a matrix
    else:
        return np.array(out)

# Map the features to higher-degree polynomial features
X_train = (mapFeature(X, 6)).T
Y_train = Y.reshape(1,-1)  # Reshape Y to be a row vector
print(X_train.shape , Y_train.shape)

# Sigmoid function definition
def sigmoid(x):
    g = 1 / (1 + np.exp(-x))  # Sigmoid function
    return g

# Cost function with regularization
def costFunctionwithreg(x, y, thetatemp, lambda_):
    m = x.shape[1]  # Number of examples
    h = sigmoid(np.matmul(thetatemp.T, x))  # Hypothesis
    cost = (-1 / m) * (np.matmul(np.log(h), y.T) + np.matmul(np.log(1 - h), (1 - y).T))  # Cost function
    reg = (lambda_ / (2 * m)) * np.sum(np.square(thetatemp[1:]))  # Regularization term (excluding theta_0)
    return cost + reg  # Total cost with regularization

# Initialize theta with zeros
theta = np.zeros((X_train.shape[0], 1))

# Calculate initial cost
initial_cost = costFunctionwithreg(X_train, Y_train, theta, lambda_=1)
print("Initial cost is:", initial_cost)

# Gradient Descent function for optimization
def gradienDescent(x, y, theta, lr, epochs, lambda_=1):
    m = x.shape[1]  # Number of examples
    J_history = []  # To store the cost history
    for epoch in range(epochs):
        h = sigmoid(np.matmul(theta.T, x))  # Compute hypothesis
        temp = h - y  # Compute error
        reg = np.concatenate([np.zeros((1, 1)), theta[1:]], axis=0)  # Regularization term (excluding theta_0)
        theta -= (lr / m) * (np.matmul(x, temp.T) + lambda_ * reg)  # Update theta with gradient descent
        J_history.append(costFunctionwithreg(x, y, theta, lambda_)[0, 0])  # Append the cost for this iteration
    return theta, J_history  # Return the optimized theta and cost history

# Initialize theta and set hyperparameters
theta = np.ones((X_train.shape[0], 1))
epochs = 300000  # Number of iterations
lr = 0.001  # Learning rate
lambda_ = 0.01  # Regularization parameter

# Perform gradient descent to optimize theta
theta_final, J_history = gradienDescent(X_train, Y_train, theta, lr, epochs, lambda_)

# Print the final cost after optimization
print("last cost: ", J_history[-1])

# Function to plot the decision boundary
def plotDecisionBoundary(plotData, theta, X, y):
    plotData(X, y)  # Plot the data points
    u = np.linspace(-1, 1.5, 50)  # Create a grid for plotting the decision boundary
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))  # Initialize the decision boundary matrix
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            z[i, j] = np.dot(mapFeature(np.array([[ui, vj]])), theta)  # Calculate the decision boundary for each point
    z = z.T  # Transpose the decision boundary matrix
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')  # Plot the decision boundary
    plt.legend()
    plt.xlabel('Test 1 Score')
    plt.ylabel('Test 2 Score')
    plt.title('Decision Boundary')

# Plot the decision boundary on the data
print("shape of X is: ", X_train.shape)
print("shape of Y is: ", Y_train.shape)

plt.figure(figsize=(8, 6))
plotDecisionBoundary(plotData, theta_final, X, Y)
plt.show()

# Prediction function to predict class labels and probabilities
def predict(thetatemp, X):
    prediction_prob = sigmoid(np.matmul(np.transpose(thetatemp), X))  # Calculate probabilities using the sigmoid function
    prediction_class = prediction_prob.copy()  # Copy the probabilities to classify the output
    prediction_class[prediction_class >= 0.5] = 1  # Classify as 1 (Passed) if probability >= 0.5
    prediction_class[prediction_class < 0.5] = 0   # Classify as 0 (Not Passed) if probability < 0.5

    return prediction_prob, prediction_class  # Return probabilities and class predictions

# Predict the probabilities and classes using the optimized theta
prediction_prob, prediction_class = predict(theta_final, X_train)

# Display predicted probabilities and class predictions
print(prediction_prob)
print(prediction_class)

# Display actual labels (for comparison)
print(Y_train[0,:])
