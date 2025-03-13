import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('project3data1.csv')
data.head()

# Extract features (X) and labels (Y)
X = data.values[:, 0:2]
Y = data.values[:, 2]
m = X.shape[0]  # Number of training examples

# Plot the data
def plotData(x, y):
    pos = (y == 1)  # Passed
    neg = (y == 0)  # Not Passed
    plt.plot(x[pos, 0], x[pos, 1], 'r*', ms=10)
    plt.plot(x[neg, 0], x[neg, 1], 'bo', ms=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Passed', 'Not Passed'])

plotData(X, Y)
plt.show()

# Prepare the data for training
X_train = np.copy(X)
Y_train = Y.reshape(1, -1)
X_train = np.concatenate((np.ones((1, m)), X.T), axis=0)  # Add intercept term
theta = np.zeros((X_train.shape[0], 1))  # Initialize theta with zeros

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Cost function
def costFunction(x, y, theta):
    m = x.shape[1]
    h = sigmoid(np.matmul(theta.T, x))  # Hypothesis
    h = np.clip(h, 1e-10, 1 - 1e-10)  # Avoid log(0)
    cost = (-1 / m) * (np.matmul(np.log(h), y.T) + np.matmul(np.log(1 - h), (1 - y).T))
    return cost

# Compute initial cost
initiail_cost = costFunction(X_train, Y_train, theta)
print(f'Initial cost is {initiail_cost[0][0]:.4f}')

# Gradient descent function
def gradientDescent(x, y, theta, lr, epochs):
    m = x.shape[1]
    costs = []  # To track cost during iterations
    for epoch in range(epochs):
        cost = costFunction(x, y, theta)  # Compute cost
        costs.append(cost[0][0])  # Store cost
        h = sigmoid(np.matmul(theta.T, x))  # Compute hypothesis
        theta -= (lr / m) * np.matmul(x, (h - y).T)  # Update theta
    return theta, costs

# Train the model
import time
start_time = time.time()
final_theta, J_history = gradientDescent(X_train, Y_train, theta, 0.004, 1000000)
end_time = time.time()

# Execution time and final cost
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Final cost: {J_history[-1]:.4f}")
print(f"Final theta: {final_theta.flatten()}")

# Plot decision boundary
def plotDecisionBoundary(plotData, theta, X, y):
    theta = np.array(theta)  # Ensure theta is a numpy array
    plotData(X, y)  # Plot data points

    if X.shape[1] <= 3:  # For 2D data
        plot_x = np.array([np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2])  # x range
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])  # Compute y values
        plt.plot(plot_x, plot_y, 'g-', label='Decision Boundary')  # Plot boundary
        plt.xlim([np.min(X[:, 0]) - 10, np.max(X[:, 0]) + 10])
        plt.ylim([np.min(X[:, 1]) - 10, np.max(X[:, 1]) + 10])
        plt.legend(['Passed', 'Not Passed', 'Decision Boundary'])
    else:
        raise ValueError("Decision boundary visualization only supported for 2D data.")

X_new = X_train[1:, :].T  # Exclude intercept for visualization
Y_new = Y_train[0, :]  # Flatten Y for plotting

plotDecisionBoundary(plotData, final_theta, X_new, Y_new)

# Prediction function
def prediction(theta, x):
    x = x.reshape(1, -1)  # Reshape input
    x = np.concatenate((np.ones((1, 1)), x.T), axis=0)  # Add intercept
    prob = sigmoid(np.matmul(theta.T, x))  # Compute probability
    grade = True if prob >= 0.5 else False  # Classify
    return prob.item(), grade

# Identify wrong predictions
wrong_predictions = []
for i in range(m):
    x = X[i, :]
    prob, predicted_class = prediction(final_theta, x)
    if predicted_class != Y_train[0, i]:
        wrong_predictions.append(i)

# Print wrong predictions
for item in wrong_predictions:
    prob, predicted_class = prediction(final_theta, X[item, :])
    print(f'Prediction: {predicted_class} (Probability: {prob:.2f}), Actual: {Y_train[0, item]} - Student {item}')
