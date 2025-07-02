# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('project2-data2.csv')

X = data.values[: , 0 : 2]
Y = data.values[: , 2]
m = X.shape[0]

def normalizer(x):
    x_copy = x.copy()
    avg = np.mean(x_copy , axis = 0)
    range = np.max(x_copy , axis = 0) - np.min (x_copy , axis = 0)
    X_normal = (x_copy - avg) / range
    return X_normal

X_norm = normalizer(X)
X_norm2 = np.copy(X_norm)
X_norm = X_norm.T
X_norm = np.concatenate([np.ones((1 , m)) , X_norm] , axis = 0)
theta = np.zeros((X_norm.shape[0], 1))

def costfunction(X , Y , theta):
    m = X.shape[1]
    temp = np.matmul(theta.T, X) - Y.reshape(1, -1)
    cost = np.matmul(temp , temp.T) / (2 * m)
    return cost

def gradientDescent(X , Y , theta ,lr = 0.001 , epochs = 500):
    m = X.shape[1]
    for epoch in range(epochs) :
        temp = np.matmul(theta.T , X) - Y.reshape(1,-1)
        theta = theta - (lr / m) * np.matmul(X , temp.T)
    return theta

theta = gradientDescent(X_norm , Y , theta)

#### Test
a1_norm = (500 - np.mean(X[:, 0])) / (np.max(X[:, 0]) - np.min(X[:, 0]))
a2_norm = (5 - np.mean(X[:, 1])) / (np.max(X[:, 1]) - np.min(X[:, 1]))
price = np.matmul(theta.T, np.array([[1], [a1_norm], [a2_norm]]))
# price = theta[0] + theta[1] * a1_norm + theta[2] * a2_norm
print(f'The price prediction is: {price[0][0]:.2f}$')
