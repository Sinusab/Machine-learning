from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time

# بارگذاری داده‌ها
data = loadmat('project4data1.mat')
X = data['X']
Y = data['y']
m = Y.shape[0]

# آماده‌سازی داده‌ها
X_train = X.T
X_train = np.concatenate([np.ones((1, m)), X_train], axis=0)
Y_train = Y.T

# نمایش تصویر
def showImage(X, idx):
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(X[idx], (20, 20)).T, cmap='gray')
    plt.title(str(Y[idx]), fontsize=20)

# تابع سیگموید
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# محاسبه هزینه با تنظیم
def computeCostWithReg(X, Y, theta, lambda_):
    m = X.shape[1]
    h = sigmoid(np.matmul(theta.T, X))
    cost = (-1 / m) * (np.matmul(np.log(h), Y.T) + np.matmul(np.log(1 - h), (1 - Y).T))
    costWithReg = cost + (lambda_ / (2 * m)) * np.matmul(theta[1:].T, theta[1:])
    return costWithReg

# گرادیان نزولی با تنظیم
def gradientDescentWithReg(X, Y, theta, lr, epochs, lambda_):
    m = X.shape[1]
    J_history = []
    for _ in range(epochs):
        theta_reg = theta.copy()
        theta_reg[0] = 0
        h = sigmoid(np.matmul(theta.T, X))
        temp = (h - Y).T
        theta -= (lr / m) * (np.matmul(X, temp) + (lambda_ * theta_reg))
        J_history.append(computeCostWithReg(X, Y, theta, lambda_)[0, 0])
    return theta, J_history

# پیاده‌سازی One-vs-All
def one_vs_all(X, Y, num_classes, lr, num_epochs, lambda_):
    n, m = X.shape
    alltheta = np.zeros((n, num_classes))
    for i in range(num_classes):
        Y_temp = np.zeros(Y.shape)
        Y_temp[Y == i] = 1
        initial_theta = np.zeros((X.shape[0], 1))
        final_theta, _ = gradientDescentWithReg(X, Y_temp, initial_theta, lr, num_epochs, lambda_)
        alltheta[:, i] = final_theta[:, 0]
    return alltheta

# تابع پیش‌بینی
def predict_one_vs_all(alltheta, test_image):
    all_preds = sigmoid(np.matmul(alltheta.T, test_image))
    return np.argmax(all_preds.flatten())

# اجرای مدل
lr = 0.01
epochs = 50
lambda_ = 0.1
alltheta = one_vs_all(X_train, Y_train, 10, lr, epochs, lambda_)

# پیش‌بینی یک نمونه
idx = 3673
digit = predict_one_vs_all(alltheta, X_train[:, idx])
print(f'The predicted digit is: {digit}')
showImage(X, idx)

# مقایسه روش حلقه‌ای و برداری
start_time = time.time()
for _ in range(1000):
    preds = []
    for i in range(alltheta.shape[1]):
        preds.append(sigmoid(np.matmul(alltheta[:, i].T, X_train[:, 0])))
    np.argmax(preds)
print("Loop method:", time.time() - start_time)

start_time = time.time()
for _ in range(1000):
    np.argmax(sigmoid(np.matmul(alltheta.T, X_train[:, 0])))
print("Vectorized method:", time.time() - start_time)
