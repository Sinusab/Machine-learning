import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the dataset from the .mat file and print available keys
data = loadmat('project6data1.mat')
print(data.keys())

# Extract training, validation, and test data
X = data['X']
X_val = data['Xval']
X_test = data['Xtest']
Y = data['y']
Y_val = data['yval']
Y_test = data['ytest']
m = X.shape[0]       # Number of training examples
m_val = X_val.shape[0]  # Number of validation examples
m_test = X_test.shape[0]  # Number of test examples

# Plot the training data points
plt.plot(X, Y, 'ro', ms=10, mec='k')

# Transpose data for matrix operations and add bias term (column of ones)
X_ = X.T
X_val_ = X_val.T
X_test_ = X_test.T
Y_ = Y.T
Y_val_ = Y_val.T
Y_test_ = Y_test.T
X_ = np.concatenate([np.ones((1, m)), X_], axis=0)  # Add bias term to training data
X_val_ = np.concatenate([np.ones((1, m_val)), X_val_], axis=0)  # Add bias term to validation data
X_test_ = np.concatenate([np.ones((1, m_test)), X_test_], axis=0)  # Add bias term to test data

def computeCostWithReg(X_, Y_, thetatemp, lambda_):
    """
    Compute the regularized cost for linear regression.
    
    Parameters:
    X_ (numpy array): Feature matrix (with bias term)
    Y_ (numpy array): Target values
    thetatemp (numpy array): Model parameters (theta)
    lambda_ (float): Regularization parameter
    
    Returns:
    float: Regularized cost value
    """
    m = X_.shape[1]  # Number of training examples
    temp = np.matmul(thetatemp.T, X_) - Y_  # Compute predictions and errors
    cost = (1/(2*m)) * np.matmul(temp, temp.T)  # Mean squared error
    reg = (lambda_ / (2*m)) * np.matmul(thetatemp[1:].T, thetatemp[1:])  # Regularization term (excluding bias)
    costWithReg = cost + reg  # Total cost
    return costWithReg.item()

def gradientDescentWithReg(X_, Y_, thetatemp, lr, epochs, lambda_):
    """
    Perform gradient descent with regularization to optimize model parameters.
    
    Parameters:
    X_ (numpy array): Feature matrix (with bias term)
    Y_ (numpy array): Target values
    thetatemp (numpy array): Initial model parameters (theta)
    lr (float): Learning rate
    epochs (int): Number of iterations
    lambda_ (float): Regularization parameter
    
    Returns:
    tuple: Optimized theta, cost history, and initial gradient
    """
    m = X_.shape[1]  # Number of training examples
    J_history = []  # To store cost history
    grad = np.zeros_like(thetatemp)  # To store initial gradient
    for epoch in range(epochs):
        thetatempreg = thetatemp.copy()
        thetatempreg[0,0] = 0  # Exclude bias term from regularization
        h = np.matmul(thetatemp.T, X_)  # Compute predictions
        temp = (h - Y_).T  # Compute errors
        thetatemp = thetatemp - (lr/m) * (np.matmul(X_, temp) + lambda_ * thetatempreg)  # Update theta
        if (epoch == 0):
            grad = (1/m) * (np.matmul(X_, temp) + lambda_ * thetatempreg)  # Compute initial gradient
        J_history.append(computeCostWithReg(X_, Y_, thetatemp, lambda_))  # Append cost to history
    return thetatemp, J_history, grad

# Train a linear regression model with regularization
theta = np.random.randn(X_.shape[0], 1) * 0.01  # Initialize theta randomly
epochs = 100000  # Number of iterations
lr = 0.001  # Learning rate
thetanew, J_history, gradtest = gradientDescentWithReg(X_, Y_, theta, lr, epochs, lambda_=1)
print(f'Gradtest is: {gradtest}')
print(f'Last cost is: {J_history[-1]}')

# Plot the fitted line over the training data
plt.plot(X_[1:, :].T, Y_.T, 'ro', ms=10, mec='k', mew=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X_[1:, :].T, np.matmul(thetanew.T, X_).T, '--', lw=2)
plt.show()

def learningCurve(X_, Y_, X_val_, Y_val_, gradientDescentWithReg, computeCostWithReg, lambda_):
    """
    Compute training and validation errors for varying training set sizes (Learning Curve).
    
    Parameters:
    X_ (numpy array): Training feature matrix
    Y_ (numpy array): Training target values
    X_val_ (numpy array): Validation feature matrix
    Y_val_ (numpy array): Validation target values
    gradientDescentWithReg (function): Gradient descent function
    computeCostWithReg (function): Cost computation function
    lambda_ (float): Regularization parameter
    
    Returns:
    tuple: Training and validation errors for each training set size
    """
    m = X_.shape[1]  # Number of training examples
    train_error = np.zeros(m)  # To store training errors
    val_error = np.zeros(m)  # To store validation errors
    for i in range(1, m+1):
        theta = np.random.randn(X_.shape[0], 1) * 0.01  # Initialize theta
        epochs = 100000  # Number of iterations
        lr = 0.001  # Learning rate
        thetanew, J_history, gradtest = gradientDescentWithReg(X_[:, :i], Y_[:, :i], theta, lr, epochs, lambda_=1)
        train_error[i-1] = computeCostWithReg(X_[:, :i], Y_[:, :i], thetanew, lambda_=0)  # Compute training error
        val_error[i-1] = computeCostWithReg(X_val_, Y_val_, thetanew, lambda_=0)  # Compute validation error
    return train_error, val_error

# Compute and plot the learning curve
train_error, val_error = learningCurve(X_, Y_, X_val_, Y_val_, gradientDescentWithReg, computeCostWithReg, lambda_=1)
plt.figure()
plt.plot(range(1, m+1), train_error, label='Training Error')
plt.plot(range(1, m+1), val_error, label='Validation Error')
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

# Print the learning curve errors in a table format
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, train_error[i], val_error[i]))

plt.show()

def polyFeatures(X, d):
    """
    Generate polynomial features for the input data.
    
    Parameters:
    X (numpy array): Input data
    d (int): Degree of the polynomial
    
    Returns:
    numpy array: Polynomial features up to degree d
    """
    X_poly = np.zeros((X.shape[0], d))
    for i in range(d):
        X_poly[:, i] = X[:, 0] ** (i + 1)  # Compute X^i for each degree
    return X_poly

def featureNormalize(X, method="range"):
    """
    Normalize features using the specified method.
    
    Parameters:
    X (numpy array): Input data
    method (str): Normalization method ('range' or 'std')
    
    Returns:
    tuple: Normalized data, mean, and range/std
    """
    X_norm = X.copy()
    if method == "range":
        mean = np.mean(X_norm, axis=0)  # Compute mean
        range1 = np.max(X_norm, axis=0) - np.min(X_norm, axis=0)  # Compute range
        range1[range1 == 0] = 1  # Prevent division by zero
        X_norm = (X_norm - mean) / range1  # Normalize using range
        return X_norm, mean, range1
    elif method == "std":
        mean = np.mean(X_norm, axis=0)  # Compute mean
        std = np.std(X_norm, axis=0)  # Compute standard deviation
        std[std == 0] = 1  # Prevent division by zero
        X_norm = (X_norm - mean) / std  # Normalize using std
        return X_norm, mean, std

def optimumDegree(X, Y_, X_val, Y_val_, X_test, Y_test_, degree, polyFeatures, featureNormalize, costfunction, gradientDescentWithReg):
    """
    Find the optimal polynomial degree by evaluating validation error.
    
    Parameters:
    X (numpy array): Training data
    Y_ (numpy array): Training target values
    X_val (numpy array): Validation data
    Y_val_ (numpy array): Validation target values
    X_test (numpy array): Test data
    Y_test_ (numpy array): Test target values
    degree (int): Maximum degree to evaluate
    polyFeatures (function): Function to generate polynomial features
    featureNormalize (function): Function to normalize features
    costfunction (function): Function to compute cost
    gradientDescentWithReg (function): Gradient descent function
    
    Returns:
    tuple: Best degree, best validation error, list of thetas, cost histories, and gradients
    """
    thetanew = []  # To store optimized thetas for each degree
    J_history = []  # To store cost histories for each degree
    gradtest = []  # To store initial gradients for each degree
    m = X.shape[0]  # Number of training examples
    m_val = X_val.shape[0]  # Number of validation examples
    m_test = X_test.shape[0]  # Number of test examples
    best_degree = 1  # Initialize best degree
    best_error_val = float('inf')  # Initialize best validation error
    for i in range(1, degree + 1):
        X_poly = polyFeatures(X, i)  # Generate polynomial features for training data
        X_poly_val = polyFeatures(X_val, i)  # Generate polynomial features for validation data
        X_poly_test = polyFeatures(X_test, i)  # Generate polynomial features for test data
        
        # Normalize features using the training data's statistics
        X_poly_norm, mean, range1 = featureNormalize(X_poly, method='range')
        X_poly_normv = (X_poly_val - mean) / range1  # Normalize validation data
        X_poly_normt = (X_poly_test - mean) / range1  # Normalize test data
        
        # Transpose and add bias term
        X_poly_norm_ = X_poly_norm.T
        X_poly_normv_ = X_poly_normv.T
        X_poly_normt_ = X_poly_normt.T
        X_poly_norm_ = np.concatenate([np.ones((1, m)), X_poly_norm_], axis=0)
        X_poly_normv_ = np.concatenate([np.ones((1, m_val)), X_poly_normv_], axis=0)
        X_poly_normt_ = np.concatenate([np.ones((1, m_test)), X_poly_normt_], axis=0)

        theta = np.random.randn(X_poly_norm_.shape[0], 1) * 0.01  # Initialize theta
        epochs = 100000  # Number of iterations
        lr = 0.003  # Learning rate
        thetanew_temp, J_history_temp, gradtest_temp = gradientDescentWithReg(X_poly_norm_, Y_, theta, lr, epochs, lambda_=0)
        thetanew.append(thetanew_temp)
        J_history.append(J_history_temp)
        gradtest.append(gradtest_temp)
        
        val_error = costfunction(X_poly_normv_, Y_val_, thetanew_temp, lambda_=0)  # Compute validation error
        print(f"Degree: {i}, Validation Error: {val_error}")
        if val_error < best_error_val:
            best_error_val = val_error  # Update best validation error
            best_degree = i  # Update best degree
        
    return best_degree, best_error_val, thetanew, J_history, gradtest

# Find the optimal polynomial degree
best_degree, best_error_val, thetanew, J_history, gradtest = optimumDegree(
    X, Y_, X_val, Y_val_, X_test, Y_test_,
    degree=8, polyFeatures=polyFeatures, featureNormalize=featureNormalize,
    costfunction=computeCostWithReg, gradientDescentWithReg=gradientDescentWithReg
)
print(f'Best degree is: {best_degree}')
print(f'Validation Error for best degree: {best_error_val}')

# Generate polynomial features using the best degree
X_poly = polyFeatures(X, best_degree)
X_poly_ = X_poly.T
X_val_poly = polyFeatures(X_val, best_degree)
X_val_poly_ = X_val_poly.T
X_test_poly = polyFeatures(X_test, best_degree)
X_test_poly_ = X_test_poly.T

# Normalize features using the training data's statistics
X_poly_norm, x_mean, range1 = featureNormalize(X_poly, method='range')
X_poly_norm_ = X_poly_norm.T
X_poly_ = np.concatenate([np.ones((1, m)), X_poly_norm_], axis=0)
X_val_poly_norm = (X_val_poly - x_mean) / range1
X_val_poly_norm_ = X_val_poly_norm.T
X_val_poly_ = np.concatenate([np.ones((1, m_val)), X_val_poly_norm_], axis=0)
X_test_poly_norm = (X_test_poly - x_mean) / range1
X_test_poly_norm_ = X_test_poly_norm.T
X_test_poly_ = np.concatenate([np.ones((1, m_test)), X_test_poly_norm_], axis=0)

def valCurve(X_poly_, Y_, X_val_poly_, Y_val_, computeCostWithReg, gradientDescentWithReg):
    """
    Compute training and validation errors for different values of lambda (Validation Curve).
    
    Parameters:
    X_poly_ (numpy array): Polynomial features for training data
    Y_ (numpy array): Training target values
    X_val_poly_ (numpy array): Polynomial features for validation data
    Y_val_ (numpy array): Validation target values
    computeCostWithReg (function): Cost computation function
    gradientDescentWithReg (function): Gradient descent function
    
    Returns:
    tuple: Training and validation errors for each lambda value
    """
    lambda_vec = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]  # Lambda values to evaluate
    error_train = np.zeros(len(lambda_vec))  # To store training errors
    error_val = np.zeros(len(lambda_vec))  # To store validation errors

    for i in range(len(lambda_vec)):
        theta1 = np.random.randn(X_poly_.shape[0], 1) * 0.01  # Initialize theta
        epochs = 100000  # Number of iterations
        lr = 0.003  # Learning rate
        thetanew1, J_history, gradtest = gradientDescentWithReg(X_poly_, Y_, theta1, lr, epochs, lambda_=lambda_vec[i])
        error_train[i] = computeCostWithReg(X_poly_, Y_, thetanew1, lambda_=0)  # Compute training error
        error_val[i] = computeCostWithReg(X_val_poly_, Y_val_, thetanew1, lambda_=0)  # Compute validation error
        print(f"lambda = {lambda_vec[i]}:")
        print(f"train error: {error_train[i]}")
        print(f"val error: {error_val[i]}")
        print(f"J_history[-1]: {J_history[-1]}\n")

    # Plot the validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec, error_train, label='Train Error', color='blue', linestyle='-', marker='o', markersize=8, linewidth=2)
    plt.plot(lambda_vec, error_val, label='Validation Error', color='red', linestyle='--', marker='s', markersize=8, linewidth=2)
    plt.xlabel('Lambda', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Validation Curve for Different Lambda Values', fontsize=16, pad=15)
    plt.legend(fontsize=12)
    plt.grid(True, ls="--", lw=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

    # Print the validation curve errors in a table format
    print('# Lambda\t\tTrain Error\tValidation Error')
    for i in range(len(lambda_vec)):
        print('  \t%.4f\t\t%.6f\t%.6f' % (lambda_vec[i], error_train[i], error_val[i]))

    return error_train, error_val

# Compute and plot the validation curve
train_errors, val_errors = valCurve(X_poly_, Y_, X_val_poly_, Y_val_, computeCostWithReg, gradientDescentWithReg)