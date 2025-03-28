# Import necessary libraries
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------------
# Part 1: Linear SVM on project7data1.mat
# ----------------------------------------------
def plot_linear_svm():
    """Train and plot a linear SVM on project7data1.mat dataset."""
    # Load dataset
    data = loadmat('project7data1.mat')
    X = data['X']  # Feature matrix
    Y = data['y']  # Labels
    m = X.shape[0]  # Number of samples
    print(f"Number of samples in dataset 1: {m}")

    # Separate positive and negative classes for plotting
    pos = Y[:, 0] == 1
    neg = Y[:, 0] == 0

    # Plot data points
    plt.figure(figsize=(8, 6))
    plt.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k', label='Positive')
    plt.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k', label='Negative')
    plt.grid(True)
    plt.legend()

    # Train Linear SVM
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, Y[:, 0])

    # Calculate decision boundary
    w = clf.coef_[0]    # Weights
    b = clf.intercept_[0]  # Bias
    x_points = np.linspace(0, 5, 100)  # X-axis range for plotting
    y_points = -(w[0] / w[1]) * x_points - (b / w[1])  # Decision boundary line

    # Plot decision boundary
    plt.plot(x_points, y_points, c='r', label='Decision Boundary')
    plt.title("Linear SVM on Dataset 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# ----------------------------------------------
# Part 2: RBF SVM on project7data2.mat
# ----------------------------------------------
def plot_rbf_svm():
    """Train and plot an RBF SVM on project7data2.mat dataset."""
    # Load dataset
    data = loadmat('project7data2.mat')
    X = data['X']
    Y = data['y']
    m = X.shape[0]
    print(f"Number of samples in dataset 2: {m}")

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separate positive and negative classes
    pos = Y[:, 0] == 1
    neg = Y[:, 0] == 0

    # Train RBF SVM
    clf = svm.SVC(kernel='rbf', gamma=10, C=10)
    clf.fit(X, Y[:, 0])

    # Create mesh grid for contour plot
    h = 0.01  # Step size in the mesh
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("RBF SVM on Dataset 2")
    plt.xlabel("Scaled Feature 1")
    plt.ylabel("Scaled Feature 2")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    # Hyperparameter tuning with GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    grid.fit(X, Y[:, 0])
    print(f"Best parameters for RBF SVM: {grid.best_params_}")

# ----------------------------------------------
# Part 3: RBF SVM with Validation on project7data3.mat
# ----------------------------------------------
def validate_rbf_svm():
    """Train, validate, and tune RBF SVM on project7data3.mat dataset."""
    # Load dataset
    data = loadmat('project7data3.mat')
    X_train = data['X']
    Y_train = data['y']
    X_val = data['Xval']
    Y_val = data['yval']
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Plot training data
    pos = Y_train[:, 0] == 1
    neg = Y_train[:, 0] == 0
    plt.figure(figsize=(8, 6))
    plt.plot(X_train[pos, 0], X_train[pos, 1], 'X', mec='b', ms=6, mew=1, mfc='g', label='Positive')
    plt.plot(X_train[neg, 0], X_train[neg, 1], 'o', mec='r', ms=6, mfc='y', mew=1, label='Negative')
    plt.grid(True)
    plt.title("Training Data (Dataset 3)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

    # Manual hyperparameter tuning
    gamma_selection = [0.1, 1, 10, 100, 200, 500, 1000]
    C_selection = [0.1, 1, 10, 100, 1000]
    accuracy = []

    for C_ in C_selection:
        for gamma_ in gamma_selection:
            clf = svm.SVC(kernel='rbf', gamma=gamma_, C=C_)
            clf.fit(X_train, Y_train[:, 0])
            acc = clf.score(X_val, Y_val[:, 0])
            accuracy.append(acc)
            print(f"Accuracy with C={C_}, gamma={gamma_}: {acc:.4f}")

    # Find the best parameters from manual tuning
    max_index = np.argmax(accuracy)  # Index of the maximum accuracy
    max_acc = accuracy[max_index]
    n_gamma = len(gamma_selection)
    best_C_index = max_index // n_gamma
    best_gamma_index = max_index % n_gamma
    best_C = C_selection[best_C_index]
    best_gamma = gamma_selection[best_gamma_index]
    print(f"Best manual parameters: C={best_C}, gamma={best_gamma}, Accuracy: {max_acc:.4f}")

    # Train final model with best manual parameters
    clf = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    clf.fit(X_train, Y_train[:, 0])
    print(f"Validation accuracy with best manual parameters: {clf.score(X_val, Y_val[:, 0]):.4f}")

    # Contour plot for the best model
    h = 0.01
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and training points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"RBF SVM Decision Boundary (C={best_C}, gamma={best_gamma}) on Dataset 3")
    plt.show()

    # GridSearchCV for comparison
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 1, 10, 100, 200, 500, 1000]}
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    grid.fit(X_train, Y_train[:, 0])
    print(f"Best parameters from GridSearchCV: {grid.best_params_}")

# ----------------------------------------------
# Main Execution
# ----------------------------------------------
if __name__ == "__main__":
    print("# Running Linear SVM on Dataset 1")
    plot_linear_svm()

    print("\n# Running RBF SVM on Dataset 2")
    plot_rbf_svm()

    print("\n# Running RBF SVM with Validation on Dataset 3")
    validate_rbf_svm()
