import sklearn
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = loadmat('project4data1.mat')
X = data['X']
Y = data['y']

# Display dataset information
print("Shape of X:", X.shape)
print("Shape of y:", Y.shape)
print("Sample X:", X[:5])  # Display first 5 samples
print("Sample y:", Y[:5])

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

# Display real vs predicted labels
print("Real labels:", Y_test[:10])
print("Predicted labels:", y_pred[:10])

# Print classification report
print(classification_report(Y_test, y_pred))
