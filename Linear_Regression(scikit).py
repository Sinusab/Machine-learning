import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('project2-data2.csv')

# Select features (first two columns) and target (third column)
X = data.values[:, 0:2]
Y = data.values[:, 2]

# Standardize the features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Initialize the model
model = SGDRegressor(
    loss='squared_error',  # Use squared error loss for regression
    fit_intercept=True,    # Include the intercept in the model
    max_iter=5000,         # Maximum number of iterations
    learning_rate='constant',  # Use a constant learning rate
    eta0=0.001             # Initial learning rate
)

# Train the model
model.fit(X_norm, Y)

# New data for prediction
new_data = np.array([[1600, 3]])

# Standardize the new data using the same scaler
new_data_norm = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_norm)

# Output the predicted value
print("Predicted value:", prediction[0])
