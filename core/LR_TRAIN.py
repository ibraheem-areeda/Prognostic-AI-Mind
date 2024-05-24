# train_and_save_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# File paths
Train_in_path = r'./model/input.xlsx'
Output_path = r'./model/output.xlsx'
model_path = r'./model/linear_model.pkl'
scaler_path = r'./model/scaler.pkl'

# Ensure the output directory exists
output_dir = os.path.dirname(model_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read Excel files without header
Train_input = pd.read_excel(Train_in_path, header=None)
Output = pd.read_excel(Output_path, header=None)

# Extract input and output values
Train_input_values = Train_input.values
Output_values = Output.iloc[:, 0].astype(float).values  # Ensure output values are floats

# Standardize the input data
scaler = StandardScaler()
Train_input_scaled = scaler.fit_transform(Train_input_values)

# Train the linear regression model on the scaled data
mdl = LinearRegression().fit(Train_input_scaled, Output_values)

# Calculate predictions for training data
train_predictions = mdl.predict(Train_input_scaled)

# Calculate RMSE and R-squared
rmse = mean_squared_error(Output_values, train_predictions, squared=False)
r2 = r2_score(Output_values, train_predictions)

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Save the scaler using pickle
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Save the model using pickle
with open(model_path, 'wb') as f:
    pickle.dump(mdl, f)