# train_and_save_svm_model.py

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# File paths
Train_in_path = r'./model/input.xlsx'
Output_path = r'./model/output.xlsx'
model_path = r'./model/svm_model.pkl'
scaler_path = r'./model/svm_scaler.pkl'

# Ensure the output directory exists
output_dir = os.path.dirname(model_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read Excel files without header
Train_input = pd.read_excel(Train_in_path, header=None)
Output = pd.read_excel(Output_path, header=None)

# Extract input and output columns
X_train = Train_input.values
y_train = Output.iloc[:, 0].astype(float).values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1]
}

svm_model = SVR()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_svm_model = grid_search.best_estimator_

# Save the scaler using pickle
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Save the model using pickle
with open(model_path, 'wb') as f:
    pickle.dump(best_svm_model, f)

# Calculate predictions for training data
y_train_pred = best_svm_model.predict(X_train_scaled)

# Calculate RMSE and R-squared
rmse = mean_squared_error(y_train, y_train_pred, squared=False)
r2 = r2_score(y_train, y_train_pred)

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
