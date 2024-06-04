import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# File paths
scaler_input_path = r'./model/ann_scaler_input.pkl'
scaler_output_path = r'./model/ann_scaler_output.pkl'
input_file = r'./model/input.xlsx'
output_file = r'./model/output.xlsx'
model_path = r'./model/ann_model.pkl'


# Step 1: Read data from Excel files
input_data = pd.read_excel(input_file, header=None)
output_data = pd.read_excel(output_file, header=None)

# Step 2: Standardize the data using Z-score normalization
scaler_input = StandardScaler()
scaler_output = StandardScaler()

input_standardized = scaler_input.fit_transform(input_data)
output_standardized = scaler_output.fit_transform(output_data)

# Step 3: Split the data into training and temporary sets (80% training, 20% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(input_standardized, output_standardized, test_size=0.2, random_state=42, shuffle=True)

# Step 4: Split the temporary set into validation and test sets (50% each of the remaining 20%)
# This results in 10% of the original data for validation and 10% for testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# Step 5: Define the neural network model with two hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Step 6: Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Step 7: Train the model with 50 epochs and include validation data
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Step 8: Evaluate the model on the test data
test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test RMSE: {test_rmse}")

# Calculate RMSE and R-squared for the training data
y_train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training RMSE: {train_rmse}")
print(f"Training R-squared: {train_r2}")

# Calculate R-squared for the test data
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test R-squared: {test_r2}")

# Save the scalers using pickle
with open(scaler_input_path, 'wb') as f:
    pickle.dump(scaler_input, f)

with open(scaler_output_path, 'wb') as f:
    pickle.dump(scaler_output, f)

# Save the model architecture and weights as a dictionary using pickle
model_dict = {
    "model_config": model.to_json(),
    "model_weights": model.get_weights()
}
with open(model_path, 'wb') as f:
    pickle.dump(model_dict, f)