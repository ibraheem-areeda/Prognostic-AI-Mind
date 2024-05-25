import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


class ANNPredictor:
    def __init__(self, test_in_path, predictions_output_path, graph_output_path, model_path, scaler_input_path, scaler_output_path):
        self.test_in_path = test_in_path
        self.predictions_output_path = predictions_output_path
        self.graph_output_path = graph_output_path
        self.model_path = model_path
        self.scaler_input_path = scaler_input_path
        self.scaler_output_path = scaler_output_path

    def load_data(self):
        # Read Excel file without header
        test_input = pd.read_excel(self.test_in_path, header=None)
        # Drop rows with NaN values in test input data
        test_input.dropna(inplace=True)
        # Extract columns 2 and 3 as input values
        self.test_input_values = test_input.iloc[:, 1:3].values

    def load_scalers(self):
        # Load the scalers using pickle
        with open(self.scaler_input_path, 'rb') as f:
            self.scaler_input = pickle.load(f)
        with open(self.scaler_output_path, 'rb') as f:
            self.scaler_output = pickle.load(f)

    def load_model(self):
        # Load the model
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self):
        # Standardize the test input data
        test_input_scaled = self.scaler_input.transform(self.test_input_values)
        # Make predictions on scaled test data
        predictions_standardized = self.model.predict(test_input_scaled)
        # Inverse transform the predictions to get them back to the original scale
        self.predictions_original = self.scaler_output.inverse_transform(predictions_standardized)
        # Set negative predictions to zero
        self.predictions_original = np.maximum(self.predictions_original, 0)

    def save_predictions(self):
        # Save predictions to Excel
        df_predictions = pd.DataFrame(self.predictions_original, columns=['Predictions'])
        df_predictions.to_excel(self.predictions_output_path, index=False)

    def plot_predictions(self):
        # Plot predictions
        plt.plot(self.predictions_original, label='Predicted')
        plt.xlabel('Data Point')
        plt.ylabel('RUL')
        plt.title('ANN Predicted Remaining Useful Life (RUL)')
        plt.legend()
        plt.savefig(self.graph_output_path)  # Save the plot as a PNG file
        plt.close()

    def run(self):
        self.load_data()
        self.load_scalers()
        self.load_model()
        self.predict()
        self.save_predictions()
        self.plot_predictions()

