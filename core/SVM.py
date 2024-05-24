import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

class SVMPredictor:
    def __init__(self, test_in_path, predictions_output_path, graph_output_path, model_path, scaler_path):
        self.test_in_path = test_in_path
        self.predictions_output_path = predictions_output_path
        self.graph_output_path = graph_output_path
        self.model_path = model_path
        self.scaler_path = scaler_path

    def load_data(self):
        # Read Excel file without header
        test_input = pd.read_excel(self.test_in_path, header=None)
        # Drop rows with NaN values in test input data
        test_input.dropna(inplace=True)
        # Extract columns 2 and 3 as input values
        self.test_input_values = test_input.iloc[:, 1:3].values

    def load_scaler(self):
        # Load the scaler using pickle
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def load_model(self):
        # Load the model using pickle
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self):
        # Standardize the test input data
        test_input_scaled = self.scaler.transform(self.test_input_values)
        # Make predictions on scaled test data
        self.predictions = self.model.predict(test_input_scaled)
        # Set negative predictions to zero
        self.predictions = np.maximum(self.predictions, 0)

    def save_predictions(self):
        # Save predictions to Excel
        df_predictions = pd.DataFrame(self.predictions, columns=['Predictions'])
        df_predictions.to_excel(self.predictions_output_path, index=False)

    def plot_predictions(self):
        # Plot predictions
        plt.plot(self.predictions, label='Predicted')
        plt.xlabel('Data Point')
        plt.ylabel('RUL')
        plt.title('SVM Predicted Remaining Useful Life (RUL)')
        plt.legend()
        plt.savefig(self.graph_output_path)  # Save the plot as a PNG file
        plt.close()

    def run(self):
        self.load_data()
        self.load_scaler()
        self.load_model()
        self.predict()
        self.save_predictions()
        self.plot_predictions()


