import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ANNModelPredictor:
    def __init__(self, test_in_path, model_path, scaler_input_path, scaler_output_path, predictions_output_path, graph_output_path):
        self.test_in_path = test_in_path
        self.model_path = model_path
        self.scaler_input_path = scaler_input_path
        self.scaler_output_path = scaler_output_path
        self.predictions_output_path = predictions_output_path
        self.graph_output_path = graph_output_path
        
        self.load_data()
        self.load_scalers()
        self.load_model()
        
    def load_data(self):
        self.test_input = pd.read_excel(self.test_in_path, header=None)
        self.test_input.dropna(inplace=True)
        self.test_input_values = self.test_input.iloc[:, 1:3].values
        
    def load_scalers(self):
        with open(self.scaler_input_path, 'rb') as f:
            self.scaler_input = pickle.load(f)
        with open(self.scaler_output_path, 'rb') as f:
            self.scaler_output = pickle.load(f)
        
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            model_dict = pickle.load(f)
        self.model = tf.keras.models.model_from_json(model_dict["model_config"])
        self.model.set_weights(model_dict["model_weights"])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
    def make_predictions(self):
        self.test_input_scaled = self.scaler_input.transform(self.test_input_values)
        predictions_standardized = self.model.predict(self.test_input_scaled)
        self.predictions_original = self.scaler_output.inverse_transform(predictions_standardized)
        self.predictions_original = np.maximum(self.predictions_original, 0)
        
    def plot_predictions(self):
        plt.plot(self.predictions_original, label='Predicted')
        plt.xlabel('Data Point')
        plt.ylabel('RUL')
        plt.title('ANN Predicted Remaining Useful Life (RUL)')
        plt.legend()
        plt.savefig(self.graph_output_path)
        plt.close()
        
    def save_predictions(self):
        df_predictions = pd.DataFrame(self.predictions_original, columns=['Predictions'])
        df_predictions.to_excel(self.predictions_output_path, index=False)
        
    def run(self):
        self.make_predictions()
        self.plot_predictions()
        self.save_predictions()
        

