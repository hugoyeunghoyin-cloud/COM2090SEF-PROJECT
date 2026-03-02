"""
Elevator Flow Model Class
Handles data generation, training, prediction, evaluation, and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class ElevatorFlowModel:
    def __init__(self, seed=42):
        self.seed = seed
        self.data = None
        self.model = None
        self.X = None
        self.y = None
        self.threshold = None  # Will be set as average flow

    def generate_data(self):
        """Generate synthetic elevator passenger flow data."""
        np.random.seed(self.seed)
        time = np.arange(-15, 16, 1)  # -15 to +15 minutes
        # Symmetric quadratic peaked at time=0
        flow_true = 60 - 0.2 * time**2
        flow = flow_true + np.random.normal(0, 5, size=len(time))
        self.data = pd.DataFrame({'time_min': time, 'passenger_flow': flow})
        # Set threshold as average flow
        self.threshold = self.data['passenger_flow'].mean()
        return self.data

    def train(self):
        """Train a linear regression model on the data."""
        if self.data is None:
            self.generate_data()
        self.X = self.data[['time_min']]
        self.y = self.data['passenger_flow']
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        return self.model

    def predict(self, time_input):
        """Predict flow for a given time (minutes relative to lesson start)."""
        if self.model is None:
            self.train()
        return self.model.predict([[time_input]])[0]

    def evaluate(self):
        """Compute MSE and R²."""
        if self.model is None:
            self.train()
        predictions = self.model.predict(self.X)
        mse = mean_squared_error(self.y, predictions)
        r2 = self.model.score(self.X, self.y)
        return mse, r2

    def plot(self):
        """Plot the data and regression line with threshold = average flow."""
        if self.model is None:
            self.train()
        time_smooth = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)
        flow_smooth = self.model.predict(time_smooth)

        plt.figure(figsize=(10,6))
        plt.scatter(self.data['time_min'], self.data['passenger_flow'],
                    alpha=0.7, label='Observed flow')
        plt.plot(time_smooth, flow_smooth, color='red', linewidth=2,
                 label='Regression line')
        plt.axvline(x=0, color='gray', linestyle='--',
                    label='Lesson start (10:00)')
        plt.axhline(y=self.threshold, color='orange', linestyle=':',
                    label=f'Threshold (avg flow = {self.threshold:.2f})')
        plt.xlabel('Time relative to lesson start (minutes)')
        plt.ylabel('Passenger flow (people/min)')
        plt.title('HKMU Elevator Passenger Flow Near Lesson Time')
        plt.legend()
        plt.grid(True)
        plt.show()