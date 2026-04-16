%%writefile models.py
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class HKMUBaseModel(ABC):
    """Abstraction: Abstract Base Class for Campus Traffic"""
    def __init__(self, campus_name):
        self._campus_name = campus_name
        self._a = 0.0
        self._b = 0.0
        self._c = 0.0

    @abstractmethod
    def train_with_data(self, x_values, y_values):
        """Self-study Algorithm: Quadratic/Polynomial Regression"""
        pass

    def predict(self, x):
        """Encapsulation: Formula y = ax^2 + bx + c"""
        val = (self._a * (x**2)) + (self._b * x) + self._c
        return max(0, val)

    def plot_trend(self, target_min):
        """Visualization: Graph from -30 to +30"""
        time_range = np.linspace(-30, 30, 100)
        predictions = [self.predict(t) for t in time_range]

        plt.figure(figsize=(10, 5))
        plt.plot(time_range, predictions, color='forestgreen', linewidth=2, label='Traffic Flow')
        plt.scatter([target_min], [self.predict(target_min)], color='red', s=100, zorder=5, label='Your Time')

        plt.title(f"HKMU Traffic Analysis: {self._campus_name}")
        plt.xlabel("Minutes from Class Start (0 = Peak)")
        plt.ylabel("Traffic Intensity")
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        plt.xlim(-30, 30)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.show()

class CampusQuadratic(HKMUBaseModel):
    """Inheritance: Implementation of the Polynomial Fit"""
    def train_with_data(self, x_values, y_values):
        # Self-study Algorithm: Basic Polynomial Fitting
        # y = ax^2 + c (Simplified for university level to show increase/decrease)
        peak_y = max(y_values)
        self._c = peak_y
        self._a = - (peak_y / 900) # Simple curve fit to hit 0 at +/- 30 mins
        print(f"Algorithm: Quadratic Model for {self._campus_name} is ready.")

class IOHModel(CampusQuadratic):
    def train_with_data(self, x_values, y_values):
        super().train_with_data(x_values, y_values)

class MCModel(CampusQuadratic):
    def train_with_data(self, x_values, y_values):
        super().train_with_data(x_values, y_values)

class JCCModel(CampusQuadratic):
    def train_with_data(self, x_values, y_values):
        super().train_with_data(x_values, y_values)
