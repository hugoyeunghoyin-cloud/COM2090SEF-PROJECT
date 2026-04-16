# campus_models.py
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class HKMUBaseModel(ABC):
    """Abstraction: Template for Campus Traffic Models"""
    def __init__(self, campus_name):
        self._campus_name = campus_name # Encapsulation: Protected attribute
        self._a = 0.0 # Curvature coefficient
        self._c = 0.0 # Peak intensity coefficient

    @abstractmethod
    def train_with_data(self, peak_val):
        """Self-study Algorithm Interface: Quadratic Regression"""
        pass

    def predict(self, x):
        """Encapsulation: Formula y = ax^2 + c"""
        val = (self._a * (x**2)) + self._c
        return max(0, val)

    def plot_trend(self, target_min):
        """Visualization logic for Task 1"""
        time_range = np.linspace(-30, 30, 100)
        predictions = [self.predict(t) for t in time_range]
        plt.figure(figsize=(8, 4))
        plt.plot(time_range, predictions, color='teal', label='Predicted Flow')
        plt.scatter([target_min], [self.predict(target_min)], color='red', zorder=5)
        plt.title(f"Traffic Analysis: {self._campus_name}")
        plt.xlabel("Minutes from Start (0 = Peak)")
        plt.ylabel("Intensity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

class CampusQuadratic(HKMUBaseModel):
    """Inheritance: Logic for the Polynomial/Quadratic fit"""
    def train_with_data(self, peak_val):
        # Algorithm: Quadratic curve y = ax^2 + c
        self._c = peak_val
        self._a = - (peak_val / 900) 
        print(f"Model for {self._campus_name} is ready.")

class IOHModel(CampusQuadratic): pass
class MCModel(CampusQuadratic): pass
class JCCModel(CampusQuadratic): pass
