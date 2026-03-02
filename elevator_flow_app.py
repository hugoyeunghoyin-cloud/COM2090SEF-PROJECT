"""
Elevator Flow Application
User interface that calls the ElevatorFlowModel.
Includes lab example functions as static methods.
"""

import math
from elevator_flow_model import ElevatorFlowModel


class ElevatorFlowApp:
    def __init__(self):
        self.model = ElevatorFlowModel()

    @staticmethod
    def volume_of_sphere(radius):
        """Calculate volume of a sphere (from Lab 3)."""
        return round((4.0/3.0) * math.pi * radius**3, 2)

    @staticmethod
    def area_of_triangle(length, height):
        """Calculate area of a triangle (from Lab 3)."""
        return (length * height) / 2.0

    @staticmethod
    def quadratic_roots(a, b, c):
        """Find roots of quadratic equation (from Lab 3)."""
        if a == 0:
            raise ValueError("Not a quadratic: a must be non-zero")
        d = b**2 - 4*a*c
        if d < 0:
            raise ValueError("No real roots (negative discriminant)")
        root1 = (-b + math.sqrt(d)) / (2*a)
        root2 = (-b - math.sqrt(d)) / (2*a)
        return root1, root2

    def run_full_analysis(self):
        """Generate data, train, predict, evaluate, and plot."""
        print("\n--- Generating Data ---")
        data = self.model.generate_data()
        print("Data sample:")
        print(data.head())
        print(f"\nTime range: {data['time_min'].min()} to {data['time_min'].max()} minutes")
        print(f"Flow range: {data['passenger_flow'].min():.1f} to {data['passenger_flow'].max():.1f} people/min")
        print(f"Threshold (average flow): {self.model.threshold:.2f}")

        print("\n--- Training Model ---")
        model = self.model.train()
        print(f"Intercept (β₀): {model.intercept_:.2f}")
        print(f"Coefficient for time (β₁): {model.coef_[0]:.2f}")

        # Get user time
        try:
            user_time = float(input("\nEnter time relative to lesson start (e.g., -5, 0, 10): "))
        except ValueError:
            user_time = 0
            print("Invalid input, using time 0.")

        predicted = self.model.predict(user_time)
        print(f"Predicted flow at time {user_time}: {predicted:.2f} people/min")

        if predicted >= self.model.threshold:
            print("🚶‍♂️🚶‍♀️🚶 High flow: A lot of people expected!")
        else:
            print("🙂 Low to moderate flow.")

        print(f"\nEquation: passenger_flow = {model.intercept_:.2f} + ({model.coef_[0]:.2f}) * time_min")

        mse, r2 = self.model.evaluate()
        print(f"\nMean Squared Error (MSE): {mse:.2f}")
        print(f"R² score: {r2:.4f}")

        print("\n--- Plotting ---")
        self.model.plot()
        input("\nPress Enter to continue...")

    def demo_lab_functions(self):
        """Demonstrate the lab example functions."""
        print("\n--- Lab Functions Demo ---")
        print(f"Volume of sphere with radius 5: {self.volume_of_sphere(5)}")
        print(f"Area of triangle (length=3, height=4): {self.area_of_triangle(3, 4)}")
        try:
            r1, r2 = self.quadratic_roots(1, -3, 2)
            print(f"Quadratic roots of x² - 3x + 2: {r1}, {r2}")
        except ValueError as e:
            print(e)
        input("\nPress Enter to continue...")

    def custom_prediction(self):
        """Allow user to input a time and see prediction (without re‑generating)."""
        print("\n--- Custom Prediction ---")
        # Train model if not already done
        if self.model.model is None:
            print("Training model first...")
            self.model.train()

        # Show model info
        print(f"Model coefficients:")
        print(f"Intercept (β₀): {self.model.model.intercept_:.2f}")
        print(f"Coefficient for time (β₁): {self.model.model.coef_[0]:.2f}")
        print(f"Threshold (average flow): {self.model.threshold:.2f}")

        try:
            user_time = float(input("\nEnter time relative to lesson start (e.g., -5, 0, 10): "))
        except ValueError:
            user_time = 0
            print("Invalid input, using time 0.")

        predicted = self.model.predict(user_time)
        print(f"Predicted flow at time {user_time}: {predicted:.2f} people/min")

        if predicted >= self.model.threshold:
            print("🚶‍♂️🚶‍♀️🚶 High flow: A lot of people expected!")
        else:
            print("🙂 Low to moderate flow.")
        input("\nPress Enter to continue...")

    def run(self):
        """Main menu loop."""
        while True:
            print("\n" + "="*40)
            print("   ELEVATOR PASSENGER FLOW PREDICTION")
            print("="*40)
            print("1. Run full analysis (generate, train, predict, evaluate, plot)")
            print("2. Demonstrate lab functions")
            print("3. Make a custom prediction (enter time)")
            print("4. Exit")
            print("\nHint: Try times near 0 (e.g., -5, 0, 5) to see high flow.")
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                self.run_full_analysis()
            elif choice == '2':
                self.demo_lab_functions()
            elif choice == '3':
                self.custom_prediction()
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice, please try again.")
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    app = ElevatorFlowApp()
    app.run()