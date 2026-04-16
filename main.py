# main.py
from models import IOHModel, MCModel, JCCModel
from data_config import CAMPUS_DATA

def main():
    # Polymorphism: Different objects handled via a uniform interface
    campus_registry = {
        '1': IOHModel(CAMPUS_DATA['1']['name']),
        '2': MCModel(CAMPUS_DATA['2']['name']),
        '3': JCCModel(CAMPUS_DATA['3']['name'])
    }

    while True:
        print("\n--- HKMU Traffic Monitor ---")
        print("1: IOH | 2: MC | 3: JCC | 4: Exit")
        choice = input("Select Campus (1-4): ")

        if choice == '4': break
        if choice in campus_registry:
            model = campus_registry[choice]
            model.train_with_data(CAMPUS_DATA[choice]['peak'])
            
            while True:
                try:
                    user_min = int(input("Enter time (-30 to 30 mins): "))
                    if -30 <= user_min <= 30: break
                    print("Error: Input out of range.")
                except ValueError: print("Error: Enter a number.")

            print(f"Prediction for {model._campus_name}: {model.predict(user_min):.2f}")
            if input("Show graph? (y/n): ").lower() == 'y':
                model.plot_trend(user_min)
        else:
            print("Invalid Selection.")

if __name__ == "__main__":
    main()
