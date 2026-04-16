from models import IOHModel, MCModel, JCCModel

def main():
    # Data provided per 10 min: -30, -20, -10, 0, 10, 20, 30
    time_steps = [-30, -20, -10, 0, 10, 20, 30]

    campuses = {
        '1': (IOHModel("IOH Main Campus"), [10, 15, 25, 30, 23, 15, 10]),
        '2': (MCModel("MC Mong Kok"), [5, 8, 15, 25, 20, 5, 3]),
        '3': (JCCModel("JCC Jockey Club"), [6, 9, 18, 28, 25, 10, 5])
    }

    print("\n" + "="*45)
    print("   HKMU MODULAR TRAFFIC SYSTEM")
    print("="*45)
    print("1. IOH (Main Campus)")
    print("2. MC (Mong Kok)")
    print("3. JCC (Jockey Club)")
    print("4. Exit")

    choice = input("\nSelect building (1-3) or 4 to Exit: ")

    if choice == '4':
        print("System Exited.")
        return

    if choice in campuses:
        model, traffic_data = campuses[choice]
        model.train_with_data(time_steps, traffic_data)

        while True:
            try:
                user_min = int(input("Enter minutes (-30 to 30): "))
                if -30 <= user_min <= 30:
                    break
                print("Error: Input must be between -30 and 30.")
            except ValueError:
                print("Error: Please enter a whole number.")

        print(f"\n>>> {model._campus_name} | Predicted Traffic: {model.predict(user_min):.2f}")

        if input("See trend graph? (y/n): ").lower() == 'y':
            model.plot_trend(user_min)

        print("\nProgram finished after analysis.")
    else:
        print("Invalid Selection.")

if __name__ == "__main__":
    main()
