import gzip
import pickle
import matplotlib.pyplot as plt
import os

# Correct path to data directory
data_path = os.path.join("data", "data.pkl.gzip")

try:
    # Load the dataset
    with gzip.open(data_path, 'rb') as f:
        data = pickle.load(f)

    actions = data['action']

    # Separate steering and throttle (assumes 2D action)
    steering = [a[0] for a in actions]
    throttle = [a[1] for a in actions]

    # Plot the actions
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(steering, label='Steering', color='orange')
    plt.ylabel('Steering')
    plt.title('Driving Actions Analysis')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(throttle, label='Throttle', color='blue')
    plt.ylabel('Throttle')
    plt.xlabel('Timestep')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Could not find data file at: {data_path}")
    print("Please make sure to:")
    print("1. Run drive_manually.py with --collect_data flag first")
    print("2. Check if the data file exists in the 'data' directory")
    print("\nTo collect data, run:")
    print("python drive_manually.py --collect_data")

print("Min throttle:", min(throttle))
print("Max throttle:", max(throttle))
print("Mean throttle:", sum(throttle) / len(throttle))
